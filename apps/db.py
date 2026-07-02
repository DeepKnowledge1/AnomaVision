"""
db.py
PostgreSQL persistence for prediction records.
Pool created once at startup (api.py lifespan) and reused per request.
Can be fully disabled via DB_ENABLED=0.
"""
import os
import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://anomavision:anomavision@postgres-svc:5432/anomavision"
)
DB_ENABLED = os.getenv("DB_ENABLED", "1") == "1"

_pool: asyncpg.Pool | None = None

async def init_pool():
    """
    Initialize DB pool and schema.
    Safe no-op if DB is disabled or connection fails.
    """
    global _pool, DB_ENABLED
    if not DB_ENABLED:
        print("[db] disabled via DB_ENABLED=0")
        return

    try:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id              BIGSERIAL PRIMARY KEY,
                    image_filename  TEXT NOT NULL,
                    original_path   TEXT,
                    heatmap_path    TEXT,
                    boundary_path   TEXT,
                    prediction      TEXT NOT NULL CHECK (prediction IN ('Normal', 'Defective')),
                    anomaly_score   REAL NOT NULL,
                    model_version   TEXT NOT NULL,
                    processing_ms   INTEGER NOT NULL,
                    user_id         TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                ON predictions (created_at DESC);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_prediction
                ON predictions (prediction);
            """)
        print("[db] pool initialized")

    except Exception as e:
        # ANSI color codes
        ORANGE = "\033[38;5;208m"   # Vibrant Orange
        RESET = "\033[0m"           # Resets color back to default
        BOLD = "\033[1m"            # Bold text
        DIM = "\033[2m"             # Dimmed text for commands

        print("\n" + ORANGE + "="*70)
        print(f"{BOLD}⚠️  WARNING: Failed to connect to PostgreSQL database!{RESET}{ORANGE}")
        print(f"Error: {e}")
        print("-" * 70)
        print("The API will continue running, but prediction history will NOT be saved.")
        print("To enable history, run the following commands in your terminal:")
        print()
        print(f"{BOLD}1. Start PostgreSQL (Docker):{RESET}{ORANGE}")
        print(f"{DIM}docker run -d --name anomavision-postgres `")
        print("  -e POSTGRES_USER=anomavision `")
        print("  -e POSTGRES_PASSWORD=anomavision `")
        print("  -e POSTGRES_DB=anomavision `")
        print("  -p 5432:5432 `")
        print(f"  postgres:16-alpine{RESET}{ORANGE}")
        print()
        print(f"{BOLD}2. Set the Database URL (PowerShell):{RESET}{ORANGE}")
        print(f'{DIM}$env:DATABASE_URL="postgresql://anomavision:anomavision@127.0.0.1:5432/anomavision"{RESET}{ORANGE}')
        print()
        print(f"{BOLD}3. Restart the API:{RESET}{ORANGE}")
        print(f"{DIM}python apps\\api.py{RESET}{ORANGE}")
        print("="*70 + RESET + "\n")

        # Gracefully disable DB features so the app doesn't crash later
        DB_ENABLED = False
        _pool = None

    # except Exception as e:
    #     # 256-color ANSI codes for true, vibrant colors
    #     ORANGE = "\033[38;5;208m"   # Vibrant Orange (Best for Warnings)
    #     BRIGHT_RED = "\033[91m"     # Bright Red (Best for Errors/Alarms)
    #     RESET = "\033[0m"           # Resets back to normal
    #     BOLD = "\033[1m"            # Bold text

    #     # --- Change the variable below to BRIGHT_RED if you prefer red ---
    #     COLOR = ORANGE

    #     print("\n" + COLOR + "="*60)
    #     print("[db] ⚠️⚠️⚠️  WARNING: Failed to connect to PostgreSQL database!⚠️⚠️⚠️")
    #     print(f"[db] Error: {e}")
    #     print("[db] The API will continue running, but prediction history")
    #     print("[db] will NOT be saved. To enable history, start PostgreSQL")
    #     print("[db] and ensure DATABASE_URL is correct.")
    #     print("[api] Exact instructions can be found in the comments at the bottom of your api.py file.")
    #     print("="*60 + RESET + "\n")
    #     DB_ENABLED = False
    #     _pool = None


async def insert_prediction(
    image_filename: str,
    original_path: str | None,
    heatmap_path: str | None,
    boundary_path: str | None,
    prediction: str,
    anomaly_score: float,
    model_version: str,
    processing_ms: int,
    user_id: str | None = None,
) -> int | None:
    """
    Insert prediction record.
    Returns inserted ID, or None if DB disabled.
    """
    if not DB_ENABLED or _pool is None:
        return None

    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO predictions
                (image_filename, original_path, heatmap_path, boundary_path,
                 prediction, anomaly_score, model_version, processing_ms, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id;
            """,
            image_filename,
            original_path,
            heatmap_path,
            boundary_path,
            prediction,
            anomaly_score,
            model_version,
            processing_ms,
            user_id,
        )

async def get_predictions_count() -> int:
    """Get total number of predictions."""
    if not DB_ENABLED or _pool is None:
        return 0

    async with _pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM predictions")

async def get_predictions_paginated(limit: int = 10, offset: int = 0) -> list:
    """
    Get predictions with pagination.
    Returns list of prediction records ordered by created_at DESC.
    """
    if not DB_ENABLED or _pool is None:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                id, image_filename, original_path, heatmap_path, boundary_path,
                prediction, anomaly_score, model_version, processing_ms,
                user_id, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)

        return [dict(row) for row in rows]

async def get_prediction_by_id(prediction_id: int) -> dict | None:
    """Get a specific prediction by ID."""
    if not DB_ENABLED or _pool is None:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                id, image_filename, original_path, heatmap_path, boundary_path,
                prediction, anomaly_score, model_version, processing_ms,
                user_id, created_at
            FROM predictions
            WHERE id = $1
        """, prediction_id)

        return dict(row) if row else None


async def get_next_prediction_id(current_id: int) -> int | None:
    """Get the next prediction ID (older than current - lower ID)."""
    if not DB_ENABLED or _pool is None:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM predictions
            WHERE id < $1
            ORDER BY id DESC
            LIMIT 1
        """, current_id)

        return row["id"] if row else None

async def get_previous_prediction_id(current_id: int) -> int | None:
    """Get the previous prediction ID (newer than current - higher ID)."""
    if not DB_ENABLED or _pool is None:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM predictions
            WHERE id > $1
            ORDER BY id ASC
            LIMIT 1
        """, current_id)

        return row["id"] if row else None

async def close_pool():
    """
    Optional cleanup hook (can be called on shutdown).
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

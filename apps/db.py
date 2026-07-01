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
    Safe no-op if DB is disabled.
    """
    global _pool
    if not DB_ENABLED:
        print("[db] disabled via DB_ENABLED=0")
        return

    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

    async with _pool.acquire() as conn:
        # Updated schema to include paths for all 3 images
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

async def close_pool():
    """
    Optional cleanup hook (can be called on shutdown).
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

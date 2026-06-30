"""
db.py
-----
PostgreSQL persistence for prediction records. Pool created once at
startup (api.py's lifespan) and reused for every request.
"""

# docker exec -it anomavision-postgres psql -U anomavision -d anomavision -c "SELECT * FROM predictions;"

import os

import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://anomavision:anomavision@postgres-svc:5432/anomavision"
)

_pool: asyncpg.Pool | None = None


async def init_pool():
    global _pool
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              BIGSERIAL PRIMARY KEY,
                image_filename  TEXT NOT NULL,
                image_path      TEXT,
                prediction      TEXT NOT NULL CHECK (prediction IN ('Normal', 'Defective')),
                anomaly_score   REAL NOT NULL,
                model_version   TEXT NOT NULL,
                processing_ms   INTEGER NOT NULL,
                user_id         TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions (prediction);"
        )


async def insert_prediction(
    image_filename: str,
    image_path: str | None,
    prediction: str,
    anomaly_score: float,
    model_version: str,
    processing_ms: int,
    user_id: str | None = None,
) -> int:
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO predictions
                (image_filename, image_path, prediction, anomaly_score, model_version, processing_ms, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id;
            """,
            image_filename,
            image_path,
            prediction,
            anomaly_score,
            model_version,
            processing_ms,
            user_id,
        )

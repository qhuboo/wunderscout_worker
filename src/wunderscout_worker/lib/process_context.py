from contextlib import asynccontextmanager
import aioboto3
import redis.asyncio as aioredis
from psycopg_pool import AsyncConnectionPool
import os


@asynccontextmanager
async def process_context():
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            raise RuntimeError(
                "WORKER[process_context][RuntimeError]: DATABASE_URL env variable is required."
            )
        print(f"WORKER[process_context]: Creating pg pool with URL: {database_url}")
        pool = AsyncConnectionPool(database_url)
        await pool.open()
        print("WORKER[process_context]: PG pool opened.")
        await pool.wait()
        print("WORKER[process_context]: PG pool ready.")

        redis_url = os.getenv("REDIS_URL")
        if redis_url is None:
            raise RuntimeError(
                "WORKER[process_context][RuntimeError]: REDIS_URL env variable is required."
            )
        redis = aioredis.ConnectionPool.from_url(redis_url)

        try:
            yield {"s3": s3, "pg": pool, "redis": redis}
        finally:
            await pool.close()
            await redis.aclose()


@asynccontextmanager
async def get_redis(ctx):
    client = aioredis.Redis(connection_pool=ctx["redis"])
    try:
        yield client
    finally:
        await client.aclose()

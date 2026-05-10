"""Разовый скрипт для переноса данных glass.csv в таблицу glass."""
import pandas as pd
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from database import settings

async def transfer():
    df = pd.read_csv("data/glass.csv")
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: df.to_sql(
                "glass", sync_conn, if_exists="append", index=False
            )
        )
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(transfer())
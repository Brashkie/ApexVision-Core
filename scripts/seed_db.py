"""ApexVision-Core — Seed initial DB"""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.db.session import engine, Base

async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database seeded OK")

asyncio.run(seed())

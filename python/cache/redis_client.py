"""ApexVision-Core — Async Redis Client"""
import redis.asyncio as aioredis
from loguru import logger
from python.config import settings

class RedisClient:
    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def connect(self):
        self._client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await self._client.ping()
        logger.info("Redis connected")

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    async def ping(self):
        return await self._client.ping()

    async def get(self, key: str):
        return await self._client.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        return await self._client.setex(key, ttl, value)

    async def delete(self, key: str):
        return await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

redis_client = RedisClient()

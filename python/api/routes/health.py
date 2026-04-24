"""ApexVision-Core — Health & Readiness Router"""
import time
from datetime import datetime, timezone
from fastapi import APIRouter
from loguru import logger
from python.config import settings

router = APIRouter()
_start_time = time.time()

@router.get("/", summary="Basic health check")
async def health() -> dict:
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.VERSION}

@router.get("/live", summary="Liveness probe")
async def liveness() -> dict:
    return {"status": "alive"}

@router.get("/ready", summary="Readiness probe — checks Redis and DB")
async def readiness() -> dict:
    checks: dict[str, str] = {}
    all_ok = True
    try:
        from python.cache.redis_client import redis_client
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_ok = False
    try:
        from python.db.session import engine
        import sqlalchemy
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
        all_ok = False
    return {"status": "ready" if all_ok else "degraded", "checks": checks}

@router.get("/status", summary="Full system status")
async def system_status() -> dict:
    uptime = int(time.time() - _start_time)
    try:
        from python.core.model_registry import ModelRegistry
        model_info = ModelRegistry.status()
    except Exception:
        model_info = {}
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": settings.DEVICE,
        "models": model_info,
        "debug": settings.DEBUG,
    }

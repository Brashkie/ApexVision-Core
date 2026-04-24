"""
ApexVision-Core — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import make_asgi_app

from python.api.routes import vision, batch, health, models, stream
from python.api.middleware.auth import APIKeyMiddleware
from python.api.middleware.rate_limit import RateLimitMiddleware
from python.api.middleware.telemetry import TelemetryMiddleware
from python.cache.redis_client import redis_client
from python.db.session import engine, Base
from python.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("ApexVision-Core starting...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready")

    await redis_client.connect()
    logger.info("Redis ready")

    from python.core.model_registry import ModelRegistry
    await ModelRegistry.warmup()
    logger.info("Models ready")

    yield

    logger.info("Shutting down...")
    await redis_client.disconnect()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ApexVision-Core",
        description="Ultra Vision API Platform — More powerful than Google Cloud Vision",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "vision",  "description": "Core vision analysis endpoints"},
            {"name": "batch",   "description": "Async batch processing jobs"},
            {"name": "models",  "description": "Model registry management"},
            {"name": "stream",  "description": "WebSocket real-time streaming"},
            {"name": "health",  "description": "Health, metrics, readiness probes"},
        ],
    )

    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyMiddleware, exclude_paths=[
        "/health", "/health/ready", "/health/live",
        "/docs", "/redoc", "/openapi.json", "/metrics",
    ])

    V = f"/api/{settings.API_VERSION}"
    app.include_router(health.router,  prefix="/health",       tags=["health"])
    app.include_router(vision.router,  prefix=f"{V}/vision",   tags=["vision"])
    app.include_router(batch.router,   prefix=f"{V}/batch",    tags=["batch"])
    app.include_router(models.router,  prefix=f"{V}/models",   tags=["models"])
    app.include_router(stream.router,  prefix=f"{V}/stream",   tags=["stream"])

    app.mount("/metrics", make_asgi_app())

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
            "request_id": request.headers.get("X-Request-ID"),
        })

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "python.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )

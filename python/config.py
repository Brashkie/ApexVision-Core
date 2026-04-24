"""
ApexVision-Core — Configuration
Todas las settings se cargan desde variables de entorno / .env
"""
from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "ApexVision-Core"
    VERSION: str = "2.0.0"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    API_KEY_HEADER: str = "X-ApexVision-Key"
    MASTER_API_KEY: str = Field(...)
    CORS_ORIGINS: list[str] = ["*"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://apex:apex@localhost:5432/apexvision"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    REDIS_JOB_TTL: int = 86400

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_MAX_RETRIES: int = 3
    CELERY_TASK_TIMEOUT: int = 300

    # Storage
    STORAGE_BACKEND: Literal["local", "s3", "gcs"] = "local"
    LOCAL_STORAGE_PATH: str = "./data"
    DELTA_LAKE_PATH: str = "./data/delta"
    PARQUET_PATH: str = "./data/parquet"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ML Models
    MODELS_PATH: str = "./models"
    DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    YOLO_MODEL: str = "yolov11n.pt"
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    SAM_CHECKPOINT: str = "sam_vit_b_01ec64.pth"
    ONNX_PROVIDERS: list[str] = ["CPUExecutionProvider"]

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    # Observability
    OTLP_ENDPOINT: str | None = None
    PROMETHEUS_ENABLED: bool = True

    @field_validator("DEVICE", mode="before")
    @classmethod
    def auto_detect_device(cls, v: str) -> str:
        if v != "cpu":
            return v
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

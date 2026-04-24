"""ApexVision-Core — Model Registry"""
from loguru import logger
from python.config import settings

class ModelRegistry:
    _models: dict = {}

    @classmethod
    async def warmup(cls):
        logger.info(f"Warming up models on device: {settings.DEVICE}")
        try:
            from ultralytics import YOLO
            path = f"{settings.MODELS_PATH}/{settings.YOLO_MODEL}"
            cls._models["yolo"] = YOLO(path)
            logger.info("YOLO loaded")
        except Exception as e:
            logger.warning(f"YOLO not loaded (will load on first request): {e}")

    @classmethod
    def get(cls, name: str):
        return cls._models.get(name)

    @classmethod
    def status(cls) -> dict:
        return {k: "loaded" for k in cls._models}

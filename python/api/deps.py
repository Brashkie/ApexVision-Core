"""ApexVision-Core — FastAPI Dependencies"""
from functools import lru_cache
from fastapi import Header, HTTPException, status
from python.core.pipeline import VisionPipeline
from python.config import settings

@lru_cache
def get_pipeline() -> VisionPipeline:
    return VisionPipeline()

async def get_current_api_key(
    x_apexvision_key: str = Header(..., alias="X-ApexVision-Key")
) -> str:
    if not x_apexvision_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    return x_apexvision_key

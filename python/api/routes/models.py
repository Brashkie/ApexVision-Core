"""ApexVision-Core — Model Registry Router"""
from fastapi import APIRouter, Depends
from python.api.deps import get_current_api_key

router = APIRouter()

@router.get("/", summary="List loaded models")
async def list_models(api_key: str = Depends(get_current_api_key)) -> dict:
    from python.core.model_registry import ModelRegistry
    from python.core.detector import YOLODetector
    return {"registry": ModelRegistry.status(), "yolo_cache": YOLODetector.loaded_models()}

@router.get("/variants", summary="List available YOLO variants")
async def list_variants() -> dict:
    from python.core.detector import YOLODetector
    return {"variants": YOLODetector.VARIANTS}

@router.delete("/cache", summary="Clear all model caches")
async def clear_cache(api_key: str = Depends(get_current_api_key)) -> dict:
    from python.core.detector import YOLODetector
    YOLODetector.clear_cache()
    return {"status": "cleared"}

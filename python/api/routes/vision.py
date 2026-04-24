"""ApexVision-Core — Vision Router"""
from __future__ import annotations
import time
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from python.config import settings
from python.core.pipeline import VisionPipeline
from python.schemas.vision import (
    ImageFormat, ImageInput, OutputFormat, VisionOptions,
    VisionRequest, VisionResponse, VisionTask, ErrorResponse,
)
from python.api.deps import get_pipeline, get_current_api_key

router = APIRouter()

@router.post("/analyze", response_model=VisionResponse,
             summary="Analyze image with multiple vision tasks")
async def analyze(
    request: VisionRequest,
    pipeline: VisionPipeline = Depends(get_pipeline),
    api_key: str = Depends(get_current_api_key),
) -> VisionResponse:
    t0 = time.perf_counter()
    logger.info(f"[{request.request_id}] tasks={request.tasks}")
    try:
        result = await pipeline.run(request)
        logger.info(f"[{request.request_id}] done in {(time.perf_counter()-t0)*1000:.1f}ms")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MemoryError:
        raise HTTPException(status_code=413, detail="Image too large")

@router.post("/analyze/upload", response_model=VisionResponse,
             summary="Analyze uploaded image file (multipart)")
async def analyze_upload(
    file: Annotated[UploadFile, File(description="Image file")],
    tasks: Annotated[str, Form()] = "detect",
    confidence: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    store_result: Annotated[bool, Form()] = False,
    pipeline: VisionPipeline = Depends(get_pipeline),
    api_key: str = Depends(get_current_api_key),
) -> VisionResponse:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB")
    import base64
    b64 = base64.b64encode(contents).decode()
    task_list = [VisionTask(t.strip()) for t in tasks.split(",") if t.strip()]
    req = VisionRequest(
        image=ImageInput(format=ImageFormat.BASE64, data=b64),
        tasks=task_list,
        options=VisionOptions(confidence_threshold=confidence),
        store_result=store_result,
    )
    return await analyze(req, pipeline, api_key)

@router.post("/detect",   response_model=VisionResponse, summary="Object detection (YOLOv11)")
async def detect(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.DETECT]; return await analyze(request, pipeline, api_key)

@router.post("/classify", response_model=VisionResponse, summary="Image classification (ViT/CLIP)")
async def classify(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.CLASSIFY]; return await analyze(request, pipeline, api_key)

@router.post("/ocr",      response_model=VisionResponse, summary="Text extraction / OCR")
async def ocr(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.OCR]; return await analyze(request, pipeline, api_key)

@router.post("/face",     response_model=VisionResponse, summary="Face detection + analysis")
async def face(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.FACE]; return await analyze(request, pipeline, api_key)

@router.post("/embed",    response_model=VisionResponse, summary="Image embedding (CLIP 512-d)")
async def embed(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.EMBED]; return await analyze(request, pipeline, api_key)

@router.post("/segment",  response_model=VisionResponse, summary="Segmentation (SAM)")
async def segment(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.SEGMENT]; return await analyze(request, pipeline, api_key)

@router.post("/depth",    response_model=VisionResponse, summary="Monocular depth estimation")
async def depth(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)):
    request.tasks = [VisionTask.DEPTH]; return await analyze(request, pipeline, api_key)

@router.post("/detect/preview", summary="Detect + return annotated image (base64 JPEG)")
async def detect_preview(request: VisionRequest, pipeline=Depends(get_pipeline), api_key=Depends(get_current_api_key)) -> dict:
    import cv2, numpy as np
    request.tasks = [VisionTask.DETECT]
    result = await analyze(request, pipeline, api_key)
    if not result.detection or not result.detection.boxes:
        return {"result": result, "preview_base64": None}
    raw = request.image.decode_bytes()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    from python.core.detector import YOLODetector
    annotated = YOLODetector.draw_boxes(img, result.detection.boxes)
    return {"result": result, "preview_base64": YOLODetector.encode_preview(annotated), "preview_mime": "image/jpeg"}

@router.post("/detect/compare", summary="Compare same image across multiple YOLO variants")
async def detect_compare(request: VisionRequest, variants: list[str] = ["nano", "small"], api_key=Depends(get_current_api_key)) -> dict:
    from python.core.detector import YOLODetector
    async def run_variant(v: str):
        try:
            det = YOLODetector.from_variant(v)
            image_np, _ = await VisionPipeline()._decode_image(request)
            r = await det.run(image_np, request.options)
            return {"variant": v, "model": YOLODetector.VARIANTS[v], "count": r.count,
                    "boxes": [b.model_dump() for b in r.boxes], "inference_ms": r.inference_ms}
        except Exception as e:
            return {"variant": v, "error": str(e)}
    import asyncio
    return {"comparisons": await asyncio.gather(*[run_variant(v) for v in variants])}

@router.get("/tasks", summary="List all available vision tasks")
async def list_tasks() -> dict:
    return {"tasks": [{"id": t.value, "name": t.name} for t in VisionTask]}

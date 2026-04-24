"""ApexVision-Core — Vision Pipeline Orchestrator"""
from __future__ import annotations
import asyncio
import hashlib
import time
import numpy as np
from loguru import logger
from python.config import settings
from python.schemas.vision import VisionRequest, VisionResponse, VisionTask

# Top-level imports — enables patch() in tests + avoids re-import on every request
from python.core.detector import YOLODetector
from python.core.classifier import VisionClassifier
from python.core.ocr_engine import OCREngine
from python.core.face_analyzer import FaceAnalyzer
from python.core.embedding_engine import EmbeddingEngine
from python.core.depth_estimator import DepthEstimator
from python.core.segmentor import SAMSegmentor


class VisionPipeline:
    async def run(self, request: VisionRequest) -> VisionResponse:
        t0 = time.perf_counter()
        image_np, (h, w) = await self._decode_image(request)

        if request.options.use_cache:
            cached = await self._cache_get(request)
            if cached:
                return cached

        results = await self._dispatch_tasks(request, image_np)
        total_ms = (time.perf_counter() - t0) * 1000

        # Map task names → VisionResponse field names
        _TASK_TO_FIELD = {
            "detect":     "detection",
            "classify":   "classification",
            "ocr":        "ocr",
            "face":       "face",
            "embed":      "embedding",
            "depth":      "depth",
            "segment":    "segmentation",
            "caption":    "caption",
            "similarity": "similarity",
            "custom":     "custom",
        }
        mapped = {_TASK_TO_FIELD.get(k, k): v for k, v in results.items()}

        response = VisionResponse(
            request_id=request.request_id,
            tasks_ran=request.tasks,
            image_width=w,
            image_height=h,
            total_inference_ms=round(total_ms, 2),
            **mapped,
        )

        if request.options.use_cache:
            await self._cache_set(request, response)

        if request.store_result:
            response.stored_at = await self._persist(request, response)

        return response

    async def _dispatch_tasks(self, request: VisionRequest, image_np: np.ndarray) -> dict:
        PARALLEL = {VisionTask.DETECT, VisionTask.CLASSIFY, VisionTask.OCR,
                    VisionTask.EMBED, VisionTask.DEPTH}
        SEQUENTIAL = {VisionTask.SEGMENT, VisionTask.FACE,
                      VisionTask.CAPTION, VisionTask.SIMILARITY, VisionTask.CUSTOM}

        parallel = [t for t in request.tasks if t in PARALLEL]
        sequential = [t for t in request.tasks if t in SEQUENTIAL]
        results: dict = {}

        if parallel:
            task_results = await asyncio.gather(
                *[self._run_task(t, request, image_np) for t in parallel],
                return_exceptions=True,
            )
            for task, result in zip(parallel, task_results):
                if isinstance(result, Exception):
                    logger.error(f"Task {task} failed: {result}")
                else:
                    results[str(task)] = result

        for task in sequential:
            try:
                results[str(task)] = await self._run_task(task, request, image_np)
            except Exception as e:
                logger.error(f"Task {task} failed: {e}")

        return results

    async def _run_task(self, task: VisionTask, request: VisionRequest, image_np: np.ndarray):
        opts = request.options
        if task == VisionTask.DETECT:
            return await YOLODetector().run(image_np, opts)
        if task == VisionTask.CLASSIFY:
            return await VisionClassifier().run(image_np, opts)
        if task == VisionTask.OCR:
            return await OCREngine().run(image_np, opts)
        if task == VisionTask.FACE:
            return await FaceAnalyzer().run(image_np, opts)
        if task == VisionTask.EMBED:
            return await EmbeddingEngine().run(image_np, opts)
        if task == VisionTask.DEPTH:
            return await DepthEstimator().run(image_np, opts)
        if task == VisionTask.SEGMENT:
            return await SAMSegmentor().run(image_np, opts)
        raise NotImplementedError(f"Task '{task}' not implemented yet")

    async def _decode_image(self, request: VisionRequest) -> tuple[np.ndarray, tuple[int, int]]:
        import cv2
        from python.schemas.vision import ImageFormat
        if request.image.format == ImageFormat.BASE64:
            raw = request.image.decode_bytes()
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image — invalid or corrupted data")
        elif request.image.format == ImageFormat.URL:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(request.image.url)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not decode image from URL: {request.image.url}")
        else:
            raise ValueError(f"Unsupported image format: {request.image.format}")
        h, w = img.shape[:2]
        if h > 8192 or w > 8192:
            raise ValueError(f"Image {w}x{h} exceeds max 8192px")
        return img, (h, w)

    def _cache_key(self, request: VisionRequest) -> str:
        payload = f"{request.image.data or request.image.url}{sorted(request.tasks)}{request.options.confidence_threshold}"
        return f"apex:vision:{hashlib.sha256(payload.encode()).hexdigest()}"

    async def _cache_get(self, request: VisionRequest) -> VisionResponse | None:
        try:
            from python.cache.redis_client import redis_client
            raw = await redis_client.get(self._cache_key(request))
            if raw:
                return VisionResponse.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    async def _cache_set(self, request: VisionRequest, response: VisionResponse) -> None:
        try:
            from python.cache.redis_client import redis_client
            await redis_client.setex(self._cache_key(request), settings.REDIS_CACHE_TTL, response.model_dump_json())
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def _persist(self, request: VisionRequest, response: VisionResponse) -> str:
        try:
            from python.storage.delta_store import DeltaStore
            return await DeltaStore().write_result(response)
        except Exception as e:
            logger.error(f"Delta persist failed: {e}")
            return ""

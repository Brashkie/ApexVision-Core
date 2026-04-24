"""
ApexVision-Core — Pipeline Integration Tests
Tests end-to-end del VisionPipeline completo con todos los engines mockeados.
Verifica: routing de tasks, concurrencia, caché, persistencia, manejo de errores.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image
import io

from python.schemas.vision import (
    VisionRequest, VisionResponse, VisionTask,
    ImageInput, ImageFormat, VisionOptions,
    DetectionResult, ClassificationResult, OCRResult,
    FaceResult, EmbeddingResult, DepthResult, SegmentationResult,
    BoundingBox,
)
from python.core.pipeline import VisionPipeline


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_b64_image(w: int = 64, h: int = 64) -> str:
    """Create a minimal valid base64 JPEG."""
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_request(
    tasks: list[VisionTask],
    use_cache: bool = False,
    store_result: bool = False,
) -> VisionRequest:
    return VisionRequest(
        image=ImageInput(format=ImageFormat.BASE64, data=make_b64_image()),
        tasks=tasks,
        options=VisionOptions(use_cache=use_cache),
        store_result=store_result,
    )


def make_detection_result(n: int = 2) -> DetectionResult:
    return DetectionResult(
        boxes=[
            BoundingBox(x1=10, y1=10, x2=200, y2=300,
                        width=190, height=290, confidence=0.9,
                        label="person", label_id=0)
        ] * n,
        count=n,
        model_used="yolov11n.pt",
        inference_ms=38.0,
    )


def make_classification_result() -> ClassificationResult:
    return ClassificationResult(
        predictions=[{"label": "cat", "confidence": 0.85, "label_id": 0}],
        model_used="vit-base",
        inference_ms=22.0,
    )


def make_ocr_result(text: str = "Hello World") -> OCRResult:
    return OCRResult(
        text=text,
        blocks=[{"text": text, "confidence": 0.95,
                 "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 30, "width": 100, "height": 30}}],
        language_detected="en",
        inference_ms=45.0,
    )


def make_embedding_result(dim: int = 512) -> EmbeddingResult:
    raw = np.random.randn(dim).astype(np.float32)
    raw = raw / np.linalg.norm(raw)
    return EmbeddingResult(
        embedding=[round(float(v), 6) for v in raw.tolist()],
        dimensions=dim,
        model_used="openai/clip-vit-base-patch32",
        inference_ms=18.0,
    )


def make_face_result(count: int = 1) -> FaceResult:
    return FaceResult(
        faces=[{
            "bbox": {"x1": 50, "y1": 60, "x2": 200, "y2": 250,
                     "width": 150, "height": 190, "confidence": 0.97,
                     "label": "face", "label_id": 0},
            "attributes": {"age": 28, "gender": "female", "emotion": "happiness"},
        }] * count,
        count=count,
        inference_ms=55.0,
    )


def make_depth_result() -> DepthResult:
    return DepthResult(
        depth_map_base64="fakeb64==",
        min_depth=0.5,
        max_depth=20.0,
        inference_ms=62.0,
    )


def make_segmentation_result(n: int = 3) -> SegmentationResult:
    return SegmentationResult(
        masks=[{"label": "object", "score": 0.9, "area": 5000,
                "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100,
                         "width": 100, "height": 100},
                "mask_rle": {"counts": [1000, 1000], "size": [64, 64]},
                "backend": "sam"}] * n,
        count=n,
        inference_ms=120.0,
    )


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def pipeline():
    return VisionPipeline()


# ─────────────────────────────────────────────
#  Integration: single tasks
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_detect_only(pipeline):
    request  = make_request([VisionTask.DETECT])
    det_result = make_detection_result(3)

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        instance = MockDet.return_value
        instance.run = AsyncMock(return_value=det_result)

        result = await pipeline.run(request)

    assert isinstance(result, VisionResponse)
    assert result.detection is not None
    assert result.detection.count == 3
    assert result.classification is None
    assert result.ocr is None
    assert VisionTask.DETECT in result.tasks_ran


@pytest.mark.asyncio
async def test_pipeline_classify_only(pipeline):
    request   = make_request([VisionTask.CLASSIFY])
    clf_result = make_classification_result()

    with patch("python.core.pipeline.VisionClassifier") as MockClf:
        instance = MockClf.return_value
        instance.run = AsyncMock(return_value=clf_result)

        result = await pipeline.run(request)

    assert result.classification is not None
    assert result.classification.predictions[0]["label"] == "cat"
    assert result.detection is None


@pytest.mark.asyncio
async def test_pipeline_ocr_only(pipeline):
    request    = make_request([VisionTask.OCR])
    ocr_result = make_ocr_result("Invoice #1234")

    with patch("python.core.pipeline.OCREngine") as MockOCR:
        instance = MockOCR.return_value
        instance.run = AsyncMock(return_value=ocr_result)

        result = await pipeline.run(request)

    assert result.ocr is not None
    assert "Invoice" in result.ocr.text


@pytest.mark.asyncio
async def test_pipeline_embed_only(pipeline):
    request    = make_request([VisionTask.EMBED])
    emb_result = make_embedding_result(512)

    with patch("python.core.pipeline.EmbeddingEngine") as MockEmb:
        instance = MockEmb.return_value
        instance.run = AsyncMock(return_value=emb_result)

        result = await pipeline.run(request)

    assert result.embedding is not None
    assert result.embedding.dimensions == 512
    assert len(result.embedding.embedding) == 512


@pytest.mark.asyncio
async def test_pipeline_face_only(pipeline):
    request     = make_request([VisionTask.FACE])
    face_result = make_face_result(2)

    with patch("python.core.pipeline.FaceAnalyzer") as MockFace:
        instance = MockFace.return_value
        instance.run = AsyncMock(return_value=face_result)

        result = await pipeline.run(request)

    assert result.face is not None
    assert result.face.count == 2


@pytest.mark.asyncio
async def test_pipeline_depth_only(pipeline):
    request     = make_request([VisionTask.DEPTH])
    dep_result  = make_depth_result()

    with patch("python.core.pipeline.DepthEstimator") as MockDep:
        instance = MockDep.return_value
        instance.run = AsyncMock(return_value=dep_result)

        result = await pipeline.run(request)

    assert result.depth is not None
    assert result.depth.min_depth == 0.5
    assert result.depth.max_depth == 20.0


@pytest.mark.asyncio
async def test_pipeline_segment_only(pipeline):
    request    = make_request([VisionTask.SEGMENT])
    seg_result = make_segmentation_result(4)

    with patch("python.core.pipeline.SAMSegmentor") as MockSeg:
        instance = MockSeg.return_value
        instance.run = AsyncMock(return_value=seg_result)

        result = await pipeline.run(request)

    assert result.segmentation is not None
    assert result.segmentation.count == 4


# ─────────────────────────────────────────────
#  Integration: multi-task
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_detect_and_classify(pipeline):
    request    = make_request([VisionTask.DETECT, VisionTask.CLASSIFY])
    det_result = make_detection_result(2)
    clf_result = make_classification_result()

    with patch("python.core.pipeline.YOLODetector") as MockDet, \
         patch("python.core.pipeline.VisionClassifier") as MockClf:
        MockDet.return_value.run = AsyncMock(return_value=det_result)
        MockClf.return_value.run = AsyncMock(return_value=clf_result)

        result = await pipeline.run(request)

    assert result.detection     is not None
    assert result.classification is not None
    assert result.detection.count == 2
    assert result.classification.predictions[0]["label"] == "cat"


@pytest.mark.asyncio
async def test_pipeline_detect_ocr_embed(pipeline):
    request    = make_request([VisionTask.DETECT, VisionTask.OCR, VisionTask.EMBED])
    det_result = make_detection_result(1)
    ocr_result = make_ocr_result("STOP")
    emb_result = make_embedding_result(512)

    with patch("python.core.pipeline.YOLODetector") as MockDet, \
         patch("python.core.pipeline.OCREngine") as MockOCR, \
         patch("python.core.pipeline.EmbeddingEngine") as MockEmb:
        MockDet.return_value.run = AsyncMock(return_value=det_result)
        MockOCR.return_value.run = AsyncMock(return_value=ocr_result)
        MockEmb.return_value.run = AsyncMock(return_value=emb_result)

        result = await pipeline.run(request)

    assert result.detection  is not None
    assert result.ocr        is not None
    assert result.embedding  is not None
    assert result.ocr.text   == "STOP"
    assert len(result.tasks_ran) == 3


@pytest.mark.asyncio
async def test_pipeline_all_parallel_tasks(pipeline):
    """All 5 parallel tasks (detect, classify, ocr, embed, depth) run together."""
    request = make_request([
        VisionTask.DETECT,
        VisionTask.CLASSIFY,
        VisionTask.OCR,
        VisionTask.EMBED,
        VisionTask.DEPTH,
    ])

    with patch("python.core.pipeline.YOLODetector") as D, \
         patch("python.core.pipeline.VisionClassifier") as C, \
         patch("python.core.pipeline.OCREngine") as O, \
         patch("python.core.pipeline.EmbeddingEngine") as E, \
         patch("python.core.pipeline.DepthEstimator") as De:

        D.return_value.run  = AsyncMock(return_value=make_detection_result())
        C.return_value.run  = AsyncMock(return_value=make_classification_result())
        O.return_value.run  = AsyncMock(return_value=make_ocr_result())
        E.return_value.run  = AsyncMock(return_value=make_embedding_result())
        De.return_value.run = AsyncMock(return_value=make_depth_result())

        result = await pipeline.run(request)

    assert result.detection      is not None
    assert result.classification is not None
    assert result.ocr            is not None
    assert result.embedding      is not None
    assert result.depth          is not None
    assert len(result.tasks_ran) == 5


# ─────────────────────────────────────────────
#  Integration: response metadata
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_response_has_image_dims(pipeline):
    request = make_request([VisionTask.DETECT])

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
        result = await pipeline.run(request)

    assert result.image_width  > 0
    assert result.image_height > 0


@pytest.mark.asyncio
async def test_pipeline_response_has_timing(pipeline):
    request = make_request([VisionTask.DETECT])

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
        result = await pipeline.run(request)

    assert result.total_inference_ms >= 0.0


@pytest.mark.asyncio
async def test_pipeline_request_id_preserved(pipeline):
    request = make_request([VisionTask.DETECT])
    original_id = request.request_id

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
        result = await pipeline.run(request)

    assert result.request_id == original_id


@pytest.mark.asyncio
async def test_pipeline_tasks_ran_matches_request(pipeline):
    tasks = [VisionTask.DETECT, VisionTask.OCR]
    request = make_request(tasks)

    with patch("python.core.pipeline.YOLODetector") as MockDet, \
         patch("python.core.pipeline.OCREngine") as MockOCR:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
        MockOCR.return_value.run = AsyncMock(return_value=make_ocr_result())
        result = await pipeline.run(request)

    for task in tasks:
        assert task in result.tasks_ran


# ─────────────────────────────────────────────
#  Integration: cache
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_cache_hit_skips_inference(pipeline):
    request = make_request([VisionTask.DETECT], use_cache=True)
    cached_response = VisionResponse(
        request_id=request.request_id,
        tasks_ran=[VisionTask.DETECT],
        detection=make_detection_result(5),
        image_width=64, image_height=64,
        total_inference_ms=1.0,
    )

    with patch.object(pipeline, "_cache_get", AsyncMock(return_value=cached_response)):
        with patch("python.core.pipeline.YOLODetector") as MockDet:
            MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
            result = await pipeline.run(request)
            # Inference should NOT have been called (cache hit)
            MockDet.return_value.run.assert_not_called()

    assert result.detection.count == 5


@pytest.mark.asyncio
async def test_pipeline_cache_miss_runs_inference(pipeline):
    request = make_request([VisionTask.DETECT], use_cache=True)

    with patch.object(pipeline, "_cache_get", AsyncMock(return_value=None)):
        with patch.object(pipeline, "_cache_set", AsyncMock()):
            with patch("python.core.pipeline.YOLODetector") as MockDet:
                MockDet.return_value.run = AsyncMock(return_value=make_detection_result(2))
                result = await pipeline.run(request)

    assert result.detection.count == 2


@pytest.mark.asyncio
async def test_pipeline_cache_set_called_on_miss(pipeline):
    request = make_request([VisionTask.DETECT], use_cache=True)

    with patch.object(pipeline, "_cache_get", AsyncMock(return_value=None)):
        with patch.object(pipeline, "_cache_set", AsyncMock()) as mock_set:
            with patch("python.core.pipeline.YOLODetector") as MockDet:
                MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
                await pipeline.run(request)

    mock_set.assert_called_once()


# ─────────────────────────────────────────────
#  Integration: cache key
# ─────────────────────────────────────────────

def test_cache_key_different_tasks():
    pipeline = VisionPipeline()
    req_a = make_request([VisionTask.DETECT])
    req_b = make_request([VisionTask.CLASSIFY])
    # Same image, different tasks → different cache keys
    key_a = pipeline._cache_key(req_a)
    key_b = pipeline._cache_key(req_b)
    assert key_a != key_b


def test_cache_key_same_request_same_key():
    pipeline = VisionPipeline()
    req = make_request([VisionTask.DETECT])
    assert pipeline._cache_key(req) == pipeline._cache_key(req)


# ─────────────────────────────────────────────
#  Integration: Delta persistence
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_store_result_calls_delta(pipeline):
    request = make_request([VisionTask.DETECT], store_result=True)

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())

        with patch.object(pipeline, "_persist", AsyncMock(return_value="/data/delta/vision_results")) as mock_persist:
            result = await pipeline.run(request)

    mock_persist.assert_called_once()
    assert result.stored_at == "/data/delta/vision_results"


@pytest.mark.asyncio
async def test_pipeline_no_store_result_skips_delta(pipeline):
    request = make_request([VisionTask.DETECT], store_result=False)

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result())
        with patch.object(pipeline, "_persist", AsyncMock()) as mock_persist:
            result = await pipeline.run(request)

    mock_persist.assert_not_called()
    assert result.stored_at is None


# ─────────────────────────────────────────────
#  Integration: error handling
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_invalid_image_raises():
    pipeline = VisionPipeline()
    request  = VisionRequest(
        image=ImageInput(format=ImageFormat.BASE64, data="not_valid_base64!!!"),
        tasks=[VisionTask.DETECT],
    )
    with pytest.raises(Exception):
        await pipeline.run(request)


@pytest.mark.asyncio
async def test_pipeline_engine_error_logged_not_raised(pipeline):
    """If a parallel task engine crashes, the pipeline continues with other tasks."""
    request = make_request([VisionTask.DETECT, VisionTask.OCR])

    with patch("python.core.pipeline.YOLODetector") as MockDet, \
         patch("python.core.pipeline.OCREngine") as MockOCR:
        MockDet.return_value.run = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        MockOCR.return_value.run = AsyncMock(return_value=make_ocr_result("Text found"))

        # Pipeline should not raise, it logs the error
        result = await pipeline.run(request)

    # OCR succeeded even though detection failed
    assert result.ocr is not None
    assert result.ocr.text == "Text found"


@pytest.mark.asyncio
async def test_pipeline_url_image_decoding(pipeline):
    """Test URL image format goes through httpx fetch path."""
    request = VisionRequest(
        image=ImageInput(format=ImageFormat.URL, url="https://example.com/image.jpg"),
        tasks=[VisionTask.DETECT],
        options=VisionOptions(use_cache=False),
    )

    fake_img_bytes = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(fake_img_bytes, format="JPEG")

    mock_response = MagicMock()
    mock_response.content = fake_img_bytes.getvalue()
    mock_response.raise_for_status = MagicMock()

    with patch("python.core.pipeline.YOLODetector") as MockDet:
        MockDet.return_value.run = AsyncMock(return_value=make_detection_result(1))

        with patch("httpx.AsyncClient") as MockHttp:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__  = AsyncMock(return_value=False)
            mock_ctx.get        = AsyncMock(return_value=mock_response)
            MockHttp.return_value = mock_ctx

            result = await pipeline.run(request)

    assert result.detection is not None
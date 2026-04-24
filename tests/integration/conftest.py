"""
ApexVision-Core — Integration Tests conftest.py
Fixtures compartidos: app client, imagen de prueba, mocks de modelos ML.

Estrategia:
  - Usa httpx.AsyncClient con la app FastAPI real en memoria (ASGI transport)
  - Mockea los engines ML (YOLO, CLIP, etc.) para no requerir modelos descargados
  - Requiere Redis y PostgreSQL corriendo (igual que npm run dev)
  - Los tests de integración verifican el stack completo: routing, auth,
    validación, pipeline, serialización de respuesta
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

# ── Fixtures de la app ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_key() -> str:
    return os.environ.get("MASTER_API_KEY", "apexvision-master-dev-key")


@pytest.fixture(scope="session")
def auth_headers(api_key: str) -> dict:
    return {"X-ApexVision-Key": api_key}


@pytest.fixture(scope="session")
def small_jpeg_b64() -> str:
    """1x1 JPEG válido en base64 — mínimo overhead para tests."""
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="session")
def small_jpeg_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def detect_payload(small_jpeg_b64: str) -> dict:
    return {
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["detect"],
        "options": {"confidence_threshold": 0.5},
    }


@pytest.fixture(scope="session")
def multitask_payload(small_jpeg_b64: str) -> dict:
    return {
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["detect", "classify"],
        "options": {"confidence_threshold": 0.4, "top_k": 3},
    }


# ── Mocks de engines ML ──────────────────────────────────────────────────────

def make_detection_result() -> MagicMock:
    from python.schemas.vision import DetectionResult, BoundingBox
    return DetectionResult(
        boxes=[
            BoundingBox(
                x1=10, y1=10, x2=200, y2=300,
                width=190, height=290,
                confidence=0.92,
                label="person", label_id=0,
            )
        ],
        count=1,
        model_used="yolo11n.pt",
        inference_ms=38.5,
    )


def make_classification_result() -> MagicMock:
    from python.schemas.vision import ClassificationResult
    return ClassificationResult(
        predictions=[
            {"label": "cat", "confidence": 0.85, "label_id": 0},
            {"label": "dog", "confidence": 0.10, "label_id": 1},
        ],
        model_used="vit-base",
        inference_ms=22.1,
    )


def make_ocr_result(text: str = "Hello World") -> MagicMock:
    from python.schemas.vision import OCRResult
    return OCRResult(
        text=text,
        blocks=[{
            "text": text,
            "confidence": 0.95,
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 30, "width": 100, "height": 30},
        }],
        language_detected="en",
        inference_ms=44.2,
    )


def make_embedding_result(dim: int = 512) -> MagicMock:
    from python.schemas.vision import EmbeddingResult
    raw = np.random.randn(dim).astype(np.float32)
    raw = raw / np.linalg.norm(raw)
    return EmbeddingResult(
        embedding=[round(float(v), 6) for v in raw.tolist()],
        dimensions=dim,
        model_used="openai/clip-vit-base-patch32",
        inference_ms=18.3,
    )


def make_face_result() -> MagicMock:
    from python.schemas.vision import FaceResult
    return FaceResult(
        faces=[{
            "bbox": {
                "x1": 50, "y1": 60, "x2": 200, "y2": 250,
                "width": 150, "height": 190,
                "confidence": 0.97, "label": "face", "label_id": 0,
            },
            "attributes": {"age": 28, "gender": "female", "emotion": "happiness"},
        }],
        count=1,
        inference_ms=55.0,
    )


def make_depth_result() -> MagicMock:
    from python.schemas.vision import DepthResult
    return DepthResult(
        depth_map_base64="fakeb64==",
        min_depth=0.5,
        max_depth=20.0,
        inference_ms=62.0,
    )


def make_segmentation_result() -> MagicMock:
    from python.schemas.vision import SegmentationResult
    return SegmentationResult(
        masks=[{
            "label": "object", "label_id": -1, "score": 0.92, "area": 5000,
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100,
                     "width": 100, "height": 100},
            "mask_rle": {"counts": [1000, 1000], "size": [64, 64]},
            "backend": "sam",
        }],
        count=1,
        inference_ms=120.0,
    )


# ── App client con mocks de ML ────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def app_client(api_key: str) -> AsyncGenerator[AsyncClient, None]:
    """
    Cliente HTTP contra la app FastAPI real en memoria.
    Mockea los engines ML para no requerir modelos descargados.
    Redis y PostgreSQL sí deben estar corriendo.
    """
    from python.main import create_app

    patches = [
        patch("python.core.pipeline.YOLODetector", autospec=True),
        patch("python.core.pipeline.VisionClassifier", autospec=True),
        patch("python.core.pipeline.OCREngine", autospec=True),
        patch("python.core.pipeline.FaceAnalyzer", autospec=True),
        patch("python.core.pipeline.EmbeddingEngine", autospec=True),
        patch("python.core.pipeline.DepthEstimator", autospec=True),
        patch("python.core.pipeline.SAMSegmentor", autospec=True),
        patch("python.core.model_registry.ModelRegistry.warmup", new_callable=AsyncMock),
    ]

    mocks = [p.start() for p in patches]

    # Configurar cada mock para devolver resultados realistas
    det_mock, clf_mock, ocr_mock, face_mock, emb_mock, dep_mock, seg_mock, _ = mocks

    det_mock.return_value.run  = AsyncMock(return_value=make_detection_result())
    clf_mock.return_value.run  = AsyncMock(return_value=make_classification_result())
    ocr_mock.return_value.run  = AsyncMock(return_value=make_ocr_result())
    face_mock.return_value.run = AsyncMock(return_value=make_face_result())
    emb_mock.return_value.run  = AsyncMock(return_value=make_embedding_result())
    dep_mock.return_value.run  = AsyncMock(return_value=make_depth_result())
    seg_mock.return_value.run  = AsyncMock(return_value=make_segmentation_result())

    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-ApexVision-Key": api_key},
    ) as client:
        yield client

    for p in patches:
        p.stop()

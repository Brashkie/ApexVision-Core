"""
ApexVision-Core — Models Endpoints Integration Tests
Verifica: list models, variants, cache clear.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch


@pytest.mark.asyncio
async def test_list_models(app_client: AsyncClient):
    r = await app_client.get("/api/v1/models/")
    assert r.status_code == 200
    body = r.json()
    # Debe devolver info del registry
    assert isinstance(body, dict)


@pytest.mark.asyncio
async def test_list_variants(app_client: AsyncClient):
    r = await app_client.get("/api/v1/models/variants")
    assert r.status_code == 200
    body = r.json()
    assert "variants" in body
    variants = body["variants"]
    assert isinstance(variants, dict)
    # YOLO variants esperadas
    for variant in ["nano", "small", "medium"]:
        assert variant in variants


@pytest.mark.asyncio
async def test_clear_cache(app_client: AsyncClient):
    with patch("python.core.detector.YOLODetector.clear_cache"), \
         patch("python.core.classifier.VisionClassifier.clear_cache"), \
         patch("python.core.ocr_engine.OCREngine.clear_cache"):
        r = await app_client.delete("/api/v1/models/cache")

    assert r.status_code == 200
    body = r.json()
    assert "status" in body

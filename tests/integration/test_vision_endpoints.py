"""
ApexVision-Core — Vision Endpoints Integration Tests
Verifica: analyze, shortcuts por task, upload, validación de input,
          estructura de respuesta, multi-task, manejo de errores.
"""

import io
import uuid
import pytest
from httpx import AsyncClient
from PIL import Image
import numpy as np


# ─────────────────────────────────────────────
#  /api/v1/vision/analyze
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_detect_returns_200(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_analyze_response_schema(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    body = r.json()
    # Campos requeridos de VisionResponse
    assert "request_id"          in body
    assert "status"              in body
    assert "tasks_ran"           in body
    assert "image_width"         in body
    assert "image_height"        in body
    assert "total_inference_ms"  in body
    assert body["status"]        == "success"
    assert "detect" in body["tasks_ran"]


@pytest.mark.asyncio
async def test_analyze_detection_result(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    body = r.json()
    det = body.get("detection")
    assert det is not None
    assert det["count"] == 1
    assert len(det["boxes"]) == 1
    box = det["boxes"][0]
    assert "label"      in box
    assert "confidence" in box
    assert "x1" in box and "y1" in box
    assert box["label"] == "person"
    assert 0.0 <= box["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_analyze_request_id_preserved(app_client: AsyncClient, small_jpeg_b64: str):
    custom_id = str(uuid.uuid4())
    payload = {
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["detect"],
        "request_id": custom_id,
    }
    r = await app_client.post("/api/v1/vision/analyze", json=payload)
    assert r.json()["request_id"] == custom_id


@pytest.mark.asyncio
async def test_analyze_multitask(app_client: AsyncClient, multitask_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=multitask_payload)
    body = r.json()
    assert r.status_code == 200
    assert "detect"   in body["tasks_ran"]
    assert "classify" in body["tasks_ran"]
    assert body["detection"]     is not None
    assert body["classification"] is not None


@pytest.mark.asyncio
async def test_analyze_classification_result(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["classify"],
        "options": {"top_k": 3},
    })
    body = r.json()
    clf = body.get("classification")
    assert clf is not None
    assert len(clf["predictions"]) > 0
    top = clf["predictions"][0]
    assert "label"      in top
    assert "confidence" in top
    assert 0.0 <= top["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_analyze_ocr(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["ocr"],
    })
    body = r.json()
    assert r.status_code == 200
    assert body.get("ocr") is not None
    assert "text" in body["ocr"]


@pytest.mark.asyncio
async def test_analyze_face(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["face"],
        "options": {"face_attributes": True},
    })
    body = r.json()
    assert r.status_code == 200
    face = body.get("face")
    assert face is not None
    assert face["count"] >= 0
    if face["count"] > 0:
        f = face["faces"][0]
        assert "bbox"       in f
        assert "attributes" in f


@pytest.mark.asyncio
async def test_analyze_embed(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["embed"],
    })
    body = r.json()
    assert r.status_code == 200
    emb = body.get("embedding")
    assert emb is not None
    assert emb["dimensions"] == 512
    assert len(emb["embedding"]) == 512
    # Verificar que el embedding está L2-normalizado (norma ~1.0)
    import math
    norm = math.sqrt(sum(v**2 for v in emb["embedding"]))
    assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def test_analyze_depth(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["depth"],
    })
    body = r.json()
    assert r.status_code == 200
    dep = body.get("depth")
    assert dep is not None
    assert "depth_map_base64" in dep
    assert dep["min_depth"] < dep["max_depth"]


@pytest.mark.asyncio
async def test_analyze_segment(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["segment"],
    })
    body = r.json()
    assert r.status_code == 200
    seg = body.get("segmentation")
    assert seg is not None
    assert seg["count"] >= 0


@pytest.mark.asyncio
async def test_analyze_inference_ms_positive(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    assert r.json()["total_inference_ms"] >= 0.0


@pytest.mark.asyncio
async def test_analyze_image_dimensions_returned(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    body = r.json()
    assert body["image_width"]  > 0
    assert body["image_height"] > 0


# ─────────────────────────────────────────────
#  Validation errors
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_missing_image_returns_422(app_client: AsyncClient):
    r = await app_client.post("/api/v1/vision/analyze", json={"tasks": ["detect"]})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyze_invalid_task_returns_422(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["not_a_real_task"],
    })
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyze_invalid_base64_returns_error(app_client: AsyncClient):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": "!!!not_valid_base64!!!"},
        "tasks": ["detect"],
    })
    assert r.status_code in (400, 422, 500)


@pytest.mark.asyncio
async def test_analyze_confidence_out_of_range(app_client: AsyncClient, small_jpeg_b64: str):
    r = await app_client.post("/api/v1/vision/analyze", json={
        "image": {"format": "base64", "data": small_jpeg_b64},
        "tasks": ["detect"],
        "options": {"confidence_threshold": 99.9},
    })
    assert r.status_code == 422


# ─────────────────────────────────────────────
#  Shortcut endpoints
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_detect_shortcut(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/detect", json=detect_payload)
    assert r.status_code == 200
    body = r.json()
    assert "detect" in body["tasks_ran"]
    assert body["detection"] is not None


@pytest.mark.asyncio
async def test_classify_shortcut(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/classify", json=detect_payload)
    assert r.status_code == 200
    assert "classify" in r.json()["tasks_ran"]


@pytest.mark.asyncio
async def test_ocr_shortcut(app_client: AsyncClient, detect_payload: dict):
    r = await app_client.post("/api/v1/vision/ocr", json=detect_payload)
    assert r.status_code == 200
    assert "ocr" in r.json()["tasks_ran"]


@pytest.mark.asyncio
async def test_list_tasks(app_client: AsyncClient):
    r = await app_client.get("/api/v1/vision/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body
    task_ids = [t["id"] for t in body["tasks"]]
    for expected in ["detect", "classify", "ocr", "face", "embed", "depth", "segment"]:
        assert expected in task_ids


# ─────────────────────────────────────────────
#  File upload
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_jpeg(app_client: AsyncClient, small_jpeg_bytes: bytes):
    r = await app_client.post(
        "/api/v1/vision/analyze/upload",
        files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        data={"tasks": "detect", "confidence": "0.5"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"]    == "success"
    assert "detect"          in body["tasks_ran"]


@pytest.mark.asyncio
async def test_upload_png(app_client: AsyncClient):
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = await app_client.post(
        "/api/v1/vision/analyze/upload",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
        data={"tasks": "classify"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_upload_invalid_content_type(app_client: AsyncClient):
    r = await app_client.post(
        "/api/v1/vision/analyze/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        data={"tasks": "detect"},
    )
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_upload_multitask(app_client: AsyncClient, small_jpeg_bytes: bytes):
    r = await app_client.post(
        "/api/v1/vision/analyze/upload",
        files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        data={"tasks": "detect,classify", "confidence": "0.4"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "detect"   in body["tasks_ran"]
    assert "classify" in body["tasks_ran"]

"""
ApexVision-Core — Batch Endpoints Integration Tests
Verifica: submit, status, cancel, estructura de respuesta.
"""

import uuid
import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────
#  POST /api/v1/batch/submit
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_submit_returns_job_id(
    app_client: AsyncClient, detect_payload: dict
):
    with patch("python.tasks.batch_tasks.process_batch_task") as mock_task:
        mock_task.apply_async.return_value = MagicMock(id="fake-job-id")
        r = await app_client.post("/api/v1/batch/submit", json={
            "requests": [detect_payload, detect_payload],
            "job_name": "test_batch",
        })

    assert r.status_code == 200
    body = r.json()
    assert "job_id"  in body
    assert "status"  in body
    assert "total"   in body
    assert body["total"] == 2
    assert body["status"] in ("pending", "submitted")


@pytest.mark.asyncio
async def test_batch_submit_single_image(app_client: AsyncClient, detect_payload: dict):
    with patch("python.tasks.batch_tasks.process_batch_task") as mock_task:
        mock_task.apply_async.return_value = MagicMock(id=str(uuid.uuid4()))
        r = await app_client.post("/api/v1/batch/submit", json={
            "requests": [detect_payload],
        })

    assert r.status_code == 200
    assert r.json()["total"] == 1


@pytest.mark.asyncio
async def test_batch_submit_missing_requests_returns_422(app_client: AsyncClient):
    r = await app_client.post("/api/v1/batch/submit", json={"job_name": "bad"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_batch_submit_empty_requests(app_client: AsyncClient):
    # Empty requests list — FastAPI may validate or accept depending on schema
    with patch("python.tasks.batch_tasks.process_batch_task") as mock_task:
        mock_task.apply_async.return_value = MagicMock(id=str(uuid.uuid4()))
        r = await app_client.post("/api/v1/batch/submit", json={"requests": []})

    assert r.status_code in (200, 422)  # schema may require min 1 request


# ─────────────────────────────────────────────
#  GET /api/v1/batch/{job_id}
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_status_structure(app_client: AsyncClient):
    job_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.info  = {
        "job_id":       job_id,
        "total":        5,
        "completed":    5,
        "failed":       0,
        "progress_pct": 100.0,
        "result_path":  "/data/parquet/results.parquet",
        "created_at":   "2025-01-01T00:00:00Z",
        "updated_at":   "2025-01-01T00:01:00Z",
    }

    with patch("python.celery_app.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result
        r = await app_client.get(f"/api/v1/batch/{job_id}")

    assert r.status_code == 200
    body = r.json()
    assert "job_id"       in body
    assert "status"       in body
    assert "total"        in body
    assert "completed"    in body
    assert "failed"       in body
    assert "progress_pct" in body


@pytest.mark.asyncio
async def test_batch_status_pending(app_client: AsyncClient):
    job_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info  = {}

    with patch("python.celery_app.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result
        r = await app_client.get(f"/api/v1/batch/{job_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("pending", "success", "failure")


@pytest.mark.asyncio
async def test_batch_status_progress(app_client: AsyncClient):
    job_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.state = "PROGRESS"
    mock_result.info  = {
        "total":        10,
        "completed":    6,
        "failed":       1,
        "progress_pct": 70.0,
    }

    with patch("python.celery_app.celery_app") as mock_celery:
        mock_celery.AsyncResult.return_value = mock_result
        r = await app_client.get(f"/api/v1/batch/{job_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["progress_pct"] >= 0.0


# ─────────────────────────────────────────────
#  DELETE /api/v1/batch/{job_id}
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_cancel(app_client: AsyncClient):
    job_id = str(uuid.uuid4())

    with patch("python.celery_app.celery_app") as mock_celery:
        mock_celery.control.revoke = MagicMock()
        r = await app_client.delete(f"/api/v1/batch/{job_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] == "cancelled"


@pytest.mark.asyncio
async def test_batch_cancel_calls_revoke(app_client: AsyncClient):
    job_id = str(uuid.uuid4())

    with patch("python.celery_app.celery_app") as mock_celery:
        mock_celery.control.revoke = MagicMock()
        await app_client.delete(f"/api/v1/batch/{job_id}")
        mock_celery.control.revoke.assert_called_once_with(job_id, terminate=True)

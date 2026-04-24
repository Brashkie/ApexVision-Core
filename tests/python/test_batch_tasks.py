"""
ApexVision-Core — Batch Tasks Tests
Celery tasks testeados con always_eager + memory backend (sin Redis real).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────
#  Celery eager config fixture
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def celery_eager(monkeypatch):
    """Make all Celery tasks run synchronously in-process, no Redis needed."""
    from python.celery_app import celery_app
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=False,
        result_backend="cache",
        cache_backend="memory",
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        result_backend="redis://localhost:6379/2",
        cache_backend=None,
    )


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_request_payload(tasks: list = None) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "image": {
            "format": "base64",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        },
        "tasks": tasks or ["detect"],
        "options": {"confidence_threshold": 0.5},
    }


def make_vision_result_dict(request_id: str = None, status: str = "success") -> dict:
    rid = request_id or str(uuid.uuid4())
    return {
        "request_id": rid,
        "status": status,
        "tasks_ran": ["detect"],
        "image_width": 640,
        "image_height": 480,
        "total_inference_ms": 42.5,
        "detection": {"count": 2, "model_used": "yolov11n.pt", "inference_ms": 38.0, "boxes": []},
        "classification": None, "ocr": None, "face": None,
        "embedding": None, "depth": None, "segmentation": None, "stored_at": None,
    }


# ─────────────────────────────────────────────
#  Unit: _error_result
# ─────────────────────────────────────────────

def test_error_result_structure():
    from python.tasks.vision_tasks import _error_result
    result = _error_result("req-123", "inference_error", "OOM")
    assert result["request_id"]         == "req-123"
    assert result["status"]             == "error"
    assert result["error_type"]         == "inference_error"
    assert result["message"]            == "OOM"
    assert result["tasks_ran"]          == []
    assert result["total_inference_ms"] == 0.0


def test_error_result_validation_type():
    from python.tasks.vision_tasks import _error_result
    result = _error_result("x", "validation_error", "bad input")
    assert result["error_type"] == "validation_error"


# ─────────────────────────────────────────────
#  Unit: analyze_image_task
# ─────────────────────────────────────────────

def test_analyze_image_task_validation_error():
    from python.tasks.vision_tasks import analyze_image_task
    bad_payload = {"not": "a valid request"}
    result = analyze_image_task.apply(args=[bad_payload]).get()
    assert result["status"]     == "error"
    assert result["error_type"] == "validation_error"


def test_analyze_image_task_returns_dict_on_inference_error():
    from python.tasks.vision_tasks import analyze_image_task
    payload = make_request_payload()
    with patch("python.tasks.vision_tasks.asyncio.run", side_effect=RuntimeError("kaboom")):
        result = analyze_image_task.apply(args=[payload]).get()
    assert isinstance(result, dict)
    assert result["status"] == "error"


def test_analyze_image_task_success():
    from python.tasks.vision_tasks import analyze_image_task
    payload      = make_request_payload()
    fake_result  = make_vision_result_dict(payload["request_id"])
    mock_resp    = MagicMock()
    mock_resp.model_dump.return_value = fake_result

    with patch("python.tasks.vision_tasks.asyncio.run", return_value=mock_resp):
        result = analyze_image_task.apply(args=[payload]).get()

    assert result["request_id"] == payload["request_id"]
    assert result["status"]     == "success"


# ─────────────────────────────────────────────
#  Unit: _build_progress
# ─────────────────────────────────────────────

def test_build_progress_pct():
    from python.tasks.batch_tasks import _build_progress
    p = _build_progress("j", total=10, completed=3, failed=1)
    assert p["progress_pct"] == 30.0


def test_build_progress_100():
    from python.tasks.batch_tasks import _build_progress
    p = _build_progress("j", total=5, completed=5, failed=0, status="done")
    assert p["progress_pct"] == 100.0
    assert p["status"]       == "done"


def test_build_progress_zero_total():
    from python.tasks.batch_tasks import _build_progress
    p = _build_progress("j", total=0, completed=0, failed=0)
    assert p["progress_pct"] == 0.0


def test_build_progress_has_updated_at():
    from python.tasks.batch_tasks import _build_progress
    p = _build_progress("j", 10, 5, 0)
    assert "updated_at" in p


# ─────────────────────────────────────────────
#  Unit: process_batch_task
# ─────────────────────────────────────────────

@patch("python.tasks.batch_tasks.process_batch_task.update_state")
@patch("python.tasks.batch_tasks._persist_results")
@patch("python.tasks.batch_tasks.analyze_image_task")
def test_process_batch_task_all_success(mock_analyze, mock_persist, mock_update):
    from python.tasks.batch_tasks import process_batch_task

    job_id   = str(uuid.uuid4())
    requests = [make_request_payload() for _ in range(3)]
    fake_results = [make_vision_result_dict(r["request_id"]) for r in requests]

    mock_group_result = MagicMock()
    mock_group_result.get.return_value = fake_results
    mock_analyze.s.return_value = MagicMock()
    mock_persist.return_value = f"/data/parquet/{job_id}/results.parquet"

    with patch("python.tasks.batch_tasks.group") as mock_group:
        mock_group.return_value.apply.return_value = mock_group_result
        summary = process_batch_task.apply(
            args=[job_id, requests],
            kwargs={"job_name": "test_job"},
        ).get()

    assert summary["job_id"]       == job_id
    assert summary["total"]        == 3
    assert summary["progress_pct"] == 100.0
    assert summary["status"] in ("done", "done_with_errors")
    assert "elapsed_ms" in summary


@patch("python.tasks.batch_tasks.process_batch_task.update_state")
@patch("python.tasks.batch_tasks._persist_results")
@patch("python.tasks.batch_tasks.analyze_image_task")
def test_process_batch_task_with_failures(mock_analyze, mock_persist, mock_update):
    from python.tasks.batch_tasks import process_batch_task

    job_id   = str(uuid.uuid4())
    requests = [make_request_payload() for _ in range(3)]
    fake_results = [
        make_vision_result_dict(requests[0]["request_id"]),
        {"request_id": requests[1]["request_id"], "status": "error", "tasks_ran": [], "total_inference_ms": 0.0},
        make_vision_result_dict(requests[2]["request_id"]),
    ]

    mock_group_result = MagicMock()
    mock_group_result.get.return_value = fake_results
    mock_analyze.s.return_value = MagicMock()
    mock_persist.return_value = "/some/path.parquet"

    with patch("python.tasks.batch_tasks.group") as mock_group:
        mock_group.return_value.apply.return_value = mock_group_result
        summary = process_batch_task.apply(args=[job_id, requests]).get()

    assert summary["failed"]    == 1
    assert summary["completed"] == 2
    assert summary["status"]    == "done_with_errors"


@patch("python.tasks.batch_tasks.process_batch_task.update_state")
@patch("python.tasks.batch_tasks._persist_results")
@patch("python.tasks.batch_tasks.analyze_image_task")
def test_process_batch_empty_job(mock_analyze, mock_persist, mock_update):
    from python.tasks.batch_tasks import process_batch_task
    job_id = str(uuid.uuid4())
    mock_persist.return_value = ""
    summary = process_batch_task.apply(args=[job_id, []]).get()
    assert summary["total"]     == 0
    assert summary["completed"] == 0


@patch("python.tasks.batch_tasks.process_batch_task.update_state")
@patch("python.tasks.batch_tasks._persist_results")
@patch("python.tasks.batch_tasks._notify_webhook")
@patch("python.tasks.batch_tasks.analyze_image_task")
def test_process_batch_calls_webhook(mock_analyze, mock_webhook, mock_persist, mock_update):
    from python.tasks.batch_tasks import process_batch_task
    job_id      = str(uuid.uuid4())
    webhook_url = "https://example.com/webhook"
    requests    = [make_request_payload()]

    mock_group_result = MagicMock()
    mock_group_result.get.return_value = [make_vision_result_dict(requests[0]["request_id"])]
    mock_analyze.s.return_value = MagicMock()
    mock_persist.return_value   = "/path.parquet"

    with patch("python.tasks.batch_tasks.group") as mock_group:
        mock_group.return_value.apply.return_value = mock_group_result
        process_batch_task.apply(
            args=[job_id, requests],
            kwargs={"webhook_url": webhook_url},
        ).get()

    mock_webhook.assert_called_once()
    assert mock_webhook.call_args[0][0] == webhook_url


@patch("python.tasks.batch_tasks.process_batch_task.update_state")
@patch("python.tasks.batch_tasks._persist_results")
@patch("python.tasks.batch_tasks.analyze_image_task")
def test_process_batch_summary_has_timing(mock_analyze, mock_persist, mock_update):
    from python.tasks.batch_tasks import process_batch_task
    job_id   = str(uuid.uuid4())
    requests = [make_request_payload() for _ in range(2)]
    mock_group_result = MagicMock()
    mock_group_result.get.return_value = [make_vision_result_dict(r["request_id"]) for r in requests]
    mock_analyze.s.return_value = MagicMock()
    mock_persist.return_value   = "/path.parquet"

    with patch("python.tasks.batch_tasks.group") as mock_group:
        mock_group.return_value.apply.return_value = mock_group_result
        summary = process_batch_task.apply(args=[job_id, requests]).get()

    assert summary["elapsed_ms"]        >= 0
    assert summary["avg_ms_per_image"] >= 0


# ─────────────────────────────────────────────
#  Unit: _notify_webhook
# ─────────────────────────────────────────────

def test_notify_webhook_posts_correct_payload():
    from python.tasks.batch_tasks import _notify_webhook
    payload = {"job_id": "abc", "status": "done", "total": 5}

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("python.tasks.batch_tasks.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__  = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        _notify_webhook("https://example.com/wh", payload)

        mock_client.post.assert_called_once()
        assert mock_client.post.call_args[0][0] == "https://example.com/wh"
        assert mock_client.post.call_args[1]["json"] == payload


def test_notify_webhook_swallows_exceptions():
    from python.tasks.batch_tasks import _notify_webhook
    with patch("python.tasks.batch_tasks.httpx.Client", side_effect=Exception("network error")):
        _notify_webhook("https://bad-url.test", {"job_id": "x"})
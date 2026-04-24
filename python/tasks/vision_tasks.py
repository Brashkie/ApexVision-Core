"""
ApexVision-Core — Celery Vision Tasks
Tasks individuales de análisis de imagen con retry, timeout y error handling.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from celery import Task
from celery.exceptions import MaxRetriesExceededError
from loguru import logger

from python.celery_app import celery_app


class AnalyzeImageTask(Task):
    """
    Custom Task base class con pipeline singleton.
    El VisionPipeline se inicializa una sola vez por worker process —
    evita recargar modelos ML en cada task.
    """
    abstract = True
    _pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from python.core.pipeline import VisionPipeline
            self._pipeline = VisionPipeline()
        return self._pipeline


@celery_app.task(
    bind=True,
    base=AnalyzeImageTask,
    queue="vision",
    name="tasks.analyze_image",
    max_retries=3,
    default_retry_delay=2,       # seconds between retries
    soft_time_limit=280,         # SoftTimeLimitExceeded → retry
    time_limit=300,              # hard kill after 5min
    acks_late=True,              # ack only after success (safe against worker crash)
    reject_on_worker_lost=True,
)
def analyze_image_task(self, request_payload: dict) -> dict:
    """
    Single-image vision analysis task.
    Retries on transient errors (OOM, model timeout).
    Fails immediately on validation errors (bad input).
    """
    from python.schemas.vision import VisionRequest

    t0 = time.perf_counter()
    request_id = request_payload.get("request_id", "unknown")

    logger.info(f"[{request_id}] analyze_image_task start | attempt={self.request.retries + 1}")

    try:
        request = VisionRequest.model_validate(request_payload)
    except Exception as e:
        # Validation errors are not retried — bad input is bad input
        logger.error(f"[{request_id}] Validation failed: {e}")
        return _error_result(request_id, "validation_error", str(e))

    try:
        result = asyncio.run(self.pipeline.run(request))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[{request_id}] done in {elapsed:.1f}ms")
        return result.model_dump()

    except MemoryError as exc:
        logger.warning(f"[{request_id}] OOM — retrying ({self.request.retries + 1}/{self.max_retries})")
        raise self.retry(exc=exc, countdown=5 * (self.request.retries + 1))

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(f"[{request_id}] failed in {elapsed:.1f}ms: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)

        return _error_result(request_id, "inference_error", str(exc))


def _error_result(request_id: str, error_type: str, message: str) -> dict:
    return {
        "request_id": request_id,
        "status":     "error",
        "error_type": error_type,
        "message":    message,
        "tasks_ran":  [],
        "total_inference_ms": 0.0,
    }

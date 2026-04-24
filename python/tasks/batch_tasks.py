"""
ApexVision-Core — Celery Batch Tasks
Procesamiento async de lotes de imágenes con:
  - Concurrencia configurable (subtasks paralelas)
  - Progress tracking en tiempo real via Redis
  - Resultados a Parquet (siempre) + Delta Lake (opcional)
  - Webhook notification al completar
  - Retry por item fallido, no por todo el batch
  - Estadísticas del batch: timings, detecciones, errores
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from celery import group
from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from python.celery_app import celery_app
from python.storage.parquet_store import ParquetStore
from python.tasks.vision_tasks import analyze_image_task


# ─────────────────────────────────────────────
#  Batch progress schema (stored in Redis)
# ─────────────────────────────────────────────

def _build_progress(
    job_id: str,
    total: int,
    completed: int,
    failed: int,
    status: str = "running",
    result_path: str = "",
    error_summary: str = "",
) -> dict:
    pct = round(completed / max(total, 1) * 100, 1)
    return {
        "job_id":        job_id,
        "status":        status,
        "total":         total,
        "completed":     completed,
        "failed":        failed,
        "progress_pct":  pct,
        "result_path":   result_path,
        "error_summary": error_summary,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────
#  Main batch task
# ─────────────────────────────────────────────

@celery_app.task(
    bind=True,
    queue="batch",
    name="tasks.process_batch",
    max_retries=0,               # batch itself doesn't retry — individual items do
    soft_time_limit=3540,        # 59 min soft limit
    time_limit=3600,             # 60 min hard kill
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_batch_task(
    self,
    job_id: str,
    requests: list[dict],
    job_name: str = "",
    webhook_url: str | None = None,
    store_to_delta: bool = False,
    concurrency: int = 4,
) -> dict:
    """
    Process a batch of image analysis requests.

    Args:
        job_id:          Unique job identifier
        requests:        List of VisionRequest dicts
        job_name:        Human-readable name for the job
        webhook_url:     Optional URL to POST results when done
        store_to_delta:  Also persist to Delta Lake (slower, ACID)
        concurrency:     How many subtasks to run in parallel
    """
    total     = len(requests)
    completed = 0
    failed    = 0
    results   = []
    start_ts  = time.perf_counter()

    logger.info(
        f"Batch [{job_id}] start | "
        f"name={job_name!r} | total={total} | concurrency={concurrency}"
    )

    # Initial progress state
    self.update_state(
        state="STARTED",
        meta=_build_progress(job_id, total, 0, 0, "running"),
    )

    # ── Process in chunks of `concurrency` ──────────────────────────
    for chunk_start in range(0, total, concurrency):
        chunk = requests[chunk_start: chunk_start + concurrency]

        # Dispatch chunk as a Celery group (parallel subtasks)
        chord_tasks = group(
            analyze_image_task.s(req).set(queue="vision")
            for req in chunk
        )

        try:
            chunk_results = chord_tasks.apply().get(
                timeout=300,
                propagate=False,       # don't raise on subtask failure
            )
        except SoftTimeLimitExceeded:
            logger.error(f"Batch [{job_id}] soft time limit hit at item {chunk_start}")
            break
        except Exception as e:
            logger.error(f"Batch [{job_id}] chunk dispatch failed: {e}")
            chunk_results = [None] * len(chunk)

        # Collect chunk results
        for i, result in enumerate(chunk_results):
            req_id = chunk[i].get("request_id", f"item-{chunk_start + i}")

            if result is None or isinstance(result, Exception):
                failed += 1
                results.append({
                    "request_id": req_id,
                    "status":     "error",
                    "error":      str(result) if result else "task_failed",
                    "tasks_ran":  [],
                    "total_inference_ms": 0.0,
                })
            else:
                if result.get("status") == "error":
                    failed += 1
                else:
                    completed += 1
                results.append(result)

        # Update progress after each chunk
        self.update_state(
            state="PROGRESS",
            meta=_build_progress(
                job_id, total,
                completed, failed,
                "running",
            ),
        )
        logger.debug(
            f"Batch [{job_id}] chunk done | "
            f"completed={completed}/{total} failed={failed}"
        )

    # ── Persist results ──────────────────────────────────────────────
    result_path = ""
    try:
        result_path = _persist_results(
            job_id, job_name, results, store_to_delta
        )
    except Exception as e:
        logger.error(f"Batch [{job_id}] persist failed: {e}")

    # ── Build final summary ──────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start_ts) * 1000
    final_status = "done" if failed == 0 else "done_with_errors"

    summary = {
        "job_id":          job_id,
        "job_name":        job_name,
        "status":          final_status,
        "total":           total,
        "completed":       completed,
        "failed":          failed,
        "progress_pct":    100.0,
        "result_path":     result_path,
        "elapsed_ms":      round(elapsed_ms, 2),
        "avg_ms_per_image": round(elapsed_ms / max(total, 1), 2),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }

    self.update_state(state="SUCCESS", meta=summary)
    logger.info(
        f"Batch [{job_id}] finished | "
        f"completed={completed} failed={failed} | "
        f"elapsed={elapsed_ms/1000:.1f}s | path={result_path}"
    )

    # ── Webhook notification ─────────────────────────────────────────
    if webhook_url:
        _notify_webhook(webhook_url, summary)

    return summary


# ─────────────────────────────────────────────
#  Persist helpers
# ─────────────────────────────────────────────

def _persist_results(
    job_id: str,
    job_name: str,
    results: list[dict],
    store_to_delta: bool,
) -> str:
    """Write results to Parquet (always) and optionally Delta Lake."""
    store = ParquetStore()
    result_path = store.write_batch_results(job_id, results, job_name)

    if store_to_delta:
        try:
            _persist_to_delta(job_id, results)
        except Exception as e:
            logger.warning(f"Delta persist skipped for batch {job_id}: {e}")

    return result_path


def _persist_to_delta(job_id: str, results: list[dict]) -> None:
    """Write successful results to Delta Lake for ACID analytics."""
    from python.storage.delta_store import DeltaStore
    from python.schemas.vision import VisionResponse

    store = DeltaStore()

    for result in results:
        if result.get("status") == "error":
            continue
        try:
            response = VisionResponse.model_validate(result)
            asyncio.run(store.write_result(response))
        except Exception as e:
            logger.debug(f"Delta item skip (job={job_id}): {e}")


# ─────────────────────────────────────────────
#  Webhook notification
# ─────────────────────────────────────────────

def _notify_webhook(url: str, payload: dict) -> None:
    """POST batch completion payload to webhook URL."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "ApexVision-Core/2.0"},
            )
            logger.info(f"Webhook sent → {url} | status={resp.status_code}")
    except Exception as e:
        logger.warning(f"Webhook failed ({url}): {e}")


# ─────────────────────────────────────────────
#  Maintenance task
# ─────────────────────────────────────────────

@celery_app.task(
    queue="batch",
    name="tasks.compact_delta",
    max_retries=1,
)
def compact_delta_task(table: str = "vision_results") -> dict:
    """
    Periodic Delta Lake compaction task.
    Schedule via Celery Beat to run nightly.
    """
    from python.storage.delta_store import DeltaStore
    store = DeltaStore()
    logger.info(f"Compacting Delta table: {table}")
    result = store.compact(table)
    logger.info(f"Compact done: {result}")
    return result


@celery_app.task(
    queue="batch",
    name="tasks.vacuum_delta",
    max_retries=1,
)
def vacuum_delta_task(
    table: str = "vision_results",
    retention_hours: int = 168,
) -> dict:
    """
    Periodic Delta Lake vacuum task.
    Removes Parquet files older than retention_hours (default: 7 days).
    """
    from python.storage.delta_store import DeltaStore
    store = DeltaStore()
    logger.info(f"Vacuuming Delta table: {table} | retention={retention_hours}h")
    deleted = store.vacuum(table, retention_hours)
    logger.info(f"Vacuum done: {len(deleted)} files deleted")
    return {"table": table, "deleted_files": len(deleted)}

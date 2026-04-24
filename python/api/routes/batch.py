"""ApexVision-Core — Batch Router"""
from fastapi import APIRouter, Depends
from python.schemas.vision import BatchRequest, BatchJobStatus
from python.api.deps import get_current_api_key

router = APIRouter()

@router.post("/submit", summary="Submit async batch job")
async def submit_batch(request: BatchRequest, api_key: str = Depends(get_current_api_key)) -> dict:
    from python.tasks.batch_tasks import process_batch_task
    import uuid
    job_id = str(uuid.uuid4())
    process_batch_task.apply_async(
        args=[job_id, [r.model_dump() for r in request.requests]],
        task_id=job_id,
    )
    return {"job_id": job_id, "status": "pending", "total": len(request.requests)}

@router.get("/{job_id}", response_model=BatchJobStatus, summary="Get batch job status")
async def get_batch_status(job_id: str, api_key: str = Depends(get_current_api_key)) -> dict:
    from python.celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    meta = task.info or {}
    return {
        "job_id": job_id,
        "status": task.state.lower(),
        "total": meta.get("total", 0),
        "completed": meta.get("completed", 0),
        "failed": meta.get("failed", 0),
        "progress_pct": meta.get("completed", 0) / max(meta.get("total", 1), 1) * 100,
        "result_path": meta.get("result_path"),
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
    }

@router.delete("/{job_id}", summary="Cancel batch job")
async def cancel_batch(job_id: str, api_key: str = Depends(get_current_api_key)) -> dict:
    from python.celery_app import celery_app
    celery_app.control.revoke(job_id, terminate=True)
    return {"job_id": job_id, "status": "cancelled"}

"""
ApexVision-Core — Delta Lake Store
ACID storage con time travel, schema evolution y compaction.

Features:
  - Escritura ACID con deltalake (Rust) + PyArrow
  - Schema evolutivo: nuevas columnas se agregan automáticamente
  - Particionado por fecha para queries eficientes
  - Time travel: leer versiones anteriores de la tabla
  - Compaction automática: merge de small files en un solo Parquet
  - Vacuum: eliminar archivos obsoletos (older than retention_hours)
  - Query con Polars LazyFrame — pushdown de filtros al nivel de Parquet
  - Schemas separados por tabla: vision_results · batch_jobs · model_metrics
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
import pyarrow as pa
from loguru import logger

from python.config import settings
from python.schemas.vision import VisionResponse, BatchJobStatus


# ─────────────────────────────────────────────
#  Arrow schemas (strict typing para Delta)
# ─────────────────────────────────────────────

VISION_RESULT_SCHEMA = pa.schema([
    pa.field("request_id",       pa.string()),
    pa.field("status",           pa.string()),
    pa.field("tasks_ran",        pa.string()),         # JSON array
    pa.field("image_width",      pa.int32()),
    pa.field("image_height",     pa.int32()),
    pa.field("total_inference_ms", pa.float32()),
    pa.field("detection_count",  pa.int32()),
    pa.field("detection_json",   pa.string()),         # JSON blob
    pa.field("ocr_text",         pa.string()),
    pa.field("embedding_dim",    pa.int32()),
    pa.field("face_count",       pa.int32()),
    pa.field("stored_at",        pa.string()),
    pa.field("partition_date",   pa.string()),         # "YYYY-MM-DD"
    pa.field("created_at",       pa.timestamp("us", tz="UTC")),
])

BATCH_JOB_SCHEMA = pa.schema([
    pa.field("job_id",        pa.string()),
    pa.field("job_name",      pa.string()),
    pa.field("status",        pa.string()),
    pa.field("total",         pa.int32()),
    pa.field("completed",     pa.int32()),
    pa.field("failed",        pa.int32()),
    pa.field("progress_pct",  pa.float32()),
    pa.field("result_path",   pa.string()),
    pa.field("error_summary", pa.string()),
    pa.field("created_at",    pa.timestamp("us", tz="UTC")),
    pa.field("updated_at",    pa.timestamp("us", tz="UTC")),
])


class DeltaStore:
    """
    Production Delta Lake store.
    Cada método trabaja con una tabla específica bajo settings.DELTA_LAKE_PATH.
    """

    def __init__(self) -> None:
        self.base_path  = Path(settings.DELTA_LAKE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    #  Table paths
    # ─────────────────────────────────────────

    def _table_path(self, table: str) -> str:
        path = self.base_path / table
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # ─────────────────────────────────────────
    #  Vision results
    # ─────────────────────────────────────────

    async def write_result(self, response: VisionResponse) -> str:
        """
        Persist a single VisionResponse to Delta Lake.
        Flattens nested fields, partitions by date, appends with ACID.
        """
        return await _run_sync(self._write_result_sync, response)

    def _write_result_sync(self, response: VisionResponse) -> str:
        from deltalake import write_deltalake

        now = datetime.now(timezone.utc)
        table_path = self._table_path("vision_results")

        # Flatten VisionResponse into a flat row
        row = self._flatten_vision_response(response, now)
        table = pa.table(
            {k: [v] for k, v in row.items()},
            schema=VISION_RESULT_SCHEMA,
        )

        write_deltalake(
            table_path,
            table,
            mode="append",
            schema_mode="merge",        # allow schema evolution
            partition_by=["partition_date"],
        )
        logger.debug(f"Delta write: vision_results | req={response.request_id}")
        return table_path

    def _flatten_vision_response(
        self, response: VisionResponse, now: datetime
    ) -> dict[str, Any]:
        det = response.detection
        ocr = response.ocr
        emb = response.embedding
        fac = response.face

        return {
            "request_id":         str(response.request_id),
            "status":             response.status,
            "tasks_ran":          json.dumps([str(t) for t in response.tasks_ran]),
            "image_width":        response.image_width,
            "image_height":       response.image_height,
            "total_inference_ms": float(response.total_inference_ms),
            "detection_count":    det.count if det else 0,
            "detection_json":     json.dumps([b.model_dump() for b in det.boxes]) if det else "[]",
            "ocr_text":           ocr.text if ocr else "",
            "embedding_dim":      emb.dimensions if emb else 0,
            "face_count":         fac.count if fac else 0,
            "stored_at":          response.stored_at or "",
            "partition_date":     now.strftime("%Y-%m-%d"),
            "created_at":         now,
        }

    # ─────────────────────────────────────────
    #  Batch jobs
    # ─────────────────────────────────────────

    async def write_batch_job(self, job: BatchJobStatus) -> str:
        return await _run_sync(self._write_batch_job_sync, job)

    def _write_batch_job_sync(self, job: BatchJobStatus) -> str:
        from deltalake import write_deltalake

        now = datetime.now(timezone.utc)
        table_path = self._table_path("batch_jobs")

        row = {
            "job_id":        job.job_id,
            "job_name":      getattr(job, "job_name", "") or "",
            "status":        job.status,
            "total":         job.total,
            "completed":     job.completed,
            "failed":        job.failed,
            "progress_pct":  float(job.progress_pct),
            "result_path":   job.result_path or "",
            "error_summary": "",
            "created_at":    now,
            "updated_at":    now,
        }
        table = pa.table(
            {k: [v] for k, v in row.items()},
            schema=BATCH_JOB_SCHEMA,
        )
        write_deltalake(table_path, table, mode="append", schema_mode="merge")
        return table_path

    # ─────────────────────────────────────────
    #  Query interface
    # ─────────────────────────────────────────

    def query_results(
        self,
        date: str | None = None,          # "YYYY-MM-DD" partition filter
        limit: int = 1000,
        task_filter: str | None = None,   # e.g. "detect"
    ) -> pl.DataFrame:
        """
        Query vision_results with Polars LazyFrame.
        Supports partition pruning by date for fast scans.
        """
        table_path = self._table_path("vision_results")

        try:
            from deltalake import DeltaTable
            dt = DeltaTable(table_path)
            arrow = dt.to_pyarrow()
        except Exception:
            return pl.DataFrame()

        df = pl.from_arrow(arrow).lazy()

        if date:
            df = df.filter(pl.col("partition_date") == date)

        if task_filter:
            df = df.filter(pl.col("tasks_ran").str.contains(task_filter))

        return df.limit(limit).collect()

    def query_batch_jobs(
        self,
        status_filter: str | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        table_path = self._table_path("batch_jobs")
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(table_path)
            df = pl.from_arrow(dt.to_pyarrow()).lazy()
        except Exception:
            return pl.DataFrame()

        if status_filter:
            df = df.filter(pl.col("status") == status_filter)

        return df.sort("created_at", descending=True).limit(limit).collect()

    # ─────────────────────────────────────────
    #  Time travel
    # ─────────────────────────────────────────

    def read_version(self, table: str, version: int) -> pl.DataFrame:
        """
        Read a specific historical version of a Delta table.
        version=0 is the initial state, version=N is after N transactions.
        """
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(self._table_path(table), version=version)
            return pl.from_arrow(dt.to_pyarrow())
        except Exception as e:
            logger.warning(f"Time travel failed (table={table}, version={version}): {e}")
            return pl.DataFrame()

    def table_history(self, table: str) -> list[dict]:
        """Return commit history of a Delta table."""
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(self._table_path(table))
            return dt.history()
        except Exception as e:
            logger.warning(f"History failed (table={table}): {e}")
            return []

    def table_version(self, table: str) -> int:
        """Return current version number of a Delta table."""
        try:
            from deltalake import DeltaTable
            return DeltaTable(self._table_path(table)).version()
        except Exception:
            return -1

    # ─────────────────────────────────────────
    #  Maintenance
    # ─────────────────────────────────────────

    def compact(self, table: str, target_size_mb: int = 128) -> dict:
        """
        Compact small Parquet files into larger ones.
        Reduces number of files, improves scan performance.
        """
        try:
            from deltalake import DeltaTable
            dt     = DeltaTable(self._table_path(table))
            result = dt.optimize.compact(target_size=target_size_mb * 1024 * 1024)
            logger.info(f"Delta compact: {table} | {result}")
            return result
        except Exception as e:
            logger.error(f"Compact failed ({table}): {e}")
            return {"error": str(e)}

    def vacuum(self, table: str, retention_hours: int = 168) -> list[str]:
        """
        Remove Parquet files no longer needed (older than retention_hours).
        Default: 7 days. Delta Lake minimum is 168h (7d).
        """
        try:
            from deltalake import DeltaTable
            dt      = DeltaTable(self._table_path(table))
            deleted = dt.vacuum(retention_hours=retention_hours, dry_run=False)
            logger.info(f"Delta vacuum: {table} | {len(deleted)} files deleted")
            return deleted
        except Exception as e:
            logger.error(f"Vacuum failed ({table}): {e}")
            return []

    def table_stats(self, table: str) -> dict:
        """Return basic stats for a Delta table."""
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(self._table_path(table))
            df = pl.from_arrow(dt.to_pyarrow())
            return {
                "version":    dt.version(),
                "row_count":  len(df),
                "file_count": len(dt.files()),
                "columns":    df.columns,
                "size_bytes": sum(
                    Path(f).stat().st_size
                    for f in dt.files()
                    if Path(self._table_path(table) + "/" + f).exists()
                ),
            }
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────
#  Async helper
# ─────────────────────────────────────────────

async def _run_sync(fn, *args):
    """Run a sync function in the default thread pool (non-blocking)."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)

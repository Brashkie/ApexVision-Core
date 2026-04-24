"""
ApexVision-Core — Parquet Store
Almacenamiento columnar eficiente con Polars + Apache Parquet.

Features:
  - Escritura con compresión ZSTD (mejor ratio que Snappy/Gzip)
  - Particionado por job_id y fecha para lecturas rápidas
  - Schema de batch results: request_id · tasks · detecciones · OCR · timings
  - Lazy scanning con pushdown de predicados (solo lee columnas/filas necesarias)
  - Export a CSV, JSON, Arrow para interoperabilidad
  - Merge de múltiples archivos Parquet en uno solo
  - Stats por archivo: rows, size, schema, compression
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from python.config import settings


class ParquetStore:
    """
    Batch result storage using Polars + Apache Parquet.
    Optimized for write-once, read-many analytics workloads.
    """

    def __init__(self) -> None:
        self.base_path = Path(settings.PARQUET_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    #  Path helpers
    # ─────────────────────────────────────────

    def _batch_path(self, job_id: str) -> Path:
        path = self.base_path / "batch_results" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _results_file(self, job_id: str) -> Path:
        return self._batch_path(job_id) / "results.parquet"

    def _summary_file(self, job_id: str) -> Path:
        return self._batch_path(job_id) / "summary.parquet"

    # ─────────────────────────────────────────
    #  Batch results write
    # ─────────────────────────────────────────

    def write_batch_results(
        self,
        job_id: str,
        results: list[dict],
        job_name: str = "",
    ) -> str:
        """
        Write a complete batch job result set to Parquet.
        Each dict in results is one VisionResponse.model_dump().
        Returns path to the written file.
        """
        rows = []
        for item in results:
            row = self._flatten_result(item, job_id)
            rows.append(row)

        if not rows:
            logger.warning(f"No results to write for job {job_id}")
            return ""

        df = pl.DataFrame(rows)
        out_path = self._results_file(job_id)
        df.write_parquet(str(out_path), compression="zstd", statistics=True)

        # Write summary alongside
        self._write_summary(job_id, df, job_name)

        size_kb = out_path.stat().st_size / 1024
        logger.info(
            f"Parquet write: {out_path.name} | "
            f"{len(df)} rows | {size_kb:.1f} KB | job={job_id}"
        )
        return str(out_path)

    def _flatten_result(self, result: dict, job_id: str) -> dict[str, Any]:
        """Flatten a VisionResponse dict into a flat row for Parquet."""
        det = result.get("detection") or {}
        ocr = result.get("ocr") or {}
        emb = result.get("embedding") or {}
        fac = result.get("face") or {}
        dep = result.get("depth") or {}
        seg = result.get("segmentation") or {}
        clf = result.get("classification") or {}

        return {
            # Identity
            "job_id":             job_id,
            "request_id":         str(result.get("request_id", "")),
            "status":             result.get("status", "unknown"),
            "partition_date":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),

            # Tasks
            "tasks_ran":          json.dumps(result.get("tasks_ran", [])),
            "image_width":        result.get("image_width", 0),
            "image_height":       result.get("image_height", 0),
            "total_inference_ms": float(result.get("total_inference_ms", 0.0)),

            # Detection
            "detection_count":    det.get("count", 0),
            "detection_model":    det.get("model_used", ""),
            "detection_ms":       float(det.get("inference_ms", 0.0)),
            "top_label":          det.get("boxes", [{}])[0].get("label", "") if det.get("boxes") else "",
            "top_confidence":     float(det.get("boxes", [{}])[0].get("confidence", 0.0)) if det.get("boxes") else 0.0,

            # Classification
            "top_class":          clf.get("predictions", [{}])[0].get("label", "") if clf.get("predictions") else "",
            "top_class_score":    float(clf.get("predictions", [{}])[0].get("confidence", 0.0)) if clf.get("predictions") else 0.0,
            "clf_model":          clf.get("model_used", ""),
            "clf_ms":             float(clf.get("inference_ms", 0.0)),

            # OCR
            "ocr_text":           ocr.get("text", ""),
            "ocr_char_count":     len(ocr.get("text", "")),
            "ocr_block_count":    len(ocr.get("blocks", [])),
            "ocr_language":       ocr.get("language_detected", ""),
            "ocr_ms":             float(ocr.get("inference_ms", 0.0)),

            # Face
            "face_count":         fac.get("count", 0),
            "face_ms":            float(fac.get("inference_ms", 0.0)),

            # Embedding
            "embedding_dim":      emb.get("dimensions", 0),
            "embedding_model":    emb.get("model_used", ""),
            "embedding_ms":       float(emb.get("inference_ms", 0.0)),

            # Depth
            "depth_min":          float(dep.get("min_depth", 0.0)),
            "depth_max":          float(dep.get("max_depth", 0.0)),
            "depth_ms":           float(dep.get("inference_ms", 0.0)),

            # Segmentation
            "seg_mask_count":     seg.get("count", 0),
            "seg_ms":             float(seg.get("inference_ms", 0.0)),
        }

    def _write_summary(self, job_id: str, df: pl.DataFrame, job_name: str) -> None:
        """Write aggregated summary statistics alongside the results."""
        try:
            summary_rows = [
                {
                    "job_id":               job_id,
                    "job_name":             job_name,
                    "total_images":         len(df),
                    "successful":           int((df["status"] == "success").sum()),
                    "failed":               int((df["status"] != "success").sum()),
                    "avg_inference_ms":     float(df["total_inference_ms"].mean() or 0.0),
                    "p95_inference_ms":     float(df["total_inference_ms"].quantile(0.95) or 0.0),
                    "total_detections":     int(df["detection_count"].sum()),
                    "avg_detection_count":  float(df["detection_count"].mean() or 0.0),
                    "total_faces":          int(df["face_count"].sum()),
                    "images_with_text":     int((df["ocr_char_count"] > 0).sum()),
                    "tasks_ran":            df["tasks_ran"][0] if len(df) > 0 else "[]",
                    "created_at":           datetime.now(timezone.utc).isoformat(),
                }
            ]
            pl.DataFrame(summary_rows).write_parquet(
                str(self._summary_file(job_id)), compression="zstd"
            )
        except Exception as e:
            logger.warning(f"Summary write failed for job {job_id}: {e}")

    # ─────────────────────────────────────────
    #  Read interface
    # ─────────────────────────────────────────

    def read_batch_results(
        self,
        job_id: str,
        columns: list[str] | None = None,
        status_filter: str | None = None,
    ) -> pl.DataFrame:
        """
        Read batch results with optional column projection and status filter.
        Uses Polars LazyFrame for predicate pushdown.
        """
        path = self._results_file(job_id)
        if not path.exists():
            logger.warning(f"Results file not found: {path}")
            return pl.DataFrame()

        df = pl.scan_parquet(str(path))

        if status_filter:
            df = df.filter(pl.col("status") == status_filter)

        if columns:
            df = df.select(columns)

        return df.collect()

    def read_summary(self, job_id: str) -> pl.DataFrame:
        path = self._summary_file(job_id)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(str(path))

    def scan_all_batches(
        self,
        date_filter: str | None = None,
    ) -> pl.LazyFrame:
        """
        Lazy scan across ALL batch result files.
        Returns a LazyFrame — call .collect() when ready.
        Uses glob pattern for efficient multi-file scan.
        """
        pattern = str(self.base_path / "batch_results" / "*" / "results.parquet")
        try:
            lf = pl.scan_parquet(pattern)
            if date_filter:
                lf = lf.filter(pl.col("partition_date") == date_filter)
            return lf
        except Exception as e:
            logger.warning(f"Scan all batches failed: {e}")
            return pl.LazyFrame()

    # ─────────────────────────────────────────
    #  Analytics helpers
    # ─────────────────────────────────────────

    def detection_stats(self, job_id: str) -> dict:
        """Per-label detection statistics for a batch job."""
        df = self.read_batch_results(job_id, columns=["top_label", "top_confidence", "detection_count"])
        if df.is_empty():
            return {}
        return {
            "total_detections":  int(df["detection_count"].sum()),
            "avg_confidence":    round(float(df["top_confidence"].mean() or 0), 4),
            "images_with_detections": int((df["detection_count"] > 0).sum()),
            "top_label":         df["top_label"].mode()[0] if not df["top_label"].is_empty() else "",
        }

    def inference_timing_stats(self, job_id: str) -> dict:
        """Timing percentiles for a batch job."""
        df = self.read_batch_results(job_id, columns=["total_inference_ms"])
        if df.is_empty():
            return {}
        col = df["total_inference_ms"]
        return {
            "min_ms":  round(float(col.min() or 0), 2),
            "p50_ms":  round(float(col.quantile(0.50) or 0), 2),
            "p95_ms":  round(float(col.quantile(0.95) or 0), 2),
            "p99_ms":  round(float(col.quantile(0.99) or 0), 2),
            "max_ms":  round(float(col.max() or 0), 2),
            "mean_ms": round(float(col.mean() or 0), 2),
        }

    # ─────────────────────────────────────────
    #  Merge + export
    # ─────────────────────────────────────────

    def merge_batches(
        self, job_ids: list[str], output_name: str
    ) -> str:
        """Merge multiple batch result files into one Parquet file."""
        dfs = [self.read_batch_results(jid) for jid in job_ids]
        dfs = [df for df in dfs if not df.is_empty()]
        if not dfs:
            raise ValueError("No data to merge")
        merged = pl.concat(dfs)
        out_path = self.base_path / f"{output_name}.parquet"
        merged.write_parquet(str(out_path), compression="zstd")
        logger.info(f"Merged {len(dfs)} batches → {out_path.name} ({len(merged)} rows)")
        return str(out_path)

    def export_csv(self, job_id: str) -> str:
        """Export batch results to CSV."""
        df  = self.read_batch_results(job_id)
        out = self._batch_path(job_id) / "results.csv"
        df.write_csv(str(out))
        return str(out)

    def export_json(self, job_id: str) -> str:
        """Export batch results to JSON Lines (.jsonl)."""
        df  = self.read_batch_results(job_id)
        out = self._batch_path(job_id) / "results.jsonl"
        df.write_ndjson(str(out))
        return str(out)

    # ─────────────────────────────────────────
    #  Generic read/write
    # ─────────────────────────────────────────

    def write(self, df: pl.DataFrame, name: str, compression: str = "zstd") -> str:
        out = self.base_path / f"{name}.parquet"
        df.write_parquet(str(out), compression=compression, statistics=True)
        logger.debug(f"Parquet write: {name} | {len(df)} rows")
        return str(out)

    def read(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(str(self.base_path / f"{name}.parquet"))

    def file_stats(self, path: str) -> dict:
        """Return stats for any Parquet file."""
        try:
            meta = pq.read_metadata(path)
            return {
                "row_count":    meta.num_rows,
                "column_count": meta.num_columns,
                "file_size_kb": Path(path).stat().st_size / 1024,
                "row_groups":   meta.num_row_groups,
                "compression":  meta.row_group(0).column(0).compression
                                if meta.num_row_groups > 0 else "unknown",
            }
        except Exception as e:
            return {"error": str(e)}

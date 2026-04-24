"""
ApexVision-Core — Storage Tests
ParquetStore y DeltaStore testeados con tmp_path de pytest.
DeltaStore tests mockean deltalake para no requerir instalación.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from python.storage.parquet_store import ParquetStore


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    """ParquetStore with isolated temp directory."""
    monkeypatch.setattr("python.storage.parquet_store.settings.PARQUET_PATH", str(tmp_path))
    return ParquetStore()


def make_vision_result(
    request_id: str | None = None,
    status: str = "success",
    detection_count: int = 3,
    ocr_text: str = "",
    tasks: list = None,
) -> dict:
    """Synthetic VisionResponse.model_dump() for testing."""
    rid = request_id or str(uuid.uuid4())
    return {
        "request_id":         rid,
        "status":             status,
        "tasks_ran":          tasks or ["detect"],
        "image_width":        640,
        "image_height":       480,
        "total_inference_ms": 45.2,
        "detection": {
            "count":        detection_count,
            "model_used":   "yolov11n.pt",
            "inference_ms": 38.1,
            "boxes": [
                {"label": "person", "confidence": 0.92,
                 "x1": 10, "y1": 10, "x2": 200, "y2": 400,
                 "width": 190, "height": 390, "label_id": 0}
            ] * detection_count,
        } if detection_count > 0 else None,
        "ocr": {
            "text":              ocr_text,
            "blocks":            [],
            "language_detected": "en" if ocr_text else "",
            "inference_ms":      22.0,
        } if ocr_text else None,
        "classification": None,
        "face":           None,
        "embedding":      None,
        "depth":          None,
        "segmentation":   None,
        "stored_at":      None,
    }


def make_job_id() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
#  ParquetStore — write_batch_results
# ─────────────────────────────────────────────

def test_write_batch_results_creates_file(store, tmp_path):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(5)]
    path    = store.write_batch_results(job_id, results, "test_job")
    assert path != ""
    assert Path(path).exists()


def test_write_batch_results_correct_row_count(store):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(10)]
    path    = store.write_batch_results(job_id, results, "job")
    df      = pl.read_parquet(path)
    assert len(df) == 10


def test_write_batch_results_empty_returns_empty_string(store):
    job_id = make_job_id()
    path   = store.write_batch_results(job_id, [], "empty_job")
    assert path == ""


def test_write_batch_results_creates_summary(store, tmp_path):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(5)]
    store.write_batch_results(job_id, results, "job")
    summary_path = store._summary_file(job_id)
    assert summary_path.exists()


def test_write_batch_results_mixed_status(store):
    job_id = make_job_id()
    results = [
        make_vision_result(status="success"),
        make_vision_result(status="error"),
        make_vision_result(status="success"),
    ]
    path = store.write_batch_results(job_id, results)
    df   = pl.read_parquet(path)
    assert len(df) == 3
    assert int((df["status"] == "success").sum()) == 2
    assert int((df["status"] == "error").sum())   == 1


# ─────────────────────────────────────────────
#  ParquetStore — flatten_result
# ─────────────────────────────────────────────

def test_flatten_result_has_all_columns(store):
    row = store._flatten_result(make_vision_result(), "job-1")
    expected_cols = [
        "job_id", "request_id", "status", "partition_date",
        "tasks_ran", "image_width", "image_height", "total_inference_ms",
        "detection_count", "detection_model", "detection_ms",
        "top_label", "top_confidence",
        "ocr_text", "ocr_char_count", "ocr_block_count", "ocr_language",
        "face_count", "embedding_dim", "depth_min", "depth_max", "seg_mask_count",
    ]
    for col in expected_cols:
        assert col in row, f"Missing column: {col}"


def test_flatten_result_detection_count(store):
    row = store._flatten_result(make_vision_result(detection_count=5), "j")
    assert row["detection_count"] == 5


def test_flatten_result_ocr_text(store):
    row = store._flatten_result(make_vision_result(ocr_text="Hello World"), "j")
    assert row["ocr_text"] == "Hello World"
    assert row["ocr_char_count"] == 11


def test_flatten_result_none_detection(store):
    result = make_vision_result(detection_count=0)
    row    = store._flatten_result(result, "j")
    assert row["detection_count"] == 0
    assert row["top_label"] == ""
    assert row["top_confidence"] == 0.0


def test_flatten_result_job_id(store):
    row = store._flatten_result(make_vision_result(), "my-job-123")
    assert row["job_id"] == "my-job-123"


def test_flatten_result_partition_date_format(store):
    row  = store._flatten_result(make_vision_result(), "j")
    date = row["partition_date"]
    # Should be YYYY-MM-DD
    parts = date.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4


# ─────────────────────────────────────────────
#  ParquetStore — read_batch_results
# ─────────────────────────────────────────────

def test_read_batch_results_returns_dataframe(store):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(5)]
    store.write_batch_results(job_id, results)
    df = store.read_batch_results(job_id)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5


def test_read_batch_results_status_filter(store):
    job_id  = make_job_id()
    results = [
        make_vision_result(status="success"),
        make_vision_result(status="error"),
        make_vision_result(status="success"),
    ]
    store.write_batch_results(job_id, results)
    df = store.read_batch_results(job_id, status_filter="success")
    assert len(df) == 2
    assert all(df["status"] == "success")


def test_read_batch_results_column_projection(store):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(3)]
    store.write_batch_results(job_id, results)
    df = store.read_batch_results(job_id, columns=["request_id", "status"])
    assert df.columns == ["request_id", "status"]


def test_read_batch_results_missing_file_returns_empty(store):
    df = store.read_batch_results("nonexistent-job-id")
    assert isinstance(df, pl.DataFrame)
    assert df.is_empty()


# ─────────────────────────────────────────────
#  ParquetStore — analytics helpers
# ─────────────────────────────────────────────

def test_detection_stats(store):
    job_id  = make_job_id()
    results = [make_vision_result(detection_count=i) for i in range(1, 6)]
    store.write_batch_results(job_id, results)
    stats = store.detection_stats(job_id)
    assert stats["total_detections"] == sum(range(1, 6))
    assert "avg_confidence" in stats
    assert "images_with_detections" in stats


def test_inference_timing_stats(store):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(10)]
    store.write_batch_results(job_id, results)
    stats = store.inference_timing_stats(job_id)
    assert "min_ms" in stats
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    assert "max_ms" in stats
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["max_ms"]


def test_detection_stats_missing_job_returns_empty(store):
    stats = store.detection_stats("nonexistent")
    assert stats == {}


# ─────────────────────────────────────────────
#  ParquetStore — summary
# ─────────────────────────────────────────────

def test_read_summary_fields(store):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(3)]
    store.write_batch_results(job_id, results, "my_job")
    summary = store.read_summary(job_id)
    assert not summary.is_empty()
    assert "total_images"   in summary.columns
    assert "successful"     in summary.columns
    assert "avg_inference_ms" in summary.columns


def test_summary_correct_counts(store):
    job_id  = make_job_id()
    results = [
        make_vision_result(status="success"),
        make_vision_result(status="success"),
        make_vision_result(status="error"),
    ]
    store.write_batch_results(job_id, results)
    summary = store.read_summary(job_id)
    assert int(summary["total_images"][0])  == 3
    assert int(summary["successful"][0])    == 2
    assert int(summary["failed"][0])        == 1


# ─────────────────────────────────────────────
#  ParquetStore — merge_batches
# ─────────────────────────────────────────────

def test_merge_batches(store, tmp_path):
    job_a = make_job_id()
    job_b = make_job_id()
    store.write_batch_results(job_a, [make_vision_result() for _ in range(3)])
    store.write_batch_results(job_b, [make_vision_result() for _ in range(4)])

    out_path = store.merge_batches([job_a, job_b], "merged_test")
    df = pl.read_parquet(out_path)
    assert len(df) == 7


def test_merge_batches_empty_raises(store):
    with pytest.raises(ValueError, match="No data to merge"):
        store.merge_batches(["nonexistent-a", "nonexistent-b"], "merged")


# ─────────────────────────────────────────────
#  ParquetStore — export
# ─────────────────────────────────────────────

def test_export_csv(store, tmp_path):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(3)]
    store.write_batch_results(job_id, results)
    csv_path = store.export_csv(job_id)
    assert Path(csv_path).exists()
    assert Path(csv_path).stat().st_size > 0


def test_export_json(store, tmp_path):
    job_id  = make_job_id()
    results = [make_vision_result() for _ in range(3)]
    store.write_batch_results(job_id, results)
    json_path = store.export_json(job_id)
    assert Path(json_path).exists()
    # Each line is a valid JSON object
    lines = Path(json_path).read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "request_id" in obj


# ─────────────────────────────────────────────
#  ParquetStore — generic read/write
# ─────────────────────────────────────────────

def test_generic_write_read_roundtrip(store, tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    store.write(df, "test_table")
    df2 = store.read("test_table")
    assert df.equals(df2)


def test_file_stats(store, tmp_path):
    df   = pl.DataFrame({"col": list(range(100))})
    path = store.write(df, "stats_test")
    stats = store.file_stats(path)
    assert stats["row_count"] == 100
    assert stats["file_size_kb"] > 0
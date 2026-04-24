"""
ApexVision-Core — SQLAlchemy ORM Models
Sincronizados con las migraciones Alembic.

Tablas:
  api_keys       → gestión de clientes con rate limits
  vision_results → un registro por análisis de imagen
  batch_jobs     → jobs de procesamiento batch
  model_metrics  → latencias y uso por modelo
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float,
    Index, Integer, JSON, String, Text,
    func, text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from python.db.session import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
#  APIKey
# ─────────────────────────────────────────────

class APIKey(Base):
    __tablename__ = "api_keys"

    id:       Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True,
                                          comment="SHA-256 hash de la API key")
    name:     Mapped[str] = mapped_column(String(100), nullable=False,
                                          comment="Nombre descriptivo del cliente")

    is_active:      Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    rate_limit:     Mapped[int]  = mapped_column(Integer, nullable=False, default=100,
                                                 comment="Requests por minuto")
    total_requests: Mapped[int]  = mapped_column(BigInteger, nullable=False, default=0)
    last_used_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        Index("ix_api_keys_key_hash",  "key_hash"),
        Index("ix_api_keys_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<APIKey name={self.name!r} active={self.is_active}>"


# ─────────────────────────────────────────────
#  VisionResult
# ─────────────────────────────────────────────

class VisionResult(Base):
    __tablename__ = "vision_results"

    id:           Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    request_id:   Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status:       Mapped[str] = mapped_column(String(20), nullable=False, default="success")

    # Tasks
    tasks_ran: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Imagen
    image_width:  Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timings
    total_inference_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Detection
    detection_count: Mapped[int]         = mapped_column(Integer, nullable=False, default=0)
    detection_model: Mapped[str | None]  = mapped_column(String(100), nullable=True)
    detection_ms:    Mapped[float]       = mapped_column(Float, nullable=False, default=0.0)
    detection_json:  Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Classification
    classification_top_label: Mapped[str | None]   = mapped_column(String(200), nullable=True)
    classification_top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_model:     Mapped[str | None]   = mapped_column(String(100), nullable=True)
    classification_ms:        Mapped[float]        = mapped_column(Float, nullable=False, default=0.0)

    # OCR
    ocr_text:        Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_char_count:  Mapped[int]        = mapped_column(Integer, nullable=False, default=0)
    ocr_block_count: Mapped[int]        = mapped_column(Integer, nullable=False, default=0)
    ocr_language:    Mapped[str | None] = mapped_column(String(10), nullable=True)
    ocr_ms:          Mapped[float]      = mapped_column(Float, nullable=False, default=0.0)

    # Face
    face_count: Mapped[int]   = mapped_column(Integer, nullable=False, default=0)
    face_ms:    Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Embedding
    embedding_dim:   Mapped[int]         = mapped_column(Integer, nullable=False, default=0)
    embedding_model: Mapped[str | None]  = mapped_column(String(100), nullable=True)
    embedding_ms:    Mapped[float]       = mapped_column(Float, nullable=False, default=0.0)

    # Depth
    depth_min_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth_max_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth_ms:    Mapped[float]        = mapped_column(Float, nullable=False, default=0.0)

    # Segmentation
    seg_mask_count: Mapped[int]   = mapped_column(Integer, nullable=False, default=0)
    seg_ms:         Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Storage
    stored_at:    Mapped[str | None] = mapped_column(String(500), nullable=True)
    batch_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_vision_results_request_id",   "request_id"),
        Index("ix_vision_results_api_key_hash", "api_key_hash"),
        Index("ix_vision_results_created_at",   "created_at"),
        Index("ix_vision_results_status",       "status"),
        Index("ix_vision_results_batch_job_id", "batch_job_id"),
        Index("ix_vision_results_created_status", "created_at", "status"),
    )

    @classmethod
    def from_response(cls, response: Any, api_key_hash: str | None = None) -> "VisionResult":
        """Crea una instancia desde un VisionResponse."""
        det = response.detection
        ocr = response.ocr
        emb = response.embedding
        fac = response.face
        dep = response.depth
        seg = response.segmentation
        clf = response.classification

        return cls(
            request_id=str(response.request_id),
            api_key_hash=api_key_hash,
            status=response.status,
            tasks_ran=[str(t) for t in response.tasks_ran],
            image_width=response.image_width,
            image_height=response.image_height,
            total_inference_ms=response.total_inference_ms,

            detection_count=det.count          if det else 0,
            detection_model=det.model_used     if det else None,
            detection_ms=det.inference_ms      if det else 0.0,
            detection_json=[b.model_dump() for b in det.boxes] if det else None,

            classification_top_label=clf.predictions[0]["label"]      if clf and clf.predictions else None,
            classification_top_score=clf.predictions[0]["confidence"] if clf and clf.predictions else None,
            classification_model=clf.model_used   if clf else None,
            classification_ms=clf.inference_ms    if clf else 0.0,

            ocr_text=ocr.text                    if ocr else None,
            ocr_char_count=len(ocr.text)         if ocr else 0,
            ocr_block_count=len(ocr.blocks)      if ocr else 0,
            ocr_language=ocr.language_detected   if ocr else None,
            ocr_ms=ocr.inference_ms              if ocr else 0.0,

            face_count=fac.count                 if fac else 0,
            face_ms=fac.inference_ms             if fac else 0.0,

            embedding_dim=emb.dimensions         if emb else 0,
            embedding_model=emb.model_used       if emb else None,
            embedding_ms=emb.inference_ms        if emb else 0.0,

            depth_min_m=dep.min_depth            if dep else None,
            depth_max_m=dep.max_depth            if dep else None,
            depth_ms=dep.inference_ms            if dep else 0.0,

            seg_mask_count=seg.count             if seg else 0,
            seg_ms=seg.inference_ms              if seg else 0.0,

            stored_at=response.stored_at,
        )

    def __repr__(self) -> str:
        return f"<VisionResult request_id={self.request_id!r} status={self.status!r}>"


# ─────────────────────────────────────────────
#  BatchJob
# ─────────────────────────────────────────────

class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id:           Mapped[str]       = mapped_column(String(36), primary_key=True)
    name:         Mapped[str | None] = mapped_column(String(200), nullable=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status:       Mapped[str]       = mapped_column(String(30), nullable=False, default="pending")

    total:     Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed:    Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    progress_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    result_path:  Mapped[str | None] = mapped_column(String(500), nullable=True)
    webhook_url:  Mapped[str | None] = mapped_column(String(500), nullable=True)
    webhook_sent: Mapped[bool]       = mapped_column(Boolean, nullable=False, default=False)

    elapsed_ms:       Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_ms_per_image: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at:  Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    updated_at:  Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_batch_jobs_status",       "status"),
        Index("ix_batch_jobs_created_at",   "created_at"),
        Index("ix_batch_jobs_api_key_hash", "api_key_hash"),
    )

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "done_with_errors", "failed")

    def __repr__(self) -> str:
        return f"<BatchJob id={self.id!r} status={self.status!r} {self.completed}/{self.total}>"


# ─────────────────────────────────────────────
#  ModelMetric
# ─────────────────────────────────────────────

class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id:         Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    task:       Mapped[str] = mapped_column(String(30), nullable=False)
    device:     Mapped[str] = mapped_column(String(10), nullable=False, default="cpu")

    inference_ms:     Mapped[float]       = mapped_column(Float, nullable=False)
    total_request_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    image_width:  Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    output_count: Mapped[int]  = mapped_column(Integer, nullable=False, default=0)
    cache_hit:    Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status:       Mapped[str]  = mapped_column(String(20), nullable=False, default="success")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_model_metrics_model_name", "model_name"),
        Index("ix_model_metrics_task",       "task"),
        Index("ix_model_metrics_created_at", "created_at"),
        Index("ix_model_metrics_model_task", "model_name", "task"),
    )

    def __repr__(self) -> str:
        return f"<ModelMetric {self.model_name!r} {self.task!r} {self.inference_ms:.1f}ms>"

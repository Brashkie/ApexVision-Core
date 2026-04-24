"""initial: create vision_results and batch_jobs tables

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01 00:00:00.000000

Tablas creadas:
  - vision_results  → almacena cada resultado de análisis de imagen
  - batch_jobs      → registro de jobs de procesamiento batch
  - api_keys        → gestión de API keys con rate limits
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ─────────────────────────────────────────────
#  Metadata
# ─────────────────────────────────────────────

revision:      str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


# ─────────────────────────────────────────────
#  upgrade
# ─────────────────────────────────────────────

def upgrade() -> None:

    # ── Extensiones PostgreSQL ─────────────────────────────────────
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    # ── api_keys ───────────────────────────────────────────────────
    # Tabla de API keys con rate limiting por key
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            sa.String(36),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()::text"),
        ),
        sa.Column("key_hash",    sa.String(64),  nullable=False, unique=True,
                  comment="SHA-256 hash de la API key"),
        sa.Column("name",        sa.String(100), nullable=False,
                  comment="Nombre descriptivo del cliente"),
        sa.Column("is_active",   sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("rate_limit",  sa.Integer(),   nullable=False, server_default="100",
                  comment="Requests por minuto"),
        sa.Column("total_requests", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_used_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        comment="Gestión de API keys y rate limits",
    )

    op.create_index("ix_api_keys_key_hash",  "api_keys", ["key_hash"])
    op.create_index("ix_api_keys_is_active", "api_keys", ["is_active"])

    # ── vision_results ─────────────────────────────────────────────
    # Una fila por cada llamada a /api/v1/vision/analyze
    op.create_table(
        "vision_results",
        sa.Column(
            "id",
            sa.String(36),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()::text"),
        ),
        sa.Column("request_id",   sa.String(36),  nullable=False, unique=True,
                  comment="UUID del request, generado por el cliente o el servidor"),
        sa.Column("api_key_hash", sa.String(64),  nullable=True,
                  comment="Hash de la API key usada (referencia soft a api_keys)"),

        # Tasks
        sa.Column("tasks_ran",    sa.JSON(),       nullable=False,
                  comment='Lista de tasks ejecutadas: ["detect","ocr"]'),
        sa.Column("status",       sa.String(20),   nullable=False, server_default="success"),

        # Imagen
        sa.Column("image_width",  sa.Integer(),    nullable=False, server_default="0"),
        sa.Column("image_height", sa.Integer(),    nullable=False, server_default="0"),

        # Timings
        sa.Column("total_inference_ms", sa.Float(), nullable=False, server_default="0.0"),

        # Detection
        sa.Column("detection_count",  sa.Integer(), nullable=False, server_default="0"),
        sa.Column("detection_model",  sa.String(100), nullable=True),
        sa.Column("detection_ms",     sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("detection_json",   sa.JSON(),    nullable=True,
                  comment="Array de BoundingBox serializados"),

        # Classification
        sa.Column("classification_top_label", sa.String(200), nullable=True),
        sa.Column("classification_top_score", sa.Float(),     nullable=True),
        sa.Column("classification_model",     sa.String(100), nullable=True),
        sa.Column("classification_ms",        sa.Float(),     nullable=False, server_default="0.0"),

        # OCR
        sa.Column("ocr_text",         sa.Text(),    nullable=True),
        sa.Column("ocr_char_count",   sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ocr_block_count",  sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ocr_language",     sa.String(10), nullable=True),
        sa.Column("ocr_ms",           sa.Float(),   nullable=False, server_default="0.0"),

        # Face
        sa.Column("face_count",   sa.Integer(), nullable=False, server_default="0"),
        sa.Column("face_ms",      sa.Float(),   nullable=False, server_default="0.0"),

        # Embedding
        sa.Column("embedding_dim",   sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        sa.Column("embedding_ms",    sa.Float(),   nullable=False, server_default="0.0"),

        # Depth
        sa.Column("depth_min_m", sa.Float(), nullable=True),
        sa.Column("depth_max_m", sa.Float(), nullable=True),
        sa.Column("depth_ms",    sa.Float(), nullable=False, server_default="0.0"),

        # Segmentation
        sa.Column("seg_mask_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("seg_ms",         sa.Float(),   nullable=False, server_default="0.0"),

        # Storage
        sa.Column("stored_at",      sa.String(500), nullable=True,
                  comment="Ruta en Delta Lake si store_result=true"),
        sa.Column("batch_job_id",   sa.String(36),  nullable=True,
                  comment="FK soft a batch_jobs si vino de un batch"),

        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        comment="Un registro por cada análisis de imagen",
    )

    # Índices en vision_results
    op.create_index("ix_vision_results_request_id",   "vision_results", ["request_id"])
    op.create_index("ix_vision_results_api_key_hash", "vision_results", ["api_key_hash"])
    op.create_index("ix_vision_results_created_at",   "vision_results", ["created_at"])
    op.create_index("ix_vision_results_status",       "vision_results", ["status"])
    op.create_index("ix_vision_results_batch_job_id", "vision_results", ["batch_job_id"])
    # Índice compuesto para queries de analytics por día
    op.create_index(
        "ix_vision_results_created_status",
        "vision_results",
        ["created_at", "status"],
    )
    # Índice GIN para buscar en JSONB de tasks_ran
    # Requiere jsonb_path_ops como operator class para columnas JSON
    op.execute("""
        ALTER TABLE vision_results
        ALTER COLUMN tasks_ran TYPE JSONB USING tasks_ran::JSONB
    """)
    op.execute("""
        CREATE INDEX ix_vision_results_tasks_ran_gin
        ON vision_results
        USING gin(tasks_ran jsonb_path_ops)
    """)
    # Full-text search en OCR text
    op.execute("""
        CREATE INDEX ix_vision_results_ocr_text_fts
        ON vision_results
        USING gin(to_tsvector('english', COALESCE(ocr_text, '')))
    """)

    # ── batch_jobs ─────────────────────────────────────────────────
    # Una fila por cada job batch enviado a Celery
    op.create_table(
        "batch_jobs",
        sa.Column(
            "id",
            sa.String(36),
            primary_key=True,
            comment="Celery task ID — también es el job_id expuesto en la API",
        ),
        sa.Column("name",       sa.String(200), nullable=True,
                  comment="Nombre descriptivo del job"),
        sa.Column("api_key_hash", sa.String(64), nullable=True),
        sa.Column("status",     sa.String(30),  nullable=False, server_default="pending",
                  comment="pending | running | done | done_with_errors | failed"),

        # Contadores
        sa.Column("total",     sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed",    sa.Integer(), nullable=False, server_default="0"),

        # Progreso
        sa.Column("progress_pct", sa.Float(), nullable=False, server_default="0.0"),

        # Resultados
        sa.Column("result_path", sa.String(500), nullable=True,
                  comment="Ruta al archivo .parquet con los resultados"),
        sa.Column("webhook_url", sa.String(500), nullable=True),
        sa.Column("webhook_sent", sa.Boolean(), nullable=False, server_default="false"),

        # Timings
        sa.Column("elapsed_ms",       sa.Float(), nullable=True),
        sa.Column("avg_ms_per_image", sa.Float(), nullable=True),

        # Errores
        sa.Column("error_summary", sa.Text(), nullable=True),

        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),

        comment="Un registro por cada job de procesamiento batch",
    )

    # Índices en batch_jobs
    op.create_index("ix_batch_jobs_status",     "batch_jobs", ["status"])
    op.create_index("ix_batch_jobs_created_at", "batch_jobs", ["created_at"])
    op.create_index("ix_batch_jobs_api_key_hash", "batch_jobs", ["api_key_hash"])

    # ── Trigger: updated_at automático ────────────────────────────
    # Función que actualiza updated_at en cualquier tabla que la use
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    """)

    op.execute("""
        CREATE TRIGGER trg_batch_jobs_updated_at
        BEFORE UPDATE ON batch_jobs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column()
    """)

    op.execute("""
        CREATE TRIGGER trg_api_keys_updated_at
        BEFORE UPDATE ON api_keys
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column()
    """)

    # ── Vista de analytics ─────────────────────────────────────────
    # Resumen diario de requests — útil para dashboards
    op.execute("""
        CREATE OR REPLACE VIEW daily_vision_stats AS
        SELECT
            DATE(created_at AT TIME ZONE 'UTC') AS day,
            COUNT(*)                            AS total_requests,
            COUNT(*) FILTER (WHERE status = 'success') AS successful,
            COUNT(*) FILTER (WHERE status != 'success') AS failed,
            AVG(total_inference_ms)             AS avg_inference_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_inference_ms) AS p95_ms,
            SUM(detection_count)                AS total_detections,
            SUM(face_count)                     AS total_faces,
            COUNT(*) FILTER (WHERE ocr_char_count > 0) AS requests_with_text
        FROM vision_results
        GROUP BY DATE(created_at AT TIME ZONE 'UTC')
        ORDER BY day DESC
    """)


# ─────────────────────────────────────────────
#  downgrade
# ─────────────────────────────────────────────

def downgrade() -> None:
    # Orden inverso al upgrade

    op.execute("DROP VIEW IF EXISTS daily_vision_stats")

    op.execute("DROP TRIGGER IF EXISTS trg_api_keys_updated_at ON api_keys")
    op.execute("DROP TRIGGER IF EXISTS trg_batch_jobs_updated_at ON batch_jobs")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    op.execute("DROP INDEX IF EXISTS ix_vision_results_ocr_text_fts")

    op.execute("DROP INDEX IF EXISTS ix_vision_results_tasks_ran_gin")
    op.drop_index("ix_vision_results_created_status", table_name="vision_results")
    op.drop_index("ix_vision_results_batch_job_id",   table_name="vision_results")
    op.drop_index("ix_vision_results_status",         table_name="vision_results")
    op.drop_index("ix_vision_results_created_at",     table_name="vision_results")
    op.drop_index("ix_vision_results_api_key_hash",   table_name="vision_results")
    op.drop_index("ix_vision_results_request_id",     table_name="vision_results")
    op.drop_table("vision_results")

    op.drop_index("ix_batch_jobs_api_key_hash", table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_created_at",   table_name="batch_jobs")
    op.drop_index("ix_batch_jobs_status",       table_name="batch_jobs")
    op.drop_table("batch_jobs")

    op.drop_index("ix_api_keys_is_active", table_name="api_keys")
    op.drop_index("ix_api_keys_key_hash",  table_name="api_keys")
    op.drop_table("api_keys")
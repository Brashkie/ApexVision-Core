"""add model_metrics table and performance indexes

Revision ID: 002_model_metrics
Revises:     001_initial
Create Date: 2025-01-01 01:00:00.000000

Cambios:
  - Nueva tabla model_metrics: latencias y contadores por modelo
  - Índice partial en vision_results para búsquedas por detección
  - Índice en ocr_language para filtros por idioma
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision:      str = "002_model_metrics"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── model_metrics ──────────────────────────────────────────────
    # Estadísticas de latencia y uso por modelo — útil para monitoreo
    op.create_table(
        "model_metrics",
        sa.Column(
            "id",
            sa.String(36),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()::text"),
        ),
        sa.Column("model_name",   sa.String(100), nullable=False),
        sa.Column("task",         sa.String(30),  nullable=False),
        sa.Column("device",       sa.String(10),  nullable=False, server_default="cpu"),

        # Latencias
        sa.Column("inference_ms",     sa.Float(), nullable=False),
        sa.Column("total_request_ms", sa.Float(), nullable=True),

        # Imagen
        sa.Column("image_width",  sa.Integer(), nullable=True),
        sa.Column("image_height", sa.Integer(), nullable=True),

        # Resultado
        sa.Column("output_count",  sa.Integer(), nullable=False, server_default="0",
                  comment="Número de detecciones/clases/máscaras devueltas"),
        sa.Column("cache_hit",     sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("status",        sa.String(20), nullable=False, server_default="success"),

        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        comment="Métricas de latencia por modelo y task — para monitoreo de performance",
    )

    op.create_index("ix_model_metrics_model_name", "model_metrics", ["model_name"])
    op.create_index("ix_model_metrics_task",       "model_metrics", ["task"])
    op.create_index("ix_model_metrics_created_at", "model_metrics", ["created_at"])
    op.create_index(
        "ix_model_metrics_model_task",
        "model_metrics",
        ["model_name", "task"],
    )

    # ── Índices adicionales en vision_results ──────────────────────

    # Partial index: sólo filas con detecciones — acelera queries de analytics
    op.execute("""
        CREATE INDEX ix_vision_results_has_detections
        ON vision_results (detection_count, created_at)
        WHERE detection_count > 0
    """)

    # Índice en ocr_language para filtrar por idioma
    op.execute("""
        CREATE INDEX ix_vision_results_ocr_language
        ON vision_results (ocr_language)
        WHERE ocr_language IS NOT NULL
    """)

    # ── Vista de métricas por modelo ───────────────────────────────
    op.execute("""
        CREATE OR REPLACE VIEW model_performance_summary AS
        SELECT
            model_name,
            task,
            device,
            COUNT(*)                                                    AS total_calls,
            AVG(inference_ms)                                           AS avg_ms,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY inference_ms) AS p50_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_ms) AS p95_ms,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_ms) AS p99_ms,
            MIN(inference_ms)                                           AS min_ms,
            MAX(inference_ms)                                           AS max_ms,
            COUNT(*) FILTER (WHERE cache_hit = true)                    AS cache_hits,
            ROUND(
                COUNT(*) FILTER (WHERE cache_hit = true)::numeric /
                NULLIF(COUNT(*), 0) * 100, 2
            )                                                           AS cache_hit_pct,
            COUNT(*) FILTER (WHERE status = 'success')                  AS successful,
            COUNT(*) FILTER (WHERE status != 'success')                 AS errors
        FROM model_metrics
        GROUP BY model_name, task, device
        ORDER BY total_calls DESC
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS model_performance_summary")
    op.execute("DROP INDEX IF EXISTS ix_vision_results_ocr_language")
    op.execute("DROP INDEX IF EXISTS ix_vision_results_has_detections")
    op.drop_index("ix_model_metrics_model_task", table_name="model_metrics")
    op.drop_index("ix_model_metrics_created_at", table_name="model_metrics")
    op.drop_index("ix_model_metrics_task",       table_name="model_metrics")
    op.drop_index("ix_model_metrics_model_name", table_name="model_metrics")
    op.drop_table("model_metrics")

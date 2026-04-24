"""
ApexVision-Core — Maintenance Tasks
Tareas periódicas ejecutadas por Celery Beat.

  compact_delta_task    → compacta small files en Delta Lake
  vacuum_delta_task     → elimina Parquet files obsoletos
  cleanup_old_results   → elimina registros viejos de PostgreSQL
  health_check_task     → verifica Redis + DB cada 5 min
  metrics_summary_task  → agrega métricas horarias a model_metrics
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

from loguru import logger

from python.celery_app import celery_app


# ─────────────────────────────────────────────
#  Delta Lake maintenance
# ─────────────────────────────────────────────

@celery_app.task(
    queue="default",
    name="tasks.compact_delta",
    max_retries=2,
    default_retry_delay=300,   # 5 min entre reintentos
)
def compact_delta_task(table: str = "vision_results") -> dict:
    """
    Compacta los small Parquet files de una tabla Delta Lake en archivos más grandes.
    Mejora el rendimiento de queries al reducir el número de archivos a leer.
    Target: archivos de ~128 MB.

    Se ejecuta a las 02:00 UTC y 02:10 UTC para las dos tablas.
    """
    logger.info(f"[Beat] compact_delta START | table={table}")
    try:
        from python.storage.delta_store import DeltaStore
        store  = DeltaStore()
        result = store.compact(table, target_size_mb=128)
        logger.info(f"[Beat] compact_delta DONE  | table={table} | result={result}")
        return {"status": "ok", "table": table, "result": result}
    except Exception as e:
        logger.error(f"[Beat] compact_delta FAILED | table={table} | error={e}")
        return {"status": "error", "table": table, "error": str(e)}


@celery_app.task(
    queue="default",
    name="tasks.vacuum_delta",
    max_retries=2,
    default_retry_delay=300,
)
def vacuum_delta_task(
    table: str = "vision_results",
    retention_hours: int = 168,   # 7 días
) -> dict:
    """
    Elimina los Parquet files que ya no son parte de ninguna versión activa
    de la tabla Delta Lake (archivos huérfanos después de overwrites/deletes).

    Retención mínima: 168h (7 días) — requerimiento de Delta Lake.
    Se ejecuta a las 02:30 UTC y 02:40 UTC para las dos tablas.
    """
    logger.info(f"[Beat] vacuum_delta START | table={table} | retention={retention_hours}h")
    try:
        from python.storage.delta_store import DeltaStore
        store   = DeltaStore()
        deleted = store.vacuum(table, retention_hours=retention_hours)
        logger.info(f"[Beat] vacuum_delta DONE  | table={table} | deleted={len(deleted)} files")
        return {"status": "ok", "table": table, "deleted_files": len(deleted)}
    except Exception as e:
        logger.error(f"[Beat] vacuum_delta FAILED | table={table} | error={e}")
        return {"status": "error", "table": table, "error": str(e)}


# ─────────────────────────────────────────────
#  PostgreSQL cleanup
# ─────────────────────────────────────────────

@celery_app.task(
    queue="default",
    name="tasks.cleanup_old_results",
    max_retries=1,
)
def cleanup_old_results(retention_days: int = 90) -> dict:
    """
    Elimina registros de vision_results de más de `retention_days` días.
    También limpia batch_jobs finalizados de más de 30 días.
    Se ejecuta a las 03:00 UTC.
    """
    logger.info(f"[Beat] cleanup_old_results START | retention={retention_days}d")

    async def _run():
        from python.db.session import AsyncSessionLocal
        import sqlalchemy as sa

        cutoff_results = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_jobs    = datetime.now(timezone.utc) - timedelta(days=30)

        async with AsyncSessionLocal() as session:
            # Eliminar vision_results viejos
            r1 = await session.execute(
                sa.text(
                    "DELETE FROM vision_results WHERE created_at < :cutoff"
                ).bindparams(cutoff=cutoff_results)
            )
            deleted_results = r1.rowcount

            # Eliminar batch_jobs finalizados viejos
            r2 = await session.execute(
                sa.text("""
                    DELETE FROM batch_jobs
                    WHERE created_at < :cutoff
                      AND status IN ('done', 'done_with_errors', 'failed')
                """).bindparams(cutoff=cutoff_jobs)
            )
            deleted_jobs = r2.rowcount

            await session.commit()

        return deleted_results, deleted_jobs

    try:
        deleted_results, deleted_jobs = asyncio.run(_run())
        logger.info(
            f"[Beat] cleanup_old_results DONE | "
            f"vision_results={deleted_results} | batch_jobs={deleted_jobs}"
        )
        return {
            "status": "ok",
            "deleted_vision_results": deleted_results,
            "deleted_batch_jobs":     deleted_jobs,
            "cutoff_days":            retention_days,
        }
    except Exception as e:
        logger.error(f"[Beat] cleanup_old_results FAILED | error={e}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────

@celery_app.task(
    queue="default",
    name="tasks.health_check",
    max_retries=0,             # no reintentar health checks
    ignore_result=True,        # no guardar resultado en Redis
)
def health_check_task() -> dict:
    """
    Verifica que Redis y PostgreSQL están accesibles.
    Se ejecuta cada 5 minutos.
    Si algo falla, lo loguea — el monitoreo externo (Prometheus/Flower)
    puede detectar la ausencia de resultados.
    """
    checks: dict[str, str] = {}

    # Redis
    try:
        import redis
        r = redis.from_url(
            __import__("python.config", fromlist=["settings"]).settings.REDIS_URL,
            socket_connect_timeout=3,
        )
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        logger.warning(f"[Beat] health_check | Redis FAILED: {e}")

    # PostgreSQL
    async def _check_db():
        from python.db.session import AsyncSessionLocal
        import sqlalchemy as sa
        async with AsyncSessionLocal() as s:
            await s.execute(sa.text("SELECT 1"))

    try:
        asyncio.run(_check_db())
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        logger.warning(f"[Beat] health_check | PostgreSQL FAILED: {e}")

    all_ok = all(v == "ok" for v in checks.values())
    if all_ok:
        logger.debug(f"[Beat] health_check OK | {checks}")
    else:
        logger.error(f"[Beat] health_check DEGRADED | {checks}")

    return {"status": "ok" if all_ok else "degraded", "checks": checks}


# ─────────────────────────────────────────────
#  Métricas horarias
# ─────────────────────────────────────────────

@celery_app.task(
    queue="default",
    name="tasks.metrics_summary",
    max_retries=1,
)
def metrics_summary_task() -> dict:
    """
    Cada hora agrega las métricas de la última hora desde vision_results
    a la tabla model_metrics.
    Permite ver tendencias de latencia por modelo sin escanear toda la tabla.
    """
    logger.info("[Beat] metrics_summary START")

    async def _run():
        from python.db.session import AsyncSessionLocal
        import sqlalchemy as sa

        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

        # Agrega latencias por modelo y task de la última hora
        query = sa.text("""
            INSERT INTO model_metrics (
                id, model_name, task, device,
                inference_ms, output_count, cache_hit, status, created_at
            )
            SELECT
                gen_random_uuid()::text,
                COALESCE(detection_model, 'unknown')   AS model_name,
                'detect'                               AS task,
                'cpu'                                  AS device,
                AVG(detection_ms)                      AS inference_ms,
                AVG(detection_count)::int              AS output_count,
                false                                  AS cache_hit,
                'success'                              AS status,
                NOW()                                  AS created_at
            FROM vision_results
            WHERE created_at >= :since
              AND detection_model IS NOT NULL
              AND detection_count > 0
            GROUP BY detection_model
            HAVING COUNT(*) > 0

            UNION ALL

            SELECT
                gen_random_uuid()::text,
                COALESCE(classification_model, 'unknown'),
                'classify', 'cpu',
                AVG(classification_ms), 0, false, 'success', NOW()
            FROM vision_results
            WHERE created_at >= :since
              AND classification_model IS NOT NULL
            GROUP BY classification_model
            HAVING COUNT(*) > 0
        """).bindparams(since=one_hour_ago)

        async with AsyncSessionLocal() as session:
            result = await session.execute(query)
            inserted = result.rowcount
            await session.commit()
            return inserted

    try:
        inserted = asyncio.run(_run())
        logger.info(f"[Beat] metrics_summary DONE | inserted={inserted} rows")
        return {"status": "ok", "inserted": inserted}
    except Exception as e:
        logger.error(f"[Beat] metrics_summary FAILED | error={e}")
        return {"status": "error", "error": str(e)}

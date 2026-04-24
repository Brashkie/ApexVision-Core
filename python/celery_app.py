"""
ApexVision-Core — Celery Application
Incluye configuración completa + Celery Beat schedule para tareas de mantenimiento.

Queues:
  vision   → análisis individual de imágenes
  batch    → procesamiento de lotes
  default  → tareas de mantenimiento y otras

Beat schedule (tareas automáticas):
  02:00 UTC → compact_delta_task    (compacta small files en Delta Lake)
  02:30 UTC → vacuum_delta_task     (elimina archivos obsoletos)
  03:00 UTC → cleanup_old_results   (limpia vision_results > 90 días)
  */5 min   → health_check_task     (verifica Redis y DB)
  */1 hora  → metrics_summary_task  (agrega métricas a model_metrics)
"""

from celery import Celery
from celery.schedules import crontab

from python.config import settings

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────

celery_app = Celery(
    "apexvision",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "python.tasks.vision_tasks",
        "python.tasks.batch_tasks",
        "python.tasks.maintenance_tasks",
    ],
)

# ─────────────────────────────────────────────
#  Configuración base
# ─────────────────────────────────────────────

celery_app.conf.update(

    # Serialización
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Comportamiento de tasks
    task_track_started=True,
    task_acks_late=True,              # ack solo después de completar (safe ante crashes)
    worker_prefetch_multiplier=1,     # un task a la vez por worker (ML es pesado)
    task_reject_on_worker_lost=True,  # re-encola si el worker muere

    # Timeouts
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT - 30,

    # Resultado TTL — mantiene resultados 24h en Redis
    result_expires=settings.REDIS_JOB_TTL,

    # Routing por queue
    task_routes={
        "tasks.analyze_image":   {"queue": "vision"},
        "tasks.process_batch":   {"queue": "batch"},
        "tasks.compact_delta":   {"queue": "default"},
        "tasks.vacuum_delta":    {"queue": "default"},
        "tasks.cleanup_old_results": {"queue": "default"},
        "tasks.health_check":    {"queue": "default"},
        "tasks.metrics_summary": {"queue": "default"},
    },

    # Reintentos por defecto
    task_max_retries=settings.CELERY_MAX_RETRIES,

    # ─────────────────────────────────────────
    #  Beat schedule
    #  Todas las horas en UTC
    # ─────────────────────────────────────────
    beat_schedule={

        # ── Delta Lake maintenance ─────────────────────────────────

        # 02:00 UTC — Compacta small Parquet files en un solo archivo grande
        # Mejora el rendimiento de queries de analytics
        "compact-delta-vision-results": {
            "task":     "tasks.compact_delta",
            "schedule": crontab(hour=2, minute=0),
            "args":     ("vision_results",),
            "options":  {"queue": "default", "expires": 3600},
        },

        # 02:10 UTC — Compacta tabla batch_jobs
        "compact-delta-batch-jobs": {
            "task":     "tasks.compact_delta",
            "schedule": crontab(hour=2, minute=10),
            "args":     ("batch_jobs",),
            "options":  {"queue": "default", "expires": 3600},
        },

        # 02:30 UTC — Vacuum: elimina Parquet files obsoletos (retención 7 días)
        # Solo archivos que ya no son parte de ninguna versión activa
        "vacuum-delta-vision-results": {
            "task":     "tasks.vacuum_delta",
            "schedule": crontab(hour=2, minute=30),
            "args":     ("vision_results",),
            "kwargs":   {"retention_hours": 168},   # 7 días
            "options":  {"queue": "default", "expires": 3600},
        },

        # 02:40 UTC — Vacuum tabla batch_jobs
        "vacuum-delta-batch-jobs": {
            "task":     "tasks.vacuum_delta",
            "schedule": crontab(hour=2, minute=40),
            "args":     ("batch_jobs",),
            "kwargs":   {"retention_hours": 168},
            "options":  {"queue": "default", "expires": 3600},
        },

        # ── Base de datos ──────────────────────────────────────────

        # 03:00 UTC — Elimina vision_results de más de 90 días
        # Evita crecimiento ilimitado de la tabla
        "cleanup-old-vision-results": {
            "task":     "tasks.cleanup_old_results",
            "schedule": crontab(hour=3, minute=0),
            "kwargs":   {"retention_days": 90},
            "options":  {"queue": "default", "expires": 7200},
        },

        # ── Monitoreo ──────────────────────────────────────────────

        # Cada 5 minutos — verifica que Redis y DB están accesibles
        "health-check": {
            "task":     "tasks.health_check",
            "schedule": crontab(minute="*/5"),
            "options":  {"queue": "default", "expires": 60},
        },

        # Cada hora — agrega métricas de latencia por modelo a model_metrics
        "metrics-hourly-summary": {
            "task":     "tasks.metrics_summary",
            "schedule": crontab(minute=0),     # inicio de cada hora
            "options":  {"queue": "default", "expires": 3600},
        },
    },

    # Persistence del schedule en Redis (para no perder el estado al reiniciar Beat)
    beat_scheduler="celery.beat:PersistentScheduler",
    beat_schedule_filename="celerybeat-schedule",   # archivo local de estado
    beat_max_loop_interval=5,                       # chequea cada 5s si hay algo que correr
)
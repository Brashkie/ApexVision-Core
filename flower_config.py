"""
ApexVision-Core — Flower Dashboard Configuration
Monitoreo en tiempo real de tareas Celery.

Uso:
    # Dev (simple)
    celery -A python.celery_app flower --port=5555

    # Con config completa
    celery -A python.celery_app flower --conf=flower_config.py

    # Con autenticación
    celery -A python.celery_app flower --port=5555 --basic_auth=admin:apexvision

    # Acceder en: http://localhost:5555
"""

import os

# ─────────────────────────────────────────────
#  Servidor
# ─────────────────────────────────────────────

port       = int(os.getenv("FLOWER_PORT", "5555"))
address    = "0.0.0.0"
url_prefix = ""

# ─────────────────────────────────────────────
#  Autenticación HTTP Basic
# ─────────────────────────────────────────────

basic_auth = [
    f"{os.getenv('FLOWER_USER', 'admin')}:{os.getenv('FLOWER_PASSWORD', 'apexvision')}"
]

# ─────────────────────────────────────────────
#  Broker
# ─────────────────────────────────────────────

broker_api = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────

# Tiempo máximo de tareas visibles en la UI
max_tasks    = 10_000
tasks_columns = (
    "name,uuid,state,args,result,"
    "received,started,runtime,worker"
)

# Refresh automático en la UI (segundos)
auto_refresh = True
refresh_rate = 2_000   # ms

# ─────────────────────────────────────────────
#  Persistencia
# ─────────────────────────────────────────────

# Guarda estado de tareas en DB para sobrevivir reinicios
db         = "flower.db"
persistent = True

# ─────────────────────────────────────────────
#  Alertas
# ─────────────────────────────────────────────

# Umbral de tareas fallidas para alertar
# (requiere configurar SMTP o webhook)
# max_retries = 3

# ─────────────────────────────────────────────
#  Seguridad
# ─────────────────────────────────────────────

# CORS — permite embeber en dashboards internos
# xheaders = True

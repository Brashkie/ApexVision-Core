"""
ApexVision-Core — pytest conftest.py
Carga .env antes de cualquier import para que pydantic-settings funcione.
Ubicado en la raíz del proyecto (mismo nivel que pyproject.toml).
"""

import os
from pathlib import Path

# ── Cargar .env para tests locales ──────────────────────────────────────────
# pytest-env ya inyecta las vars definidas en pyproject.toml [tool.pytest.ini_options].env
# Este bloque es el fallback para cuando pytest-env no está instalado.

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ── Fallbacks mínimos para CI/CD sin .env ────────────────────────────────────
_DEFAULTS = {
    "SECRET_KEY":             "test-secret-key-minimum-32-chars-long!!",
    "MASTER_API_KEY":         "test-master-key",
    "DATABASE_URL":           "postgresql+asyncpg://apex:apex@localhost:5432/apexvision",
    "REDIS_URL":              "redis://localhost:6379/0",
    "CELERY_BROKER_URL":      "redis://localhost:6379/1",
    "CELERY_RESULT_BACKEND":  "redis://localhost:6379/2",
    "DEVICE":                 "cpu",
    "YOLO_MODEL":             "yolo11n.pt",
    "DELTA_LAKE_PATH":        "./data/delta",
    "PARQUET_PATH":           "./data/parquet",
    "MODELS_PATH":            "./models",
    "LOG_LEVEL":              "WARNING",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

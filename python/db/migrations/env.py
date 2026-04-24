"""
ApexVision-Core — Alembic env.py
Soporte async (asyncpg) + autogenerate desde modelos SQLAlchemy.
"""

from __future__ import annotations

import asyncio
import sys
import os
from logging.config import fileConfig

# Agrega la raíz del proyecto al path para que 'python.*' sea importable
# Funciona tanto en Windows como en Linux
# env.py está en: ApexVision-Core/python/db/migrations/env.py
# Subimos 3 niveles:  migrations/ -> db/ -> python/ -> ApexVision-Core/
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ─────────────────────────────────────────────
#  Carga settings y modelos
# ─────────────────────────────────────────────

# Importar settings primero (lee .env)
from python.config import settings

# Importar Base y todos los modelos para que Alembic los detecte
from python.db.session import Base
import python.db.models  # noqa: F401 — registra los modelos en Base.metadata

# ─────────────────────────────────────────────
#  Config de Alembic
# ─────────────────────────────────────────────

config = context.config

# Configurar logging desde alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Sobreescribir URL desde settings (única fuente de verdad: .env)
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Metadata para autogenerate
target_metadata = Base.metadata


# ─────────────────────────────────────────────
#  Modo offline (sin conexión real a DB)
#  Genera SQL puro — útil para revisar antes de aplicar
# ─────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Genera SQL sin conectarse a la DB.
    Uso: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ─────────────────────────────────────────────
#  Modo online (conexión real, async)
# ─────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Incluye tablas del schema 'public' por defecto
        include_schemas=False,
        # Evita comparar índices automáticos de FK
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Crea un engine async y corre las migraciones."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,      # no reutilizar conexiones en migrations
        echo=False,
    )

    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
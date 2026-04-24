"""${message}

Revision ID: ${up_revision}
Revises:     ${down_revision | comma,n}
Create Date: ${create_date}
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# ─────────────────────────────────────────────
#  Metadata de la revisión
# ─────────────────────────────────────────────

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


# ─────────────────────────────────────────────
#  upgrade — aplica los cambios
# ─────────────────────────────────────────────

def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


# ─────────────────────────────────────────────
#  downgrade — revierte los cambios
# ─────────────────────────────────────────────

def downgrade() -> None:
    ${downgrades if downgrades else "pass"}

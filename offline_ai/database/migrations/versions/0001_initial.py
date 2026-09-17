"""Baseline schema revision. Runtime also applies create_all and FTS setup."""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Schema is created via Base.metadata.create_all in Database.init.
    # This revision records baseline for Alembic version tracking.
    bind = op.get_bind()
    from offline_ai.database.models import Base

    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    from offline_ai.database.models import Base

    Base.metadata.drop_all(bind=bind)

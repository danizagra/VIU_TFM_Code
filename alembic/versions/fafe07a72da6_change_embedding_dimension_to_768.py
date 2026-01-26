"""change_embedding_dimension_to_768

Revision ID: fafe07a72da6
Revises: e8ec9257d77e
Create Date: 2026-01-20 21:01:37.394200

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'fafe07a72da6'
down_revision: Union[str, Sequence[str], None] = 'e8ec9257d77e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Change embedding dimension from 384 to 768.

    This migration:
    1. Drops the old embedding columns (384 dim)
    2. Creates new embedding columns (768 dim)

    Note: Existing embeddings will be lost and need to be regenerated.
    """
    # Drop old embedding columns and recreate with new dimension
    # Articles table
    op.drop_column('articles', 'embedding')
    op.add_column(
        'articles',
        sa.Column('embedding', Vector(768), nullable=True)
    )

    # Clusters table (centroid)
    op.drop_column('clusters', 'centroid')
    op.add_column(
        'clusters',
        sa.Column('centroid', Vector(768), nullable=True)
    )


def downgrade() -> None:
    """Revert to 384 dimensions."""
    # Articles table
    op.drop_column('articles', 'embedding')
    op.add_column(
        'articles',
        sa.Column('embedding', Vector(384), nullable=True)
    )

    # Clusters table
    op.drop_column('clusters', 'centroid')
    op.add_column(
        'clusters',
        sa.Column('centroid', Vector(384), nullable=True)
    )

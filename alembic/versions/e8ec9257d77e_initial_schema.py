"""initial_schema

Revision ID: e8ec9257d77e
Revises:
Create Date: 2026-01-19 21:21:46.688396

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'e8ec9257d77e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Embedding dimension (must match settings.embedding_dimension)
EMBEDDING_DIM = 384


def upgrade() -> None:
    """Create initial schema."""
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Articles table
    op.create_table(
        'articles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('external_id', sa.String(255), nullable=True),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_name', sa.String(255), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('author', sa.String(255), nullable=True),
        sa.Column('image_url', sa.Text(), nullable=True),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('language', sa.String(10), nullable=True),
        sa.Column('country', sa.String(10), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('embedding', Vector(EMBEDDING_DIM), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_url')
    )

    # Agent sessions table
    op.create_table(
        'agent_sessions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('filters', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('articles_fetched', sa.Integer(), default=0),
        sa.Column('articles_after_filter', sa.Integer(), default=0),
        sa.Column('articles_after_dedup', sa.Integer(), default=0),
        sa.Column('clusters_found', sa.Integer(), default=0),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(50), default='running'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Clusters table
    op.create_table(
        'clusters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('cluster_label', sa.Integer(), nullable=False),
        sa.Column('topic', sa.Text(), nullable=True),
        sa.Column('centroid', Vector(EMBEDDING_DIM), nullable=True),
        sa.Column('article_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['session_id'], ['agent_sessions.id'])
    )

    # Processed articles table
    op.create_table(
        'processed_articles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('article_id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('headlines', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('angles', sa.Text(), nullable=True),
        sa.Column('is_duplicate', sa.Boolean(), default=False),
        sa.Column('duplicate_of', sa.UUID(), nullable=True),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('quality_passed', sa.Boolean(), nullable=True),
        sa.Column('llm_provider', sa.String(50), nullable=True),
        sa.Column('generation_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['article_id'], ['articles.id']),
        sa.ForeignKeyConstraint(['session_id'], ['agent_sessions.id']),
        sa.ForeignKeyConstraint(['cluster_id'], ['clusters.id']),
        sa.ForeignKeyConstraint(['duplicate_of'], ['articles.id'])
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('processed_articles')
    op.drop_table('clusters')
    op.drop_table('agent_sessions')
    op.drop_table('articles')
    op.execute('DROP EXTENSION IF EXISTS vector')

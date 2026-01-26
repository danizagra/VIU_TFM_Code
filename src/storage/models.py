"""
SQLAlchemy models for the journalist agent database.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List


def utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from src.config.settings import settings


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Article(Base):
    """
    Original articles fetched from news sources.
    """
    __tablename__ = "articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    external_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Content
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Source info
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True, unique=True)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Embedding (title + description combined)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True
    )

    # Timestamps
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )

    # Relationships
    processed_articles: Mapped[List["ProcessedArticle"]] = relationship(
        back_populates="article",
        foreign_keys="ProcessedArticle.article_id"
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:50]}...')>"


class AgentSession(Base):
    """
    Agent execution sessions for tracking and auditing.
    """
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Query info
    query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    filters: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Metrics
    articles_fetched: Mapped[int] = mapped_column(Integer, default=0)
    articles_after_filter: Mapped[int] = mapped_column(Integer, default=0)
    articles_after_dedup: Mapped[int] = mapped_column(Integer, default=0)
    clusters_found: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(String(50), default="running")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    clusters: Mapped[List["Cluster"]] = relationship(back_populates="session")
    processed_articles: Mapped[List["ProcessedArticle"]] = relationship(
        back_populates="session"
    )

    def __repr__(self) -> str:
        return f"<AgentSession(id={self.id}, status='{self.status}')>"


class Cluster(Base):
    """
    News clusters identified by HDBSCAN.
    """
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id"),
        nullable=False
    )

    # Cluster info
    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    centroid: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True
    )
    article_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )

    # Relationships
    session: Mapped["AgentSession"] = relationship(back_populates="clusters")
    processed_articles: Mapped[List["ProcessedArticle"]] = relationship(
        back_populates="cluster"
    )

    def __repr__(self) -> str:
        return f"<Cluster(id={self.id}, label={self.cluster_label}, articles={self.article_count})>"


class ProcessedArticle(Base):
    """
    Articles with generated content (summaries, headlines, angles).
    """
    __tablename__ = "processed_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id"),
        nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_sessions.id"),
        nullable=False
    )
    cluster_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("clusters.id"),
        nullable=True
    )

    # Generated content
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    headlines: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    angles: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Deduplication
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_of: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id"),
        nullable=True
    )
    similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Generation metadata
    llm_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )

    # Relationships
    article: Mapped["Article"] = relationship(
        back_populates="processed_articles",
        foreign_keys=[article_id]
    )
    session: Mapped["AgentSession"] = relationship(back_populates="processed_articles")
    cluster: Mapped[Optional["Cluster"]] = relationship(back_populates="processed_articles")

    def __repr__(self) -> str:
        return f"<ProcessedArticle(id={self.id}, has_summary={self.summary is not None})>"

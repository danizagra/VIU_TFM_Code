"""
Repository for AgentSession CRUD operations.

Handles database operations for agent execution sessions.
"""

from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.storage.models import AgentSession, Cluster


class SessionRepository:
    """
    Repository for AgentSession database operations.
    """

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db

    def get_by_id(self, session_id: UUID) -> AgentSession | None:
        """Get session by ID."""
        return self.db.get(AgentSession, session_id)

    def create(
        self,
        query: str = "",
        filters: dict | None = None,
    ) -> AgentSession:
        """
        Create a new agent session.

        Args:
            query: Search query used.
            filters: Filter configuration used.

        Returns:
            Created AgentSession model.
        """
        session = AgentSession(
            query=query or None,
            filters=filters,
            status="running",
        )
        self.db.add(session)
        self.db.flush()
        return session

    def update_metrics(
        self,
        session_id: UUID,
        articles_fetched: int | None = None,
        articles_after_filter: int | None = None,
        articles_after_dedup: int | None = None,
        clusters_found: int | None = None,
    ) -> None:
        """Update session metrics."""
        session = self.get_by_id(session_id)
        if not session:
            return

        if articles_fetched is not None:
            session.articles_fetched = articles_fetched
        if articles_after_filter is not None:
            session.articles_after_filter = articles_after_filter
        if articles_after_dedup is not None:
            session.articles_after_dedup = articles_after_dedup
        if clusters_found is not None:
            session.clusters_found = clusters_found

    def complete(self, session_id: UUID, error: str | None = None) -> None:
        """
        Mark session as completed.

        Args:
            session_id: Session ID.
            error: Optional error message if failed.
        """
        session = self.get_by_id(session_id)
        if not session:
            return

        session.completed_at = datetime.now(timezone.utc)
        session.status = "failed" if error else "completed"
        session.error_message = error

    def get_recent(self, limit: int = 10) -> Sequence[AgentSession]:
        """Get recent sessions."""
        stmt = (
            select(AgentSession)
            .order_by(AgentSession.started_at.desc())
            .limit(limit)
        )
        return self.db.execute(stmt).scalars().all()

    def get_successful(self, limit: int = 10) -> Sequence[AgentSession]:
        """Get recent successful sessions."""
        stmt = (
            select(AgentSession)
            .where(AgentSession.status == "completed")
            .order_by(AgentSession.started_at.desc())
            .limit(limit)
        )
        return self.db.execute(stmt).scalars().all()

    def count(self) -> int:
        """Count total sessions."""
        stmt = select(func.count()).select_from(AgentSession)
        return self.db.execute(stmt).scalar() or 0


class ClusterRepository:
    """
    Repository for Cluster database operations.
    """

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db

    def get_by_id(self, cluster_id: int) -> Cluster | None:
        """Get cluster by ID."""
        return self.db.get(Cluster, cluster_id)

    def get_by_session(self, session_id: UUID) -> Sequence[Cluster]:
        """Get all clusters from a session."""
        stmt = (
            select(Cluster)
            .where(Cluster.session_id == session_id)
            .order_by(Cluster.cluster_label)
        )
        return self.db.execute(stmt).scalars().all()

    def create(
        self,
        session_id: UUID,
        cluster_label: int,
        topic: str = "",
        centroid: list[float] | None = None,
        article_count: int = 0,
    ) -> Cluster:
        """
        Create a new cluster.

        Args:
            session_id: Agent session ID.
            cluster_label: HDBSCAN cluster label.
            topic: Optional topic description.
            centroid: Optional centroid vector.
            article_count: Number of articles in cluster.

        Returns:
            Created Cluster model.
        """
        cluster = Cluster(
            session_id=session_id,
            cluster_label=cluster_label,
            topic=topic or None,
            centroid=centroid,
            article_count=article_count,
        )
        self.db.add(cluster)
        self.db.flush()
        return cluster

    def count_by_session(self, session_id: UUID) -> int:
        """Count clusters in a session."""
        stmt = (
            select(func.count())
            .select_from(Cluster)
            .where(Cluster.session_id == session_id)
        )
        return self.db.execute(stmt).scalar() or 0

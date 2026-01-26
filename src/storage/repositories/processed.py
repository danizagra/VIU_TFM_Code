"""
Repository for ProcessedArticle CRUD operations.

Handles database operations for processed articles
with generated content (summaries, headlines, angles).
"""

from typing import Sequence
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.storage.models import ProcessedArticle, Article


class ProcessedArticleRepository:
    """
    Repository for ProcessedArticle database operations.
    """

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db

    def get_by_id(self, processed_id: UUID) -> ProcessedArticle | None:
        """Get processed article by ID."""
        return self.db.get(ProcessedArticle, processed_id)

    def get_by_article_id(self, article_id: UUID) -> Sequence[ProcessedArticle]:
        """Get all processed versions of an article."""
        stmt = select(ProcessedArticle).where(
            ProcessedArticle.article_id == article_id
        )
        return self.db.execute(stmt).scalars().all()

    def get_by_session_id(self, session_id: UUID) -> Sequence[ProcessedArticle]:
        """Get all processed articles from a session."""
        stmt = (
            select(ProcessedArticle)
            .where(ProcessedArticle.session_id == session_id)
            .order_by(ProcessedArticle.created_at)
        )
        return self.db.execute(stmt).scalars().all()

    def create(
        self,
        article_id: UUID,
        session_id: UUID,
        summary: str = "",
        headlines: dict | None = None,
        angles: list | None = None,
        cluster_id: int | None = None,
        quality_score: float | None = None,
        quality_passed: bool | None = None,
        llm_provider: str | None = None,
        generation_time_ms: int | None = None,
    ) -> ProcessedArticle:
        """
        Create a new processed article.

        Args:
            article_id: ID of the source article.
            session_id: ID of the agent session.
            summary: Generated summary.
            headlines: Dict with informativo, engagement, seo headlines.
            angles: List of angle dicts.
            cluster_id: Assigned cluster ID (if any).
            quality_score: Quality check score.
            quality_passed: Whether quality check passed.
            llm_provider: Name of LLM provider used.
            generation_time_ms: Generation time in milliseconds.

        Returns:
            Created ProcessedArticle model.
        """
        # Convert angles list to JSON string if provided
        angles_json = None
        if angles:
            import json
            angles_json = json.dumps(angles, ensure_ascii=False)

        processed = ProcessedArticle(
            article_id=article_id,
            session_id=session_id,
            summary=summary,
            headlines=headlines,
            angles=angles_json,
            cluster_id=cluster_id,
            quality_score=quality_score,
            quality_passed=quality_passed,
            llm_provider=llm_provider,
            generation_time_ms=generation_time_ms,
        )
        self.db.add(processed)
        self.db.flush()
        return processed

    def get_with_article(self, processed_id: UUID) -> tuple[ProcessedArticle, Article] | None:
        """Get processed article with its source article."""
        stmt = (
            select(ProcessedArticle, Article)
            .join(Article, ProcessedArticle.article_id == Article.id)
            .where(ProcessedArticle.id == processed_id)
        )
        result = self.db.execute(stmt).first()
        if result:
            return (result.ProcessedArticle, result.Article)
        return None

    def get_recent_summaries(
        self,
        limit: int = 10,
        quality_min: float | None = None,
    ) -> Sequence[tuple[ProcessedArticle, Article]]:
        """
        Get recent processed articles with their source articles.

        Args:
            limit: Maximum results.
            quality_min: Minimum quality score filter.

        Returns:
            List of (ProcessedArticle, Article) tuples.
        """
        stmt = (
            select(ProcessedArticle, Article)
            .join(Article, ProcessedArticle.article_id == Article.id)
            .where(ProcessedArticle.summary.isnot(None))
            .where(ProcessedArticle.summary != "")
        )

        if quality_min is not None:
            stmt = stmt.where(ProcessedArticle.quality_score >= quality_min)

        stmt = stmt.order_by(ProcessedArticle.created_at.desc()).limit(limit)

        results = self.db.execute(stmt).all()
        return [(row.ProcessedArticle, row.Article) for row in results]

    def count(self) -> int:
        """Count total processed articles."""
        stmt = select(func.count()).select_from(ProcessedArticle)
        return self.db.execute(stmt).scalar() or 0

    def count_by_session(self, session_id: UUID) -> int:
        """Count processed articles in a session."""
        stmt = (
            select(func.count())
            .select_from(ProcessedArticle)
            .where(ProcessedArticle.session_id == session_id)
        )
        return self.db.execute(stmt).scalar() or 0

    def get_quality_stats(self, session_id: UUID | None = None) -> dict:
        """
        Get quality statistics.

        Args:
            session_id: Optional session filter.

        Returns:
            Dict with count, avg_score, passed_count, failed_count.
        """
        stmt = select(
            func.count().label("total"),
            func.avg(ProcessedArticle.quality_score).label("avg_score"),
            func.sum(
                func.cast(ProcessedArticle.quality_passed == True, Integer)
            ).label("passed"),
        ).select_from(ProcessedArticle)

        if session_id:
            stmt = stmt.where(ProcessedArticle.session_id == session_id)

        from sqlalchemy import Integer
        result = self.db.execute(stmt).first()

        return {
            "total": result.total or 0,
            "avg_score": float(result.avg_score) if result.avg_score else 0.0,
            "passed": result.passed or 0,
            "failed": (result.total or 0) - (result.passed or 0),
        }

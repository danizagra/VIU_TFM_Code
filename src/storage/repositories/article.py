"""
Repository for Article CRUD operations.

Handles database operations for raw articles,
including deduplication checks and vector similarity search.
"""

from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.connectors.base import RawArticle
from src.storage.models import Article


class ArticleRepository:
    """
    Repository for Article database operations.

    Supports:
    - CRUD operations
    - Deduplication by URL
    - Vector similarity search with pgvector
    """

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db

    def get_by_id(self, article_id: UUID) -> Article | None:
        """Get article by ID."""
        return self.db.get(Article, article_id)

    def get_by_url(self, url: str) -> Article | None:
        """Get article by source URL."""
        stmt = select(Article).where(Article.source_url == url)
        return self.db.execute(stmt).scalar_one_or_none()

    def exists_by_url(self, url: str) -> bool:
        """Check if article exists by URL."""
        stmt = select(func.count()).select_from(Article).where(Article.source_url == url)
        count = self.db.execute(stmt).scalar()
        return count > 0

    def get_existing_urls(self, urls: list[str]) -> set[str]:
        """Get set of URLs that already exist in database."""
        if not urls:
            return set()
        stmt = select(Article.source_url).where(Article.source_url.in_(urls))
        result = self.db.execute(stmt).scalars().all()
        return set(result)

    def create(self, raw_article: RawArticle, embedding: np.ndarray | None = None) -> Article:
        """
        Create a new article from RawArticle.

        Args:
            raw_article: Raw article data from connector.
            embedding: Optional embedding vector.

        Returns:
            Created Article model.
        """
        article = Article(
            external_id=raw_article.external_id,
            title=raw_article.title,
            content=raw_article.content,
            description=raw_article.description,
            source_name=raw_article.source_name,
            source_url=raw_article.source_url,
            author=raw_article.author,
            image_url=raw_article.image_url,
            published_at=raw_article.published_at,
            language=raw_article.language,
            country=raw_article.country,
            category=raw_article.category,
            embedding=embedding.tolist() if embedding is not None else None,
            fetched_at=datetime.now(timezone.utc),
        )
        self.db.add(article)
        self.db.flush()  # Get ID without committing
        return article

    def create_many(
        self,
        raw_articles: list[RawArticle],
        embeddings: np.ndarray | None = None,
    ) -> list[Article]:
        """
        Create multiple articles, skipping existing ones by URL.

        Args:
            raw_articles: List of raw articles.
            embeddings: Optional embeddings matrix (one per article).

        Returns:
            List of created Article models (only new ones).
        """
        # Get existing URLs
        urls = [a.source_url for a in raw_articles if a.source_url]
        existing_urls = self.get_existing_urls(urls)

        created = []
        for i, raw in enumerate(raw_articles):
            # Skip if already exists
            if raw.source_url and raw.source_url in existing_urls:
                continue

            embedding = embeddings[i] if embeddings is not None else None
            article = self.create(raw, embedding)
            created.append(article)

        return created

    def update_embedding(self, article_id: UUID, embedding: np.ndarray) -> None:
        """Update article embedding."""
        article = self.get_by_id(article_id)
        if article:
            article.embedding = embedding.tolist()

    def find_similar(
        self,
        embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.7,
        exclude_ids: list[UUID] | None = None,
        require_content: bool = True,
    ) -> Sequence[tuple[Article, float]]:
        """
        Find similar articles using pgvector cosine distance.

        Args:
            embedding: Query embedding vector.
            limit: Maximum results to return.
            threshold: Minimum similarity (0-1, higher = more similar).
            exclude_ids: Article IDs to exclude from results.
            require_content: If True, only return articles with non-empty content.

        Returns:
            List of (Article, similarity_score) tuples.
        """
        # pgvector uses cosine distance (0 = identical, 2 = opposite)
        # Convert threshold to distance: distance = 1 - similarity
        max_distance = 1 - threshold

        # Build query with cosine distance
        distance = Article.embedding.cosine_distance(embedding.tolist())

        stmt = (
            select(Article, (1 - distance).label("similarity"))
            .where(Article.embedding.isnot(None))
            .where(distance <= max_distance)
        )

        # Filter out articles without content (they cannot be summarized)
        if require_content:
            # Content must exist and be non-empty (at least 50 chars for meaningful content)
            stmt = stmt.where(Article.content.isnot(None))
            stmt = stmt.where(func.length(Article.content) >= 50)

        if exclude_ids:
            stmt = stmt.where(Article.id.notin_(exclude_ids))

        stmt = stmt.order_by(distance).limit(limit)

        results = self.db.execute(stmt).all()
        return [(row.Article, row.similarity) for row in results]

    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[Article]:
        """Find articles within a date range."""
        stmt = select(Article).where(Article.published_at >= start_date)

        if end_date:
            stmt = stmt.where(Article.published_at <= end_date)

        stmt = stmt.order_by(Article.published_at.desc()).limit(limit)

        return self.db.execute(stmt).scalars().all()

    def get_recent(self, days: int = 7, limit: int = 100) -> Sequence[Article]:
        """Get recent articles from the last N days."""
        from datetime import timedelta
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        return self.find_by_date_range(start_date, limit=limit)

    def count(self) -> int:
        """Count total articles."""
        stmt = select(func.count()).select_from(Article)
        return self.db.execute(stmt).scalar() or 0

    def count_with_embeddings(self) -> int:
        """Count articles that have embeddings."""
        stmt = (
            select(func.count())
            .select_from(Article)
            .where(Article.embedding.isnot(None))
        )
        return self.db.execute(stmt).scalar() or 0

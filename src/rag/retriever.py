"""
News retriever for RAG system.

Wraps ArticleRepository to provide semantic search over news articles.
"""

from typing import Optional, Sequence

import structlog
from sqlalchemy.orm import Session

from src.processing.embeddings import EmbeddingGenerator, get_embedding_generator
from src.storage.models import Article
from src.storage.repositories.article import ArticleRepository

logger = structlog.get_logger()


class NewsRetriever:
    """
    Retriever for finding relevant news articles.

    Uses vector similarity search via pgvector to find articles
    semantically similar to a query.

    Usage:
        retriever = NewsRetriever(db, embedder)
        results = retriever.search("economia colombiana", limit=5)
        for article, score in results:
            print(f"{article.title}: {score:.2f}")
    """

    def __init__(
        self,
        db: Session,
        embedder: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the retriever.

        Args:
            db: Database session.
            embedder: Embedding generator (creates one if not provided).
        """
        self.db = db
        self.embedder = embedder or get_embedding_generator()
        self.article_repo = ArticleRepository(db)

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
        category: Optional[str] = None,
    ) -> Sequence[tuple[Article, float]]:
        """
        Search for articles relevant to a query.

        Args:
            query: Natural language query.
            limit: Maximum number of results.
            threshold: Minimum similarity score (0-1).
            category: Optional category filter.

        Returns:
            List of (Article, similarity_score) tuples, sorted by relevance.
        """
        logger.info(
            "Searching for articles",
            query=query[:50],
            limit=limit,
            threshold=threshold,
        )

        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)

        # Vector search
        results = self.article_repo.find_similar(
            embedding=query_embedding,
            limit=limit * 2 if category else limit,  # Over-fetch if filtering
            threshold=threshold,
        )

        # Filter by category if specified
        if category:
            results = [(a, s) for a, s in results if a.category == category]
            results = results[:limit]

        logger.info("Search completed", results_count=len(results))

        return results

    def search_by_date_range(
        self,
        query: str,
        days: int = 7,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> Sequence[tuple[Article, float]]:
        """
        Search for recent articles relevant to a query.

        Args:
            query: Natural language query.
            days: Number of days to look back.
            limit: Maximum number of results.
            threshold: Minimum similarity score.

        Returns:
            List of (Article, similarity_score) tuples.
        """
        from datetime import datetime, timedelta, timezone

        # Get recent articles first
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_articles = self.article_repo.find_by_date_range(
            start_date=start_date,
            limit=limit * 5,  # Get more to filter by similarity
        )

        if not recent_articles:
            return []

        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)

        # Calculate similarity for each article
        import numpy as np

        results = []
        for article in recent_articles:
            if article.embedding:
                # Cosine similarity
                article_emb = np.array(article.embedding)
                similarity = np.dot(query_embedding, article_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(article_emb)
                )

                if similarity >= threshold:
                    results.append((article, float(similarity)))

        # Sort by similarity and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_by_ids(self, article_ids: list) -> list[Article]:
        """
        Get articles by their IDs.

        Args:
            article_ids: List of article UUIDs.

        Returns:
            List of Article objects.
        """
        articles = []
        for aid in article_ids:
            article = self.article_repo.get_by_id(aid)
            if article:
                articles.append(article)
        return articles

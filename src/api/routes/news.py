"""
News routes for article retrieval and search.

Endpoints:
- GET /news/latest - Get latest processed articles
- GET /news/search - Search articles by text
- GET /news/{id} - Get article by ID
"""

import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import (
    ArticleListResponse,
    ArticleResponse,
    ProcessedArticleResponse,
)
from src.storage.models import Article, ProcessedArticle
from src.storage.repositories.article import ArticleRepository
from src.storage.repositories.processed import ProcessedArticleRepository

logger = structlog.get_logger()

router = APIRouter(prefix="/news", tags=["News"])


def _processed_to_response(
    processed: ProcessedArticle, article: Article
) -> ProcessedArticleResponse:
    """Convert database models to response schema."""
    # Parse angles if stored as JSON string
    angles = None
    if processed.angles:
        try:
            angles = json.loads(processed.angles) if isinstance(processed.angles, str) else processed.angles
        except json.JSONDecodeError:
            angles = None

    return ProcessedArticleResponse(
        id=processed.id,
        article_id=processed.article_id,
        summary=processed.summary,
        headlines=processed.headlines,
        angles=angles,
        quality_score=processed.quality_score,
        quality_passed=processed.quality_passed,
        created_at=processed.created_at,
        title=article.title,
        source_name=article.source_name,
        source_url=article.source_url,
        published_at=article.published_at,
    )


@router.get("/latest", response_model=ArticleListResponse)
def get_latest_news(
    limit: int = Query(10, ge=1, le=100, description="Number of articles to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    quality_min: Optional[float] = Query(
        None, ge=0, le=1, description="Minimum quality score filter"
    ),
    db: Session = Depends(get_db),
) -> ArticleListResponse:
    """
    Get the latest processed news articles.

    Returns articles with generated summaries, headlines, and angles,
    ordered by processing date (newest first).
    """
    logger.info("Fetching latest news", limit=limit, offset=offset)

    processed_repo = ProcessedArticleRepository(db)

    # Get total count for pagination
    total = processed_repo.count()

    # Get recent summaries (limit + offset to handle pagination)
    results = processed_repo.get_recent_summaries(
        limit=limit + offset,
        quality_min=quality_min,
    )

    # Apply offset manually (repository doesn't support offset)
    results = list(results)[offset : offset + limit]

    items = [_processed_to_response(p, a) for p, a in results]

    return ArticleListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/search")
def search_news(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db),
) -> ArticleListResponse:
    """
    Search news articles by text query.

    Uses vector similarity search on article embeddings.
    """
    from src.processing.embeddings import get_embedding_generator

    logger.info("Searching news", query=q, limit=limit)

    article_repo = ArticleRepository(db)
    processed_repo = ProcessedArticleRepository(db)

    # Generate embedding for query
    embedder = get_embedding_generator()
    query_embedding = embedder.embed_single(q)

    # Vector search
    similar_articles = article_repo.find_similar(
        embedding=query_embedding,
        limit=limit * 2,  # Over-fetch to filter
        threshold=0.5,  # Good threshold for nomic embeddings
    )

    # Filter by category if specified
    if category:
        similar_articles = [
            (a, s) for a, s in similar_articles if a.category == category
        ]

    # Get processed versions of these articles
    items = []
    for article, similarity in similar_articles[:limit]:
        processed_list = processed_repo.get_by_article_id(article.id)
        if processed_list:
            # Use the most recent processed version
            processed = processed_list[-1]
            items.append(_processed_to_response(processed, article))

    return ArticleListResponse(
        items=items,
        total=len(items),
        limit=limit,
        offset=0,
    )


@router.get("/{article_id}", response_model=ArticleResponse)
def get_article(
    article_id: UUID,
    db: Session = Depends(get_db),
) -> ArticleResponse:
    """
    Get a specific article by ID.

    Returns the raw article data without processed content.
    """
    logger.info("Fetching article", article_id=str(article_id))

    article_repo = ArticleRepository(db)
    article = article_repo.get_by_id(article_id)

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return ArticleResponse(
        id=article.id,
        title=article.title,
        content=article.content,
        description=article.description,
        source_name=article.source_name,
        source_url=article.source_url,
        author=article.author,
        image_url=article.image_url,
        published_at=article.published_at,
        category=article.category,
        language=article.language,
        fetched_at=article.fetched_at,
        created_at=article.created_at,
    )


@router.get("/{article_id}/processed", response_model=ProcessedArticleResponse)
def get_processed_article(
    article_id: UUID,
    db: Session = Depends(get_db),
) -> ProcessedArticleResponse:
    """
    Get the processed version of an article with generated content.

    Returns summary, headlines, and angles.
    """
    logger.info("Fetching processed article", article_id=str(article_id))

    article_repo = ArticleRepository(db)
    processed_repo = ProcessedArticleRepository(db)

    article = article_repo.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    processed_list = processed_repo.get_by_article_id(article_id)
    if not processed_list:
        raise HTTPException(
            status_code=404, detail="No processed content for this article"
        )

    # Return the most recent processed version
    processed = processed_list[-1]
    return _processed_to_response(processed, article)

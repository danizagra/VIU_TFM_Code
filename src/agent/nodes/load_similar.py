"""
Load similar node - Loads historical articles from database for context.

Part of the búsqueda híbrida (Option C) implementation.
Uses pgvector to find similar historical articles.
"""

import structlog
import numpy as np

from src.agent.state import AgentState
from src.config.settings import settings
from src.connectors.base import RawArticle
from src.storage.database import get_db
from src.storage.models import Article
from src.storage.repositories.article import ArticleRepository

logger = structlog.get_logger()


def db_article_to_raw(article: Article) -> RawArticle:
    """Convert database Article to RawArticle."""
    return RawArticle(
        title=article.title,
        source_name=article.source_name,
        source_url=article.source_url,
        description=article.description,
        content=article.content,
        author=article.author,
        image_url=article.image_url,
        published_at=article.published_at,
        language=article.language,
        country=article.country,
        category=article.category,
        external_id=article.external_id,
    )


def load_similar_from_db(state: AgentState) -> AgentState:
    """
    Load similar historical articles from database.

    Uses vector similarity search with pgvector to find
    related articles from the database, enriching the
    clustering context.

    Reads:
        - embeddings: Embeddings of current articles
        - filtered_articles: Current articles being processed

    Writes:
        - historical_articles: Similar articles from DB
        - historical_embeddings: Their embeddings
        - current_step: Updated to 'loaded_similar'
    """
    state["current_step"] = "loading_similar"

    embeddings = state.get("embeddings")
    articles = state.get("filtered_articles", [])

    if embeddings is None or len(articles) == 0:
        logger.info("No embeddings available, skipping historical load")
        state["historical_articles"] = []
        state["historical_embeddings"] = None
        state["current_step"] = "loaded_similar"
        return state

    logger.info("Loading similar articles from database", current_count=len(articles))

    try:
        with get_db() as db:
            article_repo = ArticleRepository(db)

            # Check if we have any articles with embeddings in DB
            count_with_embeddings = article_repo.count_with_embeddings()

            if count_with_embeddings == 0:
                logger.info("No historical articles with embeddings in database")
                state["historical_articles"] = []
                state["historical_embeddings"] = None
                state["current_step"] = "loaded_similar"
                return state

            # Find similar articles for each current article
            historical_set = {}  # Use dict to dedupe by URL
            similarity_threshold = settings.similarity_threshold  # Minimum similarity to include
            max_per_article = 3  # Max similar articles per current article

            for i, embedding in enumerate(embeddings):
                similar = article_repo.find_similar(
                    embedding=embedding,
                    limit=max_per_article,
                    threshold=similarity_threshold,
                )

                for article, similarity in similar:
                    if article.source_url and article.source_url not in historical_set:
                        historical_set[article.source_url] = (article, similarity)

            # Convert to lists
            historical_articles = []
            historical_embeddings = []

            for article, similarity in historical_set.values():
                raw = db_article_to_raw(article)
                historical_articles.append(raw)

                if article.embedding is not None:
                    historical_embeddings.append(np.array(article.embedding))

            state["historical_articles"] = historical_articles
            state["historical_embeddings"] = (
                np.array(historical_embeddings) if historical_embeddings else None
            )
            state["current_step"] = "loaded_similar"

            logger.info(
                "Loaded similar historical articles",
                count=len(historical_articles),
                from_db_total=count_with_embeddings,
            )

    except Exception as e:
        error_msg = f"Error loading similar articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["historical_articles"] = []
        state["historical_embeddings"] = None
        state["current_step"] = "load_similar_failed"

    return state


def merge_with_historical(state: AgentState) -> AgentState:
    """
    Merge current articles with historical ones for clustering.

    This allows clustering to group new articles with
    related historical coverage.

    Reads:
        - filtered_articles: Current articles
        - historical_articles: Similar articles from DB
        - embeddings: Current embeddings
        - historical_embeddings: Historical embeddings

    Writes:
        - all_articles: Combined articles for clustering
        - all_embeddings: Combined embeddings
        - historical_indices: Indices of historical articles
        - current_step: Updated to 'merged'
    """
    state["current_step"] = "merging"

    current_articles = state.get("filtered_articles", [])
    historical_articles = state.get("historical_articles", [])
    current_embeddings = state.get("embeddings")
    historical_embeddings = state.get("historical_embeddings")

    # If no historical, just use current
    if not historical_articles:
        state["all_articles"] = current_articles
        state["all_embeddings"] = current_embeddings
        state["historical_indices"] = []
        state["current_step"] = "merged"
        return state

    # Merge articles
    all_articles = current_articles + historical_articles
    historical_indices = list(range(len(current_articles), len(all_articles)))

    # Merge embeddings
    if current_embeddings is not None and historical_embeddings is not None:
        all_embeddings = np.vstack([current_embeddings, historical_embeddings])
    else:
        all_embeddings = current_embeddings

    state["all_articles"] = all_articles
    state["all_embeddings"] = all_embeddings
    state["historical_indices"] = historical_indices
    state["current_step"] = "merged"

    logger.info(
        "Merged articles",
        current=len(current_articles),
        historical=len(historical_articles),
        total=len(all_articles),
    )

    return state

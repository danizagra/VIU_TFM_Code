"""
Deduplicate node - Removes duplicate articles.
"""

import structlog

from src.agent.state import AgentState
from src.config.settings import settings
from src.processing.deduplication import ArticleDeduplicator

logger = structlog.get_logger()


def deduplicate_articles(state: AgentState) -> AgentState:
    """
    Remove duplicate articles based on embedding similarity.

    Reads:
        - filtered_articles: Articles to deduplicate
        - embeddings: Article embeddings

    Writes:
        - deduplicated_articles: Unique articles
        - duplicate_groups: Groups of duplicate indices
        - current_step: Updated to 'deduplicated'
    """
    state["current_step"] = "deduplicating"

    articles = state.get("filtered_articles", [])
    embeddings = state.get("embeddings")

    if not articles:
        logger.warning("No articles to deduplicate")
        state["deduplicated_articles"] = []
        state["duplicate_groups"] = []
        state["current_step"] = "deduplicated"
        return state

    if embeddings is None:
        logger.warning("No embeddings available, skipping deduplication")
        state["deduplicated_articles"] = articles
        state["duplicate_groups"] = []
        state["current_step"] = "deduplicated"
        return state

    logger.info("Deduplicating articles", count=len(articles))

    try:
        deduplicator = ArticleDeduplicator(
            duplicate_threshold=settings.duplicate_threshold,
            selection_strategy="longest",
        )

        result = deduplicator.deduplicate(articles, embeddings)

        state["deduplicated_articles"] = result.unique_articles
        state["duplicate_groups"] = result.duplicate_groups
        state["current_step"] = "deduplicated"

        logger.info(
            "Deduplication complete",
            original=len(articles),
            unique=len(result.unique_articles),
            removed=result.duplicates_removed,
            rate=f"{result.deduplication_rate:.1%}",
        )

    except Exception as e:
        error_msg = f"Error deduplicating articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        # On error, keep all articles
        state["deduplicated_articles"] = articles
        state["duplicate_groups"] = []
        state["current_step"] = "dedup_failed"

    return state

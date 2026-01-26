"""
Filter node - Applies quality and content filters to articles.
"""

import structlog

from src.agent.state import AgentState
from src.processing.filters import ArticleFilter

logger = structlog.get_logger()


def filter_articles(state: AgentState) -> AgentState:
    """
    Filter articles by quality and content criteria.

    Reads:
        - raw_articles: Articles to filter

    Writes:
        - filtered_articles: Articles that passed filters
        - current_step: Updated to 'filtered'
    """
    state["current_step"] = "filtering"

    raw_articles = state.get("raw_articles", [])

    if not raw_articles:
        logger.warning("No articles to filter")
        state["filtered_articles"] = []
        state["current_step"] = "filtered"
        return state

    logger.info("Filtering articles", count=len(raw_articles))

    try:
        # Build filter chain
        filter_chain = (
            ArticleFilter()
            .add_content_filter(
                min_title_length=10,
                min_content_length=50,
            )
            .add_date_filter(days=7)  # Last 7 days
        )

        # Apply filters
        filtered = filter_chain.apply(raw_articles)

        state["filtered_articles"] = filtered
        state["current_step"] = "filtered"

        logger.info(
            "Articles filtered",
            original=len(raw_articles),
            filtered=len(filtered),
            removed=len(raw_articles) - len(filtered),
        )

    except Exception as e:
        error_msg = f"Error filtering articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        # On error, pass through all articles
        state["filtered_articles"] = raw_articles
        state["current_step"] = "filter_failed"

    return state

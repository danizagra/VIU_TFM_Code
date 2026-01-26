"""
Check existing node - Filters out articles already in database.

Part of the cache inteligente (Option B) implementation.
"""

import structlog

from src.agent.state import AgentState
from src.storage.database import get_db
from src.storage.repositories.article import ArticleRepository

logger = structlog.get_logger()


def check_existing_articles(state: AgentState) -> AgentState:
    """
    Check which articles already exist in database and filter them out.

    This implements the "cache inteligente" pattern - avoiding
    reprocessing of articles we've already seen.

    Reads:
        - raw_articles: Fetched articles to check

    Writes:
        - raw_articles: Updated to only include new articles
        - existing_urls: Set of URLs that already exist
        - skipped_count: Number of articles skipped
        - current_step: Updated to 'checked_existing'
    """
    state["current_step"] = "checking_existing"

    raw_articles = state.get("raw_articles", [])

    if not raw_articles:
        logger.warning("No articles to check")
        state["existing_urls"] = set()
        state["skipped_count"] = 0
        state["current_step"] = "checked_existing"
        return state

    # Collect URLs to check
    urls = [a.source_url for a in raw_articles if a.source_url]

    if not urls:
        logger.warning("No URLs to check")
        state["existing_urls"] = set()
        state["skipped_count"] = 0
        state["current_step"] = "checked_existing"
        return state

    logger.info("Checking for existing articles", urls_to_check=len(urls))

    try:
        with get_db() as db:
            article_repo = ArticleRepository(db)

            # Get existing URLs
            existing_urls = article_repo.get_existing_urls(urls)

            # Filter out existing articles
            new_articles = [
                a for a in raw_articles
                if not a.source_url or a.source_url not in existing_urls
            ]

            skipped_count = len(raw_articles) - len(new_articles)

            state["raw_articles"] = new_articles
            state["existing_urls"] = existing_urls
            state["skipped_count"] = skipped_count
            state["current_step"] = "checked_existing"

            logger.info(
                "Existing articles check complete",
                original=len(raw_articles),
                existing=len(existing_urls),
                new=len(new_articles),
                skipped=skipped_count,
            )

    except Exception as e:
        error_msg = f"Error checking existing articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        # On error, keep all articles (don't skip any)
        state["existing_urls"] = set()
        state["skipped_count"] = 0
        state["current_step"] = "check_existing_failed"

    return state

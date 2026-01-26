"""
Fetch node - Retrieves articles from news sources.
"""

import structlog

from src.agent.state import AgentState
from src.config.settings import settings
from src.connectors import create_default_aggregator

logger = structlog.get_logger()

# Valid source options
VALID_SOURCES = {"rss", "newsapi", "gnews", "all"}


def fetch_articles(state: AgentState) -> AgentState:
    """
    Fetch articles from configured news sources.

    Reads:
        - query: Optional search query
        - max_articles: Maximum articles to fetch
        - sources: List of sources to use ('rss', 'newsapi', 'gnews', 'all')

    Writes:
        - raw_articles: List of fetched RawArticle objects
        - current_step: Updated to 'fetched'
        - errors: Any errors encountered
    """
    state["current_step"] = "fetching"

    query = state.get("query", "")
    max_articles = state.get("max_articles", 20)
    sources = state.get("sources", ["rss"])

    # Parse sources
    if "all" in sources:
        include_rss = True
        include_newsapi = True
        include_gnews = True
    else:
        include_rss = "rss" in sources
        include_newsapi = "newsapi" in sources
        include_gnews = "gnews" in sources

    logger.info(
        "Fetching articles",
        query=query or "(no query)",
        max_articles=max_articles,
        sources=sources,
    )

    try:
        # Create aggregator with configured sources
        aggregator = create_default_aggregator(
            include_newsapi=include_newsapi,
            include_gnews=include_gnews,
            include_colombian_rss=include_rss,
        )

        # Fetch articles with language and country from settings
        articles = aggregator.fetch_all(
            query=query if query else None,
            language=settings.default_language,  # 'es' by default
            country=settings.default_region,      # 'co' by default
            max_results=max_articles,
        )

        state["raw_articles"] = articles
        state["current_step"] = "fetched"

        logger.info("Articles fetched", count=len(articles))

    except Exception as e:
        error_msg = f"Error fetching articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["raw_articles"] = []
        state["current_step"] = "fetch_failed"

    return state

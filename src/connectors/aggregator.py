"""
News aggregator that combines multiple news sources.

This module provides a unified interface to fetch articles
from multiple connectors (NewsAPI, GNews, RSS feeds).
"""

import hashlib
from datetime import datetime
from typing import Optional
import structlog

from src.config.settings import settings
from src.connectors.base import NewsConnector, RawArticle
from src.connectors.rss import RSSConnector, COLOMBIAN_FEEDS, SPANISH_FEEDS


logger = structlog.get_logger()


class NewsAggregator:
    """
    Aggregator for multiple news sources.

    Combines articles from various connectors, handles deduplication,
    and provides a unified fetch interface.

    Usage:
        aggregator = NewsAggregator()
        aggregator.add_connector(NewsAPIConnector())
        aggregator.add_connector(GNewsConnector())

        articles = aggregator.fetch_all(
            query="inteligencia artificial",
            language="es",
            max_results=100
        )
    """

    def __init__(self):
        """Initialize empty aggregator."""
        self._connectors: list[NewsConnector] = []

    def add_connector(self, connector: NewsConnector) -> None:
        """Add a news connector to the aggregator."""
        self._connectors.append(connector)
        logger.info(
            "Added connector",
            connector=connector.source_name
        )

    def add_rss_feeds(self, feeds: list[dict]) -> None:
        """
        Add multiple RSS feeds as connectors.

        Args:
            feeds: List of feed configs with feed_url, source_name, etc.
        """
        for feed in feeds:
            connector = RSSConnector(**feed)
            self.add_connector(connector)

    @property
    def connectors(self) -> list[NewsConnector]:
        """Get list of registered connectors."""
        return self._connectors.copy()

    def fetch_all(
        self,
        query: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100,
        deduplicate: bool = True
    ) -> list[RawArticle]:
        """
        Fetch articles from all registered connectors.

        Args:
            query: Search query (keywords).
            language: Language code (e.g., 'es', 'en').
            country: Country code (e.g., 'co', 'us').
            category: News category.
            from_date: Start date for articles.
            to_date: End date for articles.
            max_results: Maximum total articles to return.
            deduplicate: Remove duplicate articles by URL/title.

        Returns:
            Combined list of RawArticle objects from all sources.
        """
        all_articles: list[RawArticle] = []

        # Calculate articles per connector (distribute evenly)
        per_connector = max(10, max_results // max(len(self._connectors), 1))

        for connector in self._connectors:
            try:
                logger.info(
                    "Fetching from connector",
                    connector=connector.source_name,
                    query=query
                )

                articles = connector.fetch_articles(
                    query=query,
                    language=language,
                    country=country,
                    category=category,
                    from_date=from_date,
                    to_date=to_date,
                    max_results=per_connector
                )

                logger.info(
                    "Fetched articles",
                    connector=connector.source_name,
                    count=len(articles)
                )

                all_articles.extend(articles)

            except Exception as e:
                logger.warning(
                    "Connector failed",
                    connector=connector.source_name,
                    error=str(e)
                )
                continue

        # Deduplicate
        if deduplicate:
            original_count = len(all_articles)
            all_articles = self._deduplicate(all_articles)
            logger.info(
                "Deduplicated articles",
                original=original_count,
                after=len(all_articles)
            )

        # Sort by published date (newest first)
        all_articles.sort(
            key=lambda a: a.published_at or datetime.min,
            reverse=True
        )

        return all_articles[:max_results]

    def _deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """
        Remove duplicate articles based on URL and title similarity.

        Uses URL as primary key, falls back to title hash.
        """
        seen: set[str] = set()
        unique: list[RawArticle] = []

        for article in articles:
            # Generate unique key
            key = self._get_article_key(article)

            if key not in seen:
                seen.add(key)
                unique.append(article)

        return unique

    def _get_article_key(self, article: RawArticle) -> str:
        """Generate a unique key for an article."""
        # Prefer URL as unique identifier
        if article.source_url:
            return article.source_url

        # Fall back to title hash
        title_normalized = article.title.lower().strip()
        return hashlib.md5(title_normalized.encode()).hexdigest()


def create_default_aggregator(
    include_newsapi: bool = True,
    include_gnews: bool = True,
    include_colombian_rss: bool = True,
    include_spanish_rss: bool = False
) -> NewsAggregator:
    """
    Create an aggregator with default connectors.

    Args:
        include_newsapi: Include NewsAPI connector (requires API key).
        include_gnews: Include GNews connector (requires API key).
        include_colombian_rss: Include Colombian news RSS feeds.
        include_spanish_rss: Include Spanish news RSS feeds.

    Returns:
        Configured NewsAggregator instance.
    """
    aggregator = NewsAggregator()

    # Add API connectors if configured
    if include_newsapi and settings.newsapi_key:
        try:
            from src.connectors.newsapi import NewsAPIConnector
            aggregator.add_connector(NewsAPIConnector())
        except ValueError as e:
            logger.warning("NewsAPI not configured", error=str(e))

    if include_gnews and settings.gnews_api_key:
        try:
            from src.connectors.gnews import GNewsConnector
            aggregator.add_connector(GNewsConnector())
        except ValueError as e:
            logger.warning("GNews not configured", error=str(e))

    # Add RSS feeds
    if include_colombian_rss:
        aggregator.add_rss_feeds(COLOMBIAN_FEEDS)

    if include_spanish_rss:
        aggregator.add_rss_feeds(SPANISH_FEEDS)

    return aggregator

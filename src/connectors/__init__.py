"""
News source connectors module.

Usage:
    from src.connectors import (
        NewsAggregator,
        create_default_aggregator,
        RawArticle
    )

    # Quick start with default connectors
    aggregator = create_default_aggregator()
    articles = aggregator.fetch_all(query="tecnología", language="es")

    # Or use individual connectors
    from src.connectors import NewsAPIConnector, RSSConnector
"""

from src.connectors.base import NewsConnector, RawArticle
from src.connectors.aggregator import NewsAggregator, create_default_aggregator

__all__ = [
    # Base classes
    "NewsConnector",
    "RawArticle",
    # Aggregator
    "NewsAggregator",
    "create_default_aggregator",
]

# Lazy imports for connectors (to avoid errors if API keys not configured)
def __getattr__(name: str):
    if name == "NewsAPIConnector":
        from src.connectors.newsapi import NewsAPIConnector
        return NewsAPIConnector
    elif name == "GNewsConnector":
        from src.connectors.gnews import GNewsConnector
        return GNewsConnector
    elif name == "RSSConnector":
        from src.connectors.rss import RSSConnector
        return RSSConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Base classes for news source connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawArticle:
    """
    Raw article data from a news source.

    This is the common format for articles fetched from any connector.
    It will be converted to the Article model for database storage.
    """
    title: str
    source_name: str

    # Content (at least one should be present)
    content: Optional[str] = None
    description: Optional[str] = None

    # Source info
    source_url: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None

    # Metadata
    published_at: Optional[datetime] = None
    language: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None

    # Connector-specific ID (for deduplication)
    external_id: Optional[str] = None

    # Additional metadata from the source
    extra: dict = field(default_factory=dict)

    def has_content(self) -> bool:
        """Check if the article has meaningful content."""
        return bool(self.content or self.description)

    def get_text_for_embedding(self) -> str:
        """Get combined text for embedding generation."""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.content and self.content != self.description:
            parts.append(self.content[:500])  # Limit content length
        return " ".join(parts)


class NewsConnector(ABC):
    """
    Abstract base class for news source connectors.

    All connectors must implement the fetch_articles method.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this news source."""
        pass

    @abstractmethod
    def fetch_articles(
        self,
        query: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> list[RawArticle]:
        """
        Fetch articles from the news source.

        Args:
            query: Search query (keywords).
            language: Language code (e.g., 'es', 'en').
            country: Country code (e.g., 'co', 'us').
            category: News category (e.g., 'technology', 'business').
            from_date: Start date for articles.
            to_date: End date for articles.
            max_results: Maximum number of articles to return.

        Returns:
            List of RawArticle objects.
        """
        pass

    def is_available(self) -> bool:
        """
        Check if the connector is properly configured and available.

        Returns:
            True if the connector can fetch articles.
        """
        try:
            # Try to fetch a small number of articles
            self.fetch_articles(max_results=1)
            return True
        except Exception:
            return False

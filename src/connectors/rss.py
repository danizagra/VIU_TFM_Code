"""
RSS/Atom feed connector for fetching news articles.

This connector can parse any standard RSS or Atom feed,
making it useful for sources that don't have APIs.
"""

from datetime import datetime
from typing import Optional
from email.utils import parsedate_to_datetime

import feedparser

from src.connectors.base import NewsConnector, RawArticle


class RSSConnector(NewsConnector):
    """
    Connector for RSS/Atom feeds.

    Usage:
        connector = RSSConnector(
            feed_url="https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
            source_name="El País"
        )
        articles = connector.fetch_articles(max_results=20)

    Common Colombian news RSS feeds:
        - El Tiempo: https://www.eltiempo.com/rss/home.xml
        - El Espectador: https://www.elespectador.com/rss.xml
        - Semana: https://www.semana.com/rss.xml
    """

    def __init__(
        self,
        feed_url: str,
        source_name: str,
        language: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None
    ):
        """
        Initialize RSS connector.

        Args:
            feed_url: URL of the RSS/Atom feed.
            source_name: Name to identify this source.
            language: Default language for articles.
            country: Default country for articles.
            category: Default category for articles.
        """
        self._feed_url = feed_url
        self._source_name = source_name
        self._language = language
        self._country = country
        self._category = category

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def feed_url(self) -> str:
        return self._feed_url

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
        Fetch articles from the RSS feed.

        Note: RSS feeds don't support search queries,
        so the query parameter filters results client-side.
        """
        articles = []

        try:
            feed = feedparser.parse(self._feed_url)

            if feed.bozo and not feed.entries:
                raise RuntimeError(
                    f"Failed to parse RSS feed: {feed.bozo_exception}"
                )

            for entry in feed.entries:
                article = self._parse_entry(
                    entry,
                    language=language or self._language,
                    country=country or self._country,
                    category=category or self._category
                )

                if not article or not article.has_content():
                    continue

                # Apply client-side filters
                if query and not self._matches_query(article, query):
                    continue

                if from_date and article.published_at:
                    if article.published_at < from_date:
                        continue

                if to_date and article.published_at:
                    if article.published_at > to_date:
                        continue

                articles.append(article)

                if len(articles) >= max_results:
                    break

        except Exception as e:
            if "parse" not in str(e).lower():
                raise RuntimeError(f"RSS fetch error: {e}") from e
            raise

        return articles

    def _parse_entry(
        self,
        entry: dict,
        language: Optional[str],
        country: Optional[str],
        category: Optional[str]
    ) -> Optional[RawArticle]:
        """Parse an RSS feed entry into RawArticle."""
        title = entry.get("title")
        if not title:
            return None

        # Parse published date
        published_at = None
        if entry.get("published"):
            published_at = self._parse_date(entry["published"])
        elif entry.get("updated"):
            published_at = self._parse_date(entry["updated"])

        # Get content
        content = None
        if entry.get("content"):
            content = entry["content"][0].get("value", "")
        elif entry.get("summary_detail"):
            content = entry["summary_detail"].get("value", "")

        description = entry.get("summary", "")

        # Clean HTML from content/description
        content = self._strip_html(content) if content else None
        description = self._strip_html(description) if description else None

        # Get image
        image_url = None
        if entry.get("media_content"):
            image_url = entry["media_content"][0].get("url")
        elif entry.get("media_thumbnail"):
            image_url = entry["media_thumbnail"][0].get("url")
        elif entry.get("enclosures"):
            for enc in entry["enclosures"]:
                if enc.get("type", "").startswith("image"):
                    image_url = enc.get("href")
                    break

        # Get author
        author = entry.get("author")
        if not author and entry.get("authors"):
            author = entry["authors"][0].get("name")

        # Get category from feed if not provided
        feed_category = category
        if not feed_category and entry.get("tags"):
            feed_category = entry["tags"][0].get("term")

        return RawArticle(
            title=title,
            source_name=self._source_name,
            content=content,
            description=description,
            source_url=entry.get("link"),
            author=author,
            image_url=image_url,
            published_at=published_at,
            language=language,
            country=country,
            category=feed_category,
            external_id=entry.get("id") or entry.get("link"),
            extra={
                "feed_url": self._feed_url,
            }
        )

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        try:
            # Try standard RSS date format (RFC 2822)
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        return None

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""

        import re
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _matches_query(self, article: RawArticle, query: str) -> bool:
        """Check if article matches search query (case-insensitive)."""
        query_lower = query.lower()
        text = article.get_text_for_embedding().lower()
        return query_lower in text


# Pre-configured feeds for common Spanish/Colombian sources
COLOMBIAN_FEEDS = [
    {
        "feed_url": "https://www.eltiempo.com/rss/colombia.xml",
        "source_name": "El Tiempo - Colombia",
        "language": "es",
        "country": "co"
    },
    {
        "feed_url": "https://www.eltiempo.com/rss/economia.xml",
        "source_name": "El Tiempo - Economía",
        "language": "es",
        "country": "co",
        "category": "business"
    },
    {
        "feed_url": "https://www.eltiempo.com/rss/tecnosfera.xml",
        "source_name": "El Tiempo - Tecnología",
        "language": "es",
        "country": "co",
        "category": "technology"
    },
    {
        "feed_url": "https://www.eltiempo.com/rss/politica.xml",
        "source_name": "El Tiempo - Política",
        "language": "es",
        "country": "co",
        "category": "politics"
    },
]

SPANISH_FEEDS = [
    {
        "feed_url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
        "source_name": "El País",
        "language": "es",
        "country": "es"
    },
    {
        "feed_url": "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml",
        "source_name": "El Mundo",
        "language": "es",
        "country": "es"
    },
]

TECH_FEEDS = [
    {
        "feed_url": "https://feeds.feedburner.com/TechCrunch",
        "source_name": "TechCrunch",
        "language": "en",
        "country": "us",
        "category": "technology"
    },
    {
        "feed_url": "https://www.wired.com/feed/rss",
        "source_name": "Wired",
        "language": "en",
        "country": "us",
        "category": "technology"
    },
]

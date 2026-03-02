"""
GNews API connector for fetching news articles.

GNews (https://gnews.io) provides access to Google News data
with a simple REST API.

Free tier limitations:
- 100 requests per day
- 10 articles per request
- Limited historical data
"""

from datetime import datetime
from typing import Optional

import httpx

from src.config.settings import settings
from src.connectors.base import NewsConnector, RawArticle


class GNewsConnector(NewsConnector):
    """
    Connector for GNews API.

    Usage:
        connector = GNewsConnector()
        articles = connector.fetch_articles(
            query="tecnología",
            language="es",
            country="co",
            max_results=50
        )
    """

    API_BASE_URL = "https://gnews.io/api/v4"

    # Valid categories for GNews
    VALID_CATEGORIES = [
        "general", "world", "nation", "business", "technology",
        "entertainment", "sports", "science", "health"
    ]

    # Language codes supported
    VALID_LANGUAGES = [
        "ar", "zh", "nl", "en", "fr", "de", "el", "he", "hi",
        "it", "ja", "ml", "mr", "no", "pt", "ro", "ru", "es",
        "sv", "ta", "te", "uk"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GNews connector.

        Args:
            api_key: GNews API key (default from settings).
        """
        self._api_key = api_key or settings.gnews_api_key

        if not self._api_key:
            raise ValueError(
                "GNews API key not configured. "
                "Set GNEWS_API_KEY in your .env file."
            )

    @property
    def source_name(self) -> str:
        return "GNews"

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
        Fetch articles from GNews API.

        Uses /search endpoint for queries,
        /top-headlines for category browsing.
        """
        articles = []

        # GNews returns max 10 articles per request on free tier
        # We need to paginate to get more
        page_size = 10
        fetched = 0

        try:
            while fetched < max_results:
                if query:
                    response = self._search(
                        query=query,
                        language=language,
                        country=country,
                        from_date=from_date,
                        to_date=to_date,
                        max_results=min(page_size, max_results - fetched)
                    )
                else:
                    response = self._top_headlines(
                        language=language,
                        country=country,
                        category=category,
                        max_results=min(page_size, max_results - fetched)
                    )

                items = response.get("articles", [])
                if not items:
                    break

                for item in items:
                    article = self._parse_article(item, language, country, category)
                    if article and article.has_content():
                        articles.append(article)
                        fetched += 1

                # GNews free tier doesn't support pagination
                # so we break after first request
                break

        except httpx.HTTPError as e:
            raise RuntimeError(f"GNews API error: {e}") from e

        return articles[:max_results]

    @staticmethod
    def _clean_query(query: str) -> str:
        """Clean query for GNews API compatibility.

        Removes characters that cause 400 errors (¿, ?, │, etc.)
        and extracts only meaningful keywords.
        """
        import re

        # Remove characters that GNews API doesn't handle
        cleaned = re.sub(r'[¿?│|!¡"""\'«»]', ' ', query)
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _search(
        self,
        query: str,
        language: Optional[str],
        country: Optional[str],
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        max_results: int
    ) -> dict:
        """Search articles using /search endpoint."""
        params = {
            "q": self._clean_query(query),
            "token": self._api_key,
            "max": max_results
        }

        if language and language in self.VALID_LANGUAGES:
            params["lang"] = language
        if country:
            params["country"] = country.lower()
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self._make_request(f"{self.API_BASE_URL}/search", params)

    def _top_headlines(
        self,
        language: Optional[str],
        country: Optional[str],
        category: Optional[str],
        max_results: int
    ) -> dict:
        """Get top headlines using /top-headlines endpoint."""
        params = {
            "token": self._api_key,
            "max": max_results
        }

        if language and language in self.VALID_LANGUAGES:
            params["lang"] = language
        if country:
            params["country"] = country.lower()
        if category and category.lower() in self.VALID_CATEGORIES:
            params["category"] = category.lower()

        return self._make_request(f"{self.API_BASE_URL}/top-headlines", params)

    def _make_request(self, url: str, params: dict) -> dict:
        """Make HTTP request to GNews API."""
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)

            if response.status_code == 403:
                raise ValueError("Invalid GNews API key")
            elif response.status_code == 429:
                raise RuntimeError("GNews API rate limit exceeded")

            response.raise_for_status()
            return response.json()

    def _parse_article(
        self,
        item: dict,
        language: Optional[str],
        country: Optional[str],
        category: Optional[str]
    ) -> Optional[RawArticle]:
        """Parse a GNews article item into RawArticle."""
        title = item.get("title")
        if not title:
            return None

        source = item.get("source", {})
        published_at = None

        if item.get("publishedAt"):
            try:
                published_at = datetime.fromisoformat(
                    item["publishedAt"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return RawArticle(
            title=title,
            source_name=source.get("name", "Unknown"),
            content=item.get("content"),
            description=item.get("description"),
            source_url=item.get("url"),
            author=None,  # GNews doesn't provide author
            image_url=item.get("image"),
            published_at=published_at,
            language=language,
            country=country,
            category=category,
            external_id=item.get("url"),  # URL as unique ID
            extra={
                "source_url": source.get("url"),
            }
        )

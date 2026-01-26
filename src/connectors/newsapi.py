"""
NewsAPI connector for fetching news articles.

NewsAPI (https://newsapi.org) provides access to breaking news headlines
and articles from over 80,000 sources.

Free tier limitations:
- 100 requests per day
- Articles up to 1 month old
- No access to /everything endpoint for some sources
"""

from datetime import datetime
from typing import Optional

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from src.config.settings import settings
from src.connectors.base import NewsConnector, RawArticle


class NewsAPIConnector(NewsConnector):
    """
    Connector for NewsAPI.

    Usage:
        connector = NewsAPIConnector()
        articles = connector.fetch_articles(
            query="inteligencia artificial",
            language="es",
            max_results=50
        )
    """

    # Category mapping (NewsAPI categories)
    VALID_CATEGORIES = [
        "business", "entertainment", "general", "health",
        "science", "sports", "technology"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI connector.

        Args:
            api_key: NewsAPI key (default from settings).
        """
        self._api_key = api_key or settings.newsapi_key

        if not self._api_key:
            raise ValueError(
                "NewsAPI key not configured. "
                "Set NEWSAPI_KEY in your .env file."
            )

        self._client = NewsApiClient(api_key=self._api_key)

    @property
    def source_name(self) -> str:
        return "NewsAPI"

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
        Fetch articles from NewsAPI.

        Uses /everything endpoint for search queries,
        /top-headlines for category/country filtering.
        """
        articles = []

        try:
            if query:
                # Use /everything endpoint for search
                response = self._fetch_everything(
                    query=query,
                    language=language,
                    from_date=from_date,
                    to_date=to_date,
                    max_results=max_results
                )
            else:
                # Use /top-headlines for browsing
                response = self._fetch_top_headlines(
                    country=country,
                    category=category,
                    max_results=max_results
                )

            # Parse response
            for item in response.get("articles", []):
                article = self._parse_article(item, language, country, category)
                if article and article.has_content():
                    articles.append(article)

        except NewsAPIException as e:
            raise RuntimeError(f"NewsAPI error: {e}") from e

        return articles[:max_results]

    def _fetch_everything(
        self,
        query: str,
        language: Optional[str],
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        max_results: int
    ) -> dict:
        """Fetch from /everything endpoint."""
        params = {
            "q": query,
            "sort_by": "publishedAt",
            "page_size": min(max_results, 100)  # API limit
        }

        if language:
            params["language"] = language
        if from_date:
            params["from_param"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")

        return self._client.get_everything(**params)

    def _fetch_top_headlines(
        self,
        country: Optional[str],
        category: Optional[str],
        max_results: int
    ) -> dict:
        """Fetch from /top-headlines endpoint."""
        params = {
            "page_size": min(max_results, 100)  # API limit
        }

        if country:
            params["country"] = country
        if category and category.lower() in self.VALID_CATEGORIES:
            params["category"] = category.lower()

        # NewsAPI requires at least one of: country, category, sources, or q
        if not any([country, category]):
            params["language"] = "es"  # Default to Spanish

        return self._client.get_top_headlines(**params)

    def _parse_article(
        self,
        item: dict,
        language: Optional[str],
        country: Optional[str],
        category: Optional[str]
    ) -> Optional[RawArticle]:
        """Parse a NewsAPI article item into RawArticle."""
        title = item.get("title")
        if not title or title == "[Removed]":
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
            author=item.get("author"),
            image_url=item.get("urlToImage"),
            published_at=published_at,
            language=language,
            country=country,
            category=category,
            external_id=item.get("url"),  # URL as unique ID
            extra={
                "source_id": source.get("id"),
            }
        )

    def get_sources(
        self,
        language: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None
    ) -> list[dict]:
        """
        Get available news sources.

        Returns:
            List of source dictionaries with id, name, description, etc.
        """
        params = {}
        if language:
            params["language"] = language
        if country:
            params["country"] = country
        if category and category.lower() in self.VALID_CATEGORIES:
            params["category"] = category.lower()

        response = self._client.get_sources(**params)
        return response.get("sources", [])

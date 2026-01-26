"""
Article filtering utilities.

Provides filters for:
- Language
- Country/Region
- Time range
- Category
- Content quality
"""

from datetime import datetime, timedelta
from typing import Optional, Callable
import structlog

from src.connectors.base import RawArticle

logger = structlog.get_logger()


class ArticleFilter:
    """
    Filter articles based on various criteria.

    Usage:
        filter = ArticleFilter()
        filter.add_language_filter("es")
        filter.add_date_filter(days=7)

        filtered = filter.apply(articles)
    """

    def __init__(self):
        """Initialize empty filter chain."""
        self._filters: list[Callable[[RawArticle], bool]] = []

    def add_filter(self, filter_func: Callable[[RawArticle], bool]) -> "ArticleFilter":
        """
        Add a custom filter function.

        Args:
            filter_func: Function that takes RawArticle and returns True to keep.

        Returns:
            Self for method chaining.
        """
        self._filters.append(filter_func)
        return self

    def add_language_filter(
        self,
        languages: str | list[str]
    ) -> "ArticleFilter":
        """
        Filter by language code(s).

        Args:
            languages: Language code or list of codes (e.g., 'es', ['es', 'en']).
        """
        if isinstance(languages, str):
            languages = [languages]

        languages_lower = [lang.lower() for lang in languages]

        def filter_func(article: RawArticle) -> bool:
            if article.language is None:
                return True  # Keep if no language specified
            return article.language.lower() in languages_lower

        self._filters.append(filter_func)
        logger.info("Added language filter", languages=languages)
        return self

    def add_country_filter(
        self,
        countries: str | list[str]
    ) -> "ArticleFilter":
        """
        Filter by country code(s).

        Args:
            countries: Country code or list of codes (e.g., 'co', ['co', 'mx']).
        """
        if isinstance(countries, str):
            countries = [countries]

        countries_lower = [c.lower() for c in countries]

        def filter_func(article: RawArticle) -> bool:
            if article.country is None:
                return True  # Keep if no country specified
            return article.country.lower() in countries_lower

        self._filters.append(filter_func)
        logger.info("Added country filter", countries=countries)
        return self

    def add_category_filter(
        self,
        categories: str | list[str]
    ) -> "ArticleFilter":
        """
        Filter by category.

        Args:
            categories: Category or list of categories.
        """
        if isinstance(categories, str):
            categories = [categories]

        categories_lower = [c.lower() for c in categories]

        def filter_func(article: RawArticle) -> bool:
            if article.category is None:
                return True  # Keep if no category specified
            return article.category.lower() in categories_lower

        self._filters.append(filter_func)
        logger.info("Added category filter", categories=categories)
        return self

    def add_date_filter(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> "ArticleFilter":
        """
        Filter by publication date.

        Args:
            from_date: Minimum publication date.
            to_date: Maximum publication date.
            days: Alternative: filter articles from last N days.
        """
        if days is not None:
            from_date = datetime.now() - timedelta(days=days)

        def filter_func(article: RawArticle) -> bool:
            if article.published_at is None:
                return True  # Keep if no date

            pub_date = article.published_at

            # Make timezone-naive for comparison if needed
            if pub_date.tzinfo is not None:
                pub_date = pub_date.replace(tzinfo=None)

            if from_date:
                from_dt = from_date
                if from_dt.tzinfo is not None:
                    from_dt = from_dt.replace(tzinfo=None)
                if pub_date < from_dt:
                    return False

            if to_date:
                to_dt = to_date
                if to_dt.tzinfo is not None:
                    to_dt = to_dt.replace(tzinfo=None)
                if pub_date > to_dt:
                    return False

            return True

        self._filters.append(filter_func)
        logger.info("Added date filter", from_date=from_date, to_date=to_date, days=days)
        return self

    def add_content_filter(
        self,
        min_title_length: int = 10,
        min_content_length: int = 50,
        require_content: bool = False
    ) -> "ArticleFilter":
        """
        Filter by content quality.

        Args:
            min_title_length: Minimum title length in characters.
            min_content_length: Minimum content/description length.
            require_content: Require non-empty content field.
        """
        def filter_func(article: RawArticle) -> bool:
            # Check title length
            if len(article.title) < min_title_length:
                return False

            # Check content
            content_text = article.content or article.description or ""
            if len(content_text) < min_content_length:
                return False

            if require_content and not article.content:
                return False

            return True

        self._filters.append(filter_func)
        logger.info(
            "Added content filter",
            min_title=min_title_length,
            min_content=min_content_length
        )
        return self

    def add_source_filter(
        self,
        sources: list[str],
        exclude: bool = False
    ) -> "ArticleFilter":
        """
        Filter by source name.

        Args:
            sources: List of source names.
            exclude: If True, exclude these sources. If False, only include.
        """
        sources_lower = [s.lower() for s in sources]

        def filter_func(article: RawArticle) -> bool:
            source = article.source_name.lower()
            is_in_list = any(s in source for s in sources_lower)
            return not is_in_list if exclude else is_in_list

        self._filters.append(filter_func)
        action = "exclude" if exclude else "include"
        logger.info(f"Added source filter ({action})", sources=sources)
        return self

    def add_keyword_filter(
        self,
        keywords: list[str],
        require_all: bool = False
    ) -> "ArticleFilter":
        """
        Filter articles containing specific keywords.

        Args:
            keywords: List of keywords to search for.
            require_all: If True, require all keywords. If False, require any.
        """
        keywords_lower = [k.lower() for k in keywords]

        def filter_func(article: RawArticle) -> bool:
            text = article.get_text_for_embedding().lower()

            if require_all:
                return all(kw in text for kw in keywords_lower)
            else:
                return any(kw in text for kw in keywords_lower)

        self._filters.append(filter_func)
        mode = "all" if require_all else "any"
        logger.info(f"Added keyword filter ({mode})", keywords=keywords)
        return self

    def apply(self, articles: list[RawArticle]) -> list[RawArticle]:
        """
        Apply all filters to a list of articles.

        Args:
            articles: List of articles to filter.

        Returns:
            Filtered list of articles.
        """
        original_count = len(articles)

        filtered = articles
        for filter_func in self._filters:
            filtered = [a for a in filtered if filter_func(a)]

        logger.info(
            "Filters applied",
            original=original_count,
            filtered=len(filtered),
            removed=original_count - len(filtered)
        )

        return filtered

    def clear(self) -> "ArticleFilter":
        """Clear all filters."""
        self._filters = []
        return self


def filter_articles(
    articles: list[RawArticle],
    language: Optional[str] = None,
    country: Optional[str] = None,
    category: Optional[str] = None,
    days: Optional[int] = None,
    min_content_length: int = 50
) -> list[RawArticle]:
    """
    Convenience function to filter articles.

    Args:
        articles: List of articles to filter.
        language: Filter by language code.
        country: Filter by country code.
        category: Filter by category.
        days: Filter by last N days.
        min_content_length: Minimum content length.

    Returns:
        Filtered list of articles.
    """
    filter_chain = ArticleFilter()

    if language:
        filter_chain.add_language_filter(language)
    if country:
        filter_chain.add_country_filter(country)
    if category:
        filter_chain.add_category_filter(category)
    if days:
        filter_chain.add_date_filter(days=days)

    filter_chain.add_content_filter(min_content_length=min_content_length)

    return filter_chain.apply(articles)

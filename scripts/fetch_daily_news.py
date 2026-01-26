#!/usr/bin/env python
"""
Daily news fetcher - Fetches articles from multiple sources.

This script is designed to run once per day (via cron or scheduler)
to populate the database with diverse news content.

TWO MODES:
1. CATEGORY MODE (default): Fetches top headlines by category (more diverse)
   - Uses /top-headlines endpoint which doesn't require a query
   - Best for getting a broad range of news across topics

2. QUERY MODE: Searches for specific terms (more targeted)
   - Uses /everything endpoint with search queries
   - Best for finding specific topics or events

API Usage Strategy (optimized for free tiers):
- NewsAPI: 7 categories × 100 articles = ~700 articles (7 requests)
- GNews: 9 categories × 10 articles = ~90 articles (9 requests)
Total: 16 requests, ~790 articles per day

Available Categories:
- NewsAPI: general, business, technology, science, health, entertainment, sports
- GNews: general, business, technology, science, health, entertainment, sports, world, nation

Usage:
    # Category mode (recommended for daily feeding)
    poetry run python scripts/fetch_daily_news.py --dry-run
    poetry run python scripts/fetch_daily_news.py

    # Query mode (for specific searches)
    poetry run python scripts/fetch_daily_news.py --mode query --dry-run
    poetry run python scripts/fetch_daily_news.py --mode query

    # Custom categories (Spanish names are mapped automatically)
    poetry run python scripts/fetch_daily_news.py --categories economia,tecnologia,deportes

    # Skip one source
    poetry run python scripts/fetch_daily_news.py --skip-gnews
    poetry run python scripts/fetch_daily_news.py --skip-newsapi

    # Different country
    poetry run python scripts/fetch_daily_news.py --country us
"""

import argparse
import sys
import time
from datetime import datetime

import structlog

from src.agent.graph import run_journalist_agent
from src.config.settings import settings
from src.connectors.newsapi import NewsAPIConnector
from src.connectors.gnews import GNewsConnector
from src.storage.database import get_db
from src.storage.repositories.article import ArticleRepository
from src.processing.embeddings import get_embedding_generator

logger = structlog.get_logger()

# NewsAPI categories (7 total)
NEWSAPI_CATEGORIES = [
    "general",       # Noticias generales
    "business",      # Economía y negocios
    "technology",    # Tecnología
    "science",       # Ciencia
    "health",        # Salud
    "entertainment", # Entretenimiento
    "sports",        # Deportes
]

# GNews categories (same as NewsAPI)
GNEWS_CATEGORIES = [
    "general",
    "business",
    "technology",
    "science",
    "health",
    "entertainment",
    "sports",
    "world",        # GNews has world
    "nation",       # GNews has nation
]

# Queries for query mode (GNews)
GNEWS_QUERIES = [
    "Colombia noticias actualidad",
    "economía negocios latinoamérica",
    "tecnología inteligencia artificial",
    "política gobierno reformas",
    "deportes fútbol",
    "salud medicina",
    "entretenimiento cultura",
    "Petro gobierno",
    "elecciones 2026",
    "medio ambiente clima",
]

# Map Spanish names to API categories
CATEGORY_MAP = {
    "general": "general",
    "politica": "general",
    "economia": "business",
    "negocios": "business",
    "tecnologia": "technology",
    "ciencia": "science",
    "salud": "health",
    "entretenimiento": "entertainment",
    "deportes": "sports",
    "mundo": "world",
    "nacional": "nation",
}


def fetch_newsapi_category(category: str, country: str = "co", max_articles: int = 100) -> dict:
    """
    Fetch articles from NewsAPI by category.

    Note: NewsAPI free tier has limited country coverage (0 sources for CO).
    Uses /everything with category as query for better coverage.

    Args:
        category: NewsAPI category (general, business, technology, etc.)
        country: Country code (used for metadata, not filtering).
        max_articles: Max articles to fetch.

    Returns:
        Dict with fetched articles and stats.
    """
    logger.info(f"[NewsAPI] Fetching category: {category}", language="es", max_articles=max_articles)

    try:
        connector = NewsAPIConnector()
        # Use /everything with category as query (better coverage than /top-headlines for non-US)
        # Map category to Spanish search terms
        category_queries = {
            "general": "noticias actualidad",
            "business": "economía negocios finanzas",
            "technology": "tecnología innovación digital",
            "science": "ciencia investigación",
            "health": "salud medicina",
            "entertainment": "entretenimiento cultura espectáculos",
            "sports": "deportes fútbol",
        }
        query = category_queries.get(category, category)

        articles = connector.fetch_articles(
            query=query,  # Use query = /everything endpoint
            category=None,
            country=None,
            language=settings.default_language,
            max_results=max_articles,
        )

        # Tag articles with category and country
        for article in articles:
            article.category = category
            article.country = country

        return {
            "source": "NewsAPI",
            "type": "category",
            "category": category,
            "articles": articles,
            "fetched": len(articles),
            "errors": [],
        }

    except Exception as e:
        logger.error(f"[NewsAPI] Error fetching category {category}", error=str(e))
        return {
            "source": "NewsAPI",
            "type": "category",
            "category": category,
            "articles": [],
            "fetched": 0,
            "errors": [str(e)],
        }


def fetch_gnews_category(category: str, country: str = "co", max_articles: int = 10) -> dict:
    """
    Fetch articles from GNews by category (uses /top-headlines).

    Args:
        category: GNews category.
        country: Country code.
        max_articles: Max articles (10 max on free tier).

    Returns:
        Dict with fetched articles and stats.
    """
    logger.info(f"[GNews] Fetching category: {category}", country=country, max_articles=max_articles)

    try:
        connector = GNewsConnector()
        articles = connector.fetch_articles(
            query=None,  # No query = use /top-headlines
            category=category,
            country=country,
            language=settings.default_language,
            max_results=min(max_articles, 10),  # GNews free tier limit
        )

        return {
            "source": "GNews",
            "type": "category",
            "category": category,
            "articles": articles,
            "fetched": len(articles),
            "errors": [],
        }

    except Exception as e:
        logger.error(f"[GNews] Error fetching category {category}", error=str(e))
        return {
            "source": "GNews",
            "type": "category",
            "category": category,
            "articles": [],
            "fetched": 0,
            "errors": [str(e)],
        }


def fetch_by_query(query: str, sources: list = None, max_articles: int = 100) -> dict:
    """
    Fetch articles by search query using the agent (full pipeline).

    Args:
        query: Search term.
        sources: List of sources to use.
        max_articles: Max articles to fetch.

    Returns:
        Agent result dict with stats.
    """
    if sources is None:
        sources = ["newsapi", "gnews"]

    logger.info(f"[Query] Fetching: {query}", sources=sources, max_articles=max_articles)

    try:
        result = run_journalist_agent(
            query=query,
            max_articles=max_articles,
            sources=sources,
            use_persistence=True,
        )

        fetched = len(result.get("raw_articles", []))
        saved = len(result.get("saved_article_ids", []))

        return {
            "source": "Agent",
            "type": "query",
            "query": query,
            "fetched": fetched,
            "saved": saved,
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"[Query] Error: {query}", error=str(e))
        return {
            "source": "Agent",
            "type": "query",
            "query": query,
            "fetched": 0,
            "saved": 0,
            "errors": [str(e)],
        }


def process_and_save_articles(all_articles: list, skip_existing: bool = True) -> dict:
    """
    Save fetched articles to DB with embeddings.

    This function only saves articles - it doesn't generate summaries/headlines.
    Use the agent or a separate processing script for that.

    Args:
        all_articles: List of RawArticle objects.
        skip_existing: Skip articles that already exist in DB.

    Returns:
        Dict with save stats.
    """
    if not all_articles:
        return {"saved": 0, "skipped": 0, "duplicates": 0, "errors": []}

    logger.info(f"Saving {len(all_articles)} articles to database")

    try:
        embedder = get_embedding_generator()

        saved_count = 0
        skipped_count = 0
        duplicate_count = 0
        error_count = 0

        # First, deduplicate by URL within the fetched batch
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.source_url:
                if article.source_url in seen_urls:
                    duplicate_count += 1
                    continue
                seen_urls.add(article.source_url)
            unique_articles.append(article)

        logger.info(f"Deduplicated: {len(unique_articles)} unique, {duplicate_count} duplicates")

        with get_db() as db:
            article_repo = ArticleRepository(db)

            # Check existing URLs in DB
            all_urls = [a.source_url for a in unique_articles if a.source_url]
            existing_urls = set(article_repo.get_existing_urls(all_urls))
            logger.info(f"Found {len(existing_urls)} existing articles in DB")

            # Filter to new articles only
            new_articles = [
                a for a in unique_articles
                if not a.source_url or a.source_url not in existing_urls
            ]

            if not new_articles:
                logger.info("All articles already exist in database")
                return {
                    "saved": 0,
                    "skipped": len(existing_urls),
                    "duplicates": duplicate_count,
                    "errors": [],
                }

            skipped_count = len(existing_urls)
            logger.info(f"Processing {len(new_articles)} new articles")

            # Generate embeddings for new articles only
            texts = [a.get_text_for_embedding() for a in new_articles]
            embeddings = embedder.embed_texts(texts)

            # Save articles one by one to handle individual failures
            for i, raw_article in enumerate(new_articles):
                try:
                    article = article_repo.create(raw_article, embeddings[i])
                    db.flush()  # Flush to detect errors early
                    saved_count += 1
                except Exception as e:
                    db.rollback()  # Rollback the failed insert
                    logger.warning(f"Error saving article: {str(e)[:100]}")
                    error_count += 1
                    continue

            db.commit()

        return {
            "saved": saved_count,
            "skipped": skipped_count,
            "duplicates": duplicate_count,
            "errors": [] if error_count == 0 else [f"{error_count} articles failed to save"],
        }

    except Exception as e:
        logger.error(f"Error saving articles", error=str(e))
        return {"saved": 0, "skipped": 0, "duplicates": 0, "errors": [str(e)]}


def fetch_from_newsapi(query: str, max_articles: int = 100) -> dict:
    """Fetch from NewsAPI only (supports up to 100 articles per request)."""
    logger.info(f"[NewsAPI] Fetching: {query}", max_articles=max_articles)

    try:
        result = run_journalist_agent(
            query=query,
            max_articles=max_articles,
            sources=["newsapi"],  # Only NewsAPI
            use_persistence=True,
        )

        fetched = len(result.get("raw_articles", []))
        saved = len(result.get("saved_article_ids", []))

        return {
            "source": "NewsAPI",
            "query": query,
            "fetched": fetched,
            "saved": saved,
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"[NewsAPI] Error: {query}", error=str(e))
        return {"source": "NewsAPI", "query": query, "fetched": 0, "saved": 0, "errors": [str(e)]}


def fetch_from_gnews(query: str, max_articles: int = 10) -> dict:
    """Fetch from GNews only (max 10 articles per request on free tier)."""
    logger.info(f"[GNews] Fetching: {query}", max_articles=max_articles)

    try:
        result = run_journalist_agent(
            query=query,
            max_articles=max_articles,
            sources=["gnews"],  # Only GNews
            use_persistence=True,
        )

        fetched = len(result.get("raw_articles", []))
        saved = len(result.get("saved_article_ids", []))

        return {
            "source": "GNews",
            "query": query,
            "fetched": fetched,
            "saved": saved,
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"[GNews] Error: {query}", error=str(e))
        return {"source": "GNews", "query": query, "fetched": 0, "saved": 0, "errors": [str(e)]}


def main():
    parser = argparse.ArgumentParser(description="Fetch daily news from APIs")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["category", "query"],
        default="category",
        help="Fetching mode: 'category' (top headlines) or 'query' (search). Default: category",
    )
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated categories for category mode (e.g., business,technology,sports)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Comma-separated queries for query mode",
    )
    parser.add_argument(
        "--newsapi-max",
        type=int,
        default=100,
        help="Max articles per NewsAPI request (default: 100)",
    )
    parser.add_argument(
        "--gnews-max",
        type=int,
        default=10,
        help="Max articles per GNews request (default: 10, max for free tier)",
    )
    parser.add_argument(
        "--skip-newsapi",
        action="store_true",
        help="Skip NewsAPI fetching",
    )
    parser.add_argument(
        "--skip-gnews",
        action="store_true",
        help="Skip GNews fetching",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="co",
        help="Country code for headlines (default: co)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DAILY NEWS FETCHER")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Country: {args.country}")
    print()

    if args.mode == "category":
        # Category mode: fetch by categories
        newsapi_categories = NEWSAPI_CATEGORIES.copy()
        gnews_categories = GNEWS_CATEGORIES.copy()

        # Override with user categories
        if args.categories:
            user_cats = [c.strip().lower() for c in args.categories.split(",")]
            # Map Spanish names to API names
            mapped_cats = [CATEGORY_MAP.get(c, c) for c in user_cats]
            newsapi_categories = [c for c in mapped_cats if c in NEWSAPI_CATEGORIES]
            gnews_categories = [c for c in mapped_cats if c in GNEWS_CATEGORIES]

        if args.skip_newsapi:
            newsapi_categories = []
        if args.skip_gnews:
            gnews_categories = []

        print("NewsAPI Categories:")
        print(f"  Categories: {len(newsapi_categories)}")
        print(f"  Articles per category: {args.newsapi_max}")
        print(f"  Estimated articles: {len(newsapi_categories) * args.newsapi_max}")
        for i, cat in enumerate(newsapi_categories, 1):
            print(f"    {i}. {cat}")
        print()
        print("GNews Categories:")
        print(f"  Categories: {len(gnews_categories)}")
        print(f"  Articles per category: {args.gnews_max}")
        print(f"  Estimated articles: {len(gnews_categories) * args.gnews_max}")
        for i, cat in enumerate(gnews_categories, 1):
            print(f"    {i}. {cat}")
        print()

        total_requests = len(newsapi_categories) + len(gnews_categories)
        total_estimated = (
            len(newsapi_categories) * args.newsapi_max +
            len(gnews_categories) * args.gnews_max
        )

        print("-" * 60)
        print(f"TOTAL API REQUESTS: {total_requests}")
        print(f"TOTAL ESTIMATED ARTICLES: {total_estimated}")
        print("=" * 60)

        if args.dry_run:
            print("\n[DRY RUN] No articles will be fetched.")
            return 0

        print("\nStarting category-based fetch...\n")

        all_articles = []
        results = []
        total_errors = 0

        # Fetch from NewsAPI categories
        if newsapi_categories:
            print("\n--- NewsAPI Categories ---")
            for cat in newsapi_categories:
                result = fetch_newsapi_category(cat, args.country, args.newsapi_max)
                results.append(result)
                all_articles.extend(result["articles"])
                total_errors += len(result["errors"])
                status = "✓" if not result["errors"] else "✗"
                print(f"  {status} {cat}: {result['fetched']} articles")

        # Fetch from GNews categories (with delay to avoid rate limiting)
        if gnews_categories:
            print("\n--- GNews Categories ---")
            for i, cat in enumerate(gnews_categories):
                if i > 0:
                    time.sleep(1)  # 1 second delay between requests
                result = fetch_gnews_category(cat, args.country, args.gnews_max)
                results.append(result)
                all_articles.extend(result["articles"])
                total_errors += len(result["errors"])
                status = "✓" if not result["errors"] else "✗"
                print(f"  {status} {cat}: {result['fetched']} articles")

        # Process and save all articles
        print(f"\n--- Processing {len(all_articles)} articles ---")
        save_result = process_and_save_articles(all_articles)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        newsapi_total = sum(r["fetched"] for r in results if r["source"] == "NewsAPI")
        gnews_total = sum(r["fetched"] for r in results if r["source"] == "GNews")

        print(f"NewsAPI: {newsapi_total} fetched")
        print(f"GNews:   {gnews_total} fetched")
        print("-" * 60)
        print(f"TOTAL FETCHED:      {len(all_articles)}")
        print(f"DUPLICATES REMOVED: {save_result.get('duplicates', 0)}")
        print(f"ALREADY IN DB:      {save_result['skipped']}")
        print(f"SAVED TO DB:        {save_result['saved']}")

        if total_errors > 0:
            print(f"\nFetch Errors: {total_errors}")
            for r in results:
                if r["errors"]:
                    print(f"  [{r['source']}] {r['category']}: {r['errors'][0][:60]}")

        print("=" * 60)
        return 0 if total_errors == 0 else 1

    else:
        # Query mode: search by queries
        newsapi_queries = GNEWS_QUERIES[:7]  # Use first 7 for NewsAPI
        gnews_queries = GNEWS_QUERIES.copy()

        if args.queries:
            user_queries = [q.strip() for q in args.queries.split(",")]
            newsapi_queries = user_queries
            gnews_queries = user_queries

        if args.skip_newsapi:
            newsapi_queries = []
        if args.skip_gnews:
            gnews_queries = []

        print("NewsAPI Queries:")
        print(f"  Queries: {len(newsapi_queries)}")
        print(f"  Articles per query: {args.newsapi_max}")
        print(f"  Estimated articles: {len(newsapi_queries) * args.newsapi_max}")
        for i, q in enumerate(newsapi_queries, 1):
            print(f"    {i}. {q}")
        print()
        print("GNews Queries:")
        print(f"  Queries: {len(gnews_queries)}")
        print(f"  Articles per query: {args.gnews_max}")
        print(f"  Estimated articles: {len(gnews_queries) * args.gnews_max}")
        for i, q in enumerate(gnews_queries, 1):
            print(f"    {i}. {q}")
        print()

        total_requests = len(newsapi_queries) + len(gnews_queries)
        total_estimated = (
            len(newsapi_queries) * args.newsapi_max +
            len(gnews_queries) * args.gnews_max
        )

        print("-" * 60)
        print(f"TOTAL API REQUESTS: {total_requests}")
        print(f"TOTAL ESTIMATED ARTICLES: {total_estimated}")
        print("=" * 60)

        if args.dry_run:
            print("\n[DRY RUN] No articles will be fetched.")
            return 0

        print("\nStarting query-based fetch...\n")

        results = []
        total_fetched = 0
        total_saved = 0
        total_errors = 0

        # Fetch from NewsAPI
        if newsapi_queries:
            print("\n--- NewsAPI Queries ---")
            for query in newsapi_queries:
                result = fetch_from_newsapi(query, args.newsapi_max)
                results.append(result)
                total_fetched += result["fetched"]
                total_saved += result["saved"]
                total_errors += len(result["errors"])
                print(f"  ✓ {query}: {result['fetched']} fetched, {result['saved']} saved")

        # Fetch from GNews
        if gnews_queries:
            print("\n--- GNews Queries ---")
            for query in gnews_queries:
                result = fetch_from_gnews(query, args.gnews_max)
                results.append(result)
                total_fetched += result["fetched"]
                total_saved += result["saved"]
                total_errors += len(result["errors"])
                print(f"  ✓ {query}: {result['fetched']} fetched, {result['saved']} saved")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        newsapi_total = sum(r["fetched"] for r in results if r["source"] == "NewsAPI")
        gnews_total = sum(r["fetched"] for r in results if r["source"] == "GNews")
        newsapi_saved = sum(r["saved"] for r in results if r["source"] == "NewsAPI")
        gnews_saved = sum(r["saved"] for r in results if r["source"] == "GNews")

        print(f"NewsAPI: {newsapi_total} fetched, {newsapi_saved} saved")
        print(f"GNews:   {gnews_total} fetched, {gnews_saved} saved")
        print("-" * 60)
        print(f"TOTAL:   {total_fetched} fetched, {total_saved} saved")

        if total_errors > 0:
            print(f"\nErrors: {total_errors}")
            for r in results:
                if r["errors"]:
                    print(f"  [{r['source']}] {r['query']}: {r['errors'][0][:60]}")

        print("=" * 60)
        return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

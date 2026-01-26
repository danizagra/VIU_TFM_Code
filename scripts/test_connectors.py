#!/usr/bin/env python3
"""
Script to test news connectors.

Usage:
    poetry run python scripts/test_connectors.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings


def test_rss_connector():
    """Test RSS feed connector."""
    print("=" * 60)
    print("Testing RSS Connector")
    print("=" * 60)

    from src.connectors.rss import RSSConnector

    # Test with El País (reliable RSS feed)
    feed_url = "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"

    try:
        connector = RSSConnector(
            feed_url=feed_url,
            source_name="El País",
            language="es",
            country="es"
        )

        print(f"Feed URL: {feed_url}")
        print(f"Source: {connector.source_name}")
        print("\nFetching articles...")

        articles = connector.fetch_articles(max_results=5)

        print(f"\n✓ Fetched {len(articles)} articles")

        if articles:
            print("\nSample articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n  {i}. {article.title[:60]}...")
                print(f"     Source: {article.source_name}")
                print(f"     Date: {article.published_at}")
                if article.source_url:
                    print(f"     URL: {article.source_url[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_newsapi_connector():
    """Test NewsAPI connector."""
    print("\n" + "=" * 60)
    print("Testing NewsAPI Connector")
    print("=" * 60)

    if not settings.newsapi_key:
        print("⚠ NewsAPI key not configured, skipping...")
        return None

    try:
        from src.connectors.newsapi import NewsAPIConnector

        connector = NewsAPIConnector()
        print(f"Source: {connector.source_name}")
        # Nota: Plan gratis de NewsAPI no soporta /top-headlines para países != US
        # Usamos /everything con query en español
        print("\nSearching 'Colombia' in Spanish (via /everything)...")

        articles = connector.fetch_articles(
            query="Colombia",
            language="es",
            max_results=5
        )

        print(f"\n✓ Fetched {len(articles)} articles")

        if articles:
            print("\nSample articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n  {i}. {article.title[:60]}...")
                print(f"     Source: {article.source_name}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_gnews_connector():
    """Test GNews connector."""
    print("\n" + "=" * 60)
    print("Testing GNews Connector")
    print("=" * 60)

    if not settings.gnews_api_key:
        print("⚠ GNews API key not configured, skipping...")
        return None

    try:
        from src.connectors.gnews import GNewsConnector

        connector = GNewsConnector()
        print(f"Source: {connector.source_name}")
        print("\nSearching for 'tecnología' in Spanish...")

        articles = connector.fetch_articles(
            query="tecnología",
            language="es",
            max_results=5
        )

        print(f"\n✓ Fetched {len(articles)} articles")

        if articles:
            print("\nSample articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n  {i}. {article.title[:60]}...")
                print(f"     Source: {article.source_name}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_aggregator():
    """Test news aggregator with all sources."""
    print("\n" + "=" * 60)
    print("Testing News Aggregator")
    print("=" * 60)

    from src.connectors import create_default_aggregator

    try:
        aggregator = create_default_aggregator(
            include_newsapi=bool(settings.newsapi_key),
            include_gnews=bool(settings.gnews_api_key),
            include_colombian_rss=True,
            include_spanish_rss=False
        )

        print(f"Connectors loaded: {len(aggregator.connectors)}")
        for conn in aggregator.connectors:
            print(f"  - {conn.source_name}")

        print("\nFetching articles from all sources...")

        articles = aggregator.fetch_all(
            language="es",
            max_results=10
        )

        print(f"\n✓ Aggregated {len(articles)} unique articles")

        if articles:
            # Group by source
            sources = {}
            for article in articles:
                src = article.source_name
                sources[src] = sources.get(src, 0) + 1

            print("\nArticles by source:")
            for src, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"  - {src}: {count}")

            print("\nNewest articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n  {i}. {article.title[:60]}...")
                print(f"     Source: {article.source_name}")
                print(f"     Date: {article.published_at}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all connector tests."""
    print("\n🔬 News Connectors Test Suite\n")

    results = []

    # Test 1: RSS (always available)
    results.append(("RSS Connector", test_rss_connector()))

    # Test 2: NewsAPI (optional)
    result = test_newsapi_connector()
    if result is not None:
        results.append(("NewsAPI Connector", result))

    # Test 3: GNews (optional)
    result = test_gnews_connector()
    if result is not None:
        results.append(("GNews Connector", result))

    # Test 4: Aggregator
    results.append(("News Aggregator", test_aggregator()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

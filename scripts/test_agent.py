#!/usr/bin/env python3
"""
Script to test the journalist agent workflow.

Runs the complete LangGraph pipeline:
fetch → check_existing → filter → embed → load_similar → cluster → deduplicate → generate → quality → save

Usage:
    poetry run python scripts/test_agent.py
    poetry run python scripts/test_agent.py --query "economía"
    poetry run python scripts/test_agent.py --max-articles 10
    poetry run python scripts/test_agent.py --sources rss,newsapi  # Multiple sources
    poetry run python scripts/test_agent.py --sources all          # All sources
    poetry run python scripts/test_agent.py --no-persistence       # Skip DB operations

Sources available:
    - rss: El Tiempo RSS feeds (default)
    - newsapi: NewsAPI (requires API key, uses /everything endpoint)
    - gnews: GNews API (requires API key)
    - all: All sources combined
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run the journalist agent."""
    parser = argparse.ArgumentParser(description="Run the journalist agent")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="",
        help="Optional search query for news",
    )
    parser.add_argument(
        "--max-articles", "-m",
        type=int,
        default=10,
        help="Maximum articles to fetch (default: 10)",
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot prompts for faster generation",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable database persistence (no save/load from DB)",
    )
    parser.add_argument(
        "--sources", "-s",
        type=str,
        default="rss",
        help="News sources to use, comma-separated. Options: rss, newsapi, gnews, all (default: rss)",
    )
    args = parser.parse_args()

    # Parse sources
    sources = [s.strip().lower() for s in args.sources.split(",")]

    print("\n🤖 Journalist Agent - Test Run\n")
    print("=" * 60)

    from src.agent.graph import run_journalist_agent

    # Run agent
    print(f"Query: {args.query or '(none)'}")
    print(f"Max articles: {args.max_articles}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Few-shot: {not args.no_few_shot}")
    print(f"Persistence: {not args.no_persistence}")
    print("=" * 60)
    print("\nStarting workflow...\n")

    final_state = run_journalist_agent(
        query=args.query,
        max_articles=args.max_articles,
        use_few_shot=not args.no_few_shot,
        use_persistence=not args.no_persistence,
        sources=sources,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Summary stats
    raw_count = len(final_state.get("raw_articles", []))
    filtered_count = len(final_state.get("filtered_articles", []))
    dedup_count = len(final_state.get("deduplicated_articles", []))
    processed_count = len(final_state.get("processed_articles", []))
    clusters = final_state.get("clusters", [])
    errors = final_state.get("errors", [])
    skipped_count = final_state.get("skipped_count", 0)
    saved_ids = final_state.get("saved_article_ids", [])
    session_id = final_state.get("session_id", None)

    print(f"\n📊 Pipeline Stats:")
    print(f"   Raw articles:         {raw_count}")
    print(f"   Skipped (in DB):      {skipped_count}")
    print(f"   After filtering:      {filtered_count}")
    print(f"   After deduplication:  {dedup_count}")
    print(f"   Processed articles:   {processed_count}")
    print(f"   Saved to DB:          {len(saved_ids)}")
    print(f"   Clusters found:       {len(clusters)}")
    print(f"   Errors:               {len(errors)}")
    if session_id:
        print(f"   Session ID:           {session_id}")

    # Show processed articles
    processed = final_state.get("processed_articles", [])
    if processed:
        print(f"\n📰 Processed Articles ({len(processed)}):")
        print("-" * 60)

        for i, article in enumerate(processed[:5]):  # Show first 5
            print(f"\n{i + 1}. {article.title[:60]}...")
            print(f"   Source: {article.source}")
            print(f"   Cluster: {article.cluster_id}")
            print(f"   Quality: {article.quality_score:.2f}")

            if article.summary:
                print(f"   Summary: {article.summary[:100]}...")

            if article.headlines:
                print(f"   Headlines:")
                for htype, headline in article.headlines.items():
                    if headline:
                        print(f"      - {htype}: {headline[:50]}...")

        if len(processed) > 5:
            print(f"\n   ... and {len(processed) - 5} more articles")

    # Show clusters
    if clusters:
        print(f"\n🗂️ Clusters ({len(clusters)}):")
        print("-" * 60)
        for cluster in clusters:
            print(f"   Cluster {cluster.cluster_id}: {len(cluster.article_indices)} articles")

    # Show errors
    if errors:
        print(f"\n⚠️ Errors ({len(errors)}):")
        print("-" * 60)
        for error in errors:
            print(f"   - {error}")

    # Quality summary
    quality_results = final_state.get("quality_results", {})
    if quality_results:
        avg_score = sum(r["overall_score"] for r in quality_results.values()) / len(quality_results)
        failed = final_state.get("failed_articles", [])
        print(f"\n✅ Quality Summary:")
        print(f"   Average score: {avg_score:.2f}")
        print(f"   Failed articles: {len(failed)}")

    # Duration
    start = final_state.get("start_time")
    end = final_state.get("end_time")
    if start and end:
        duration = (end - start).total_seconds()
        print(f"\n⏱️ Duration: {duration:.1f} seconds")

    # Export to JSON if requested
    if args.output:
        output_data = {
            "stats": {
                "raw_articles": raw_count,
                "filtered_articles": filtered_count,
                "deduplicated_articles": dedup_count,
                "processed_articles": processed_count,
                "clusters": len(clusters),
                "errors": len(errors),
            },
            "articles": [a.to_dict() for a in processed],
            "errors": errors,
        }

        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📁 Results exported to: {output_path}")

    print("\n" + "=" * 60)
    print("✓ Agent workflow complete!")
    print("=" * 60 + "\n")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Script to test the processing pipeline.

Tests embeddings, clustering, similarity, and deduplication
using real articles from news connectors.

Usage:
    poetry run python scripts/test_processing.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_embeddings():
    """Test embedding generation."""
    print("=" * 60)
    print("Testing Embedding Generation")
    print("=" * 60)

    from src.processing.embeddings import EmbeddingGenerator

    try:
        generator = EmbeddingGenerator()
        print(f"Model: {generator._model_name}")
        print("Loading model (first use)...")

        # Test single embedding
        text = "Colombia anuncia nuevo plan de infraestructura"
        embedding = generator.embed_single(text)

        print(f"\n✓ Single embedding generated")
        print(f"  Shape: {embedding.shape}")
        print(f"  Dimension: {generator.embedding_dimension}")

        # Test batch embedding
        texts = [
            "El gobierno colombiano presenta reformas económicas",
            "Nuevas medidas tributarias afectan a empresas",
            "Partido de fútbol termina en empate en Bogotá",
            "Científicos descubren nueva especie en la Amazonía"
        ]

        embeddings = generator.embed_texts(texts)
        print(f"\n✓ Batch embeddings generated")
        print(f"  Shape: {embeddings.shape}")

        return True, embeddings

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_similarity(embeddings):
    """Test similarity calculation."""
    print("\n" + "=" * 60)
    print("Testing Similarity Calculation")
    print("=" * 60)

    from src.processing.similarity import SimilarityCalculator

    try:
        calculator = SimilarityCalculator(
            similarity_threshold=0.5,
            duplicate_threshold=0.9
        )

        result = calculator.calculate_from_embeddings(embeddings)

        print(f"\n✓ Similarity matrix calculated")
        print(f"  Shape: {result.matrix.shape}")
        print(f"  Similar pairs (>0.5): {len(result.similar_pairs)}")
        print(f"  Duplicate pairs (>0.9): {len(result.duplicate_pairs)}")

        if result.similar_pairs:
            print("\n  Most similar pairs:")
            for i, j, score in result.similar_pairs[:3]:
                print(f"    Articles {i} & {j}: {score:.3f}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clustering(embeddings):
    """Test clustering."""
    print("\n" + "=" * 60)
    print("Testing Clustering")
    print("=" * 60)

    from src.processing.clustering import NewsClustering

    try:
        # For small datasets, use lower min_cluster_size
        clustering = NewsClustering(
            min_cluster_size=2,
            use_umap=False  # Too few samples for UMAP
        )

        result = clustering.fit_predict(embeddings, return_2d=False)

        print(f"\n✓ Clustering complete")
        print(f"  Clusters found: {result.n_clusters}")
        print(f"  Noise points: {result.n_noise}")
        print(f"  Cluster sizes: {result.cluster_sizes}")
        print(f"  Labels: {result.labels}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filters():
    """Test article filters."""
    print("\n" + "=" * 60)
    print("Testing Article Filters")
    print("=" * 60)

    from src.connectors.base import RawArticle
    from src.processing.filters import ArticleFilter
    from datetime import datetime, timedelta

    try:
        # Create test articles
        articles = [
            RawArticle(
                title="Noticia de Colombia sobre economía",
                source_name="El Tiempo",
                description="Descripción larga de la noticia económica",
                language="es",
                country="co",
                category="business",
                published_at=datetime.now() - timedelta(days=1)
            ),
            RawArticle(
                title="News from USA about technology",
                source_name="TechCrunch",
                description="Long description about tech news",
                language="en",
                country="us",
                category="technology",
                published_at=datetime.now() - timedelta(days=2)
            ),
            RawArticle(
                title="Short",  # Too short
                source_name="Unknown",
                description="x",  # Too short
                language="es",
                country="co"
            ),
            RawArticle(
                title="Noticia vieja de hace un mes",
                source_name="El Espectador",
                description="Descripción de noticia antigua",
                language="es",
                country="co",
                published_at=datetime.now() - timedelta(days=35)
            ),
        ]

        print(f"Original articles: {len(articles)}")

        # Test language filter
        filter1 = ArticleFilter().add_language_filter("es")
        spanish_only = filter1.apply(articles)
        print(f"\n✓ Spanish only: {len(spanish_only)} articles")

        # Test content filter
        filter2 = ArticleFilter().add_content_filter(
            min_title_length=10,
            min_content_length=20
        )
        quality = filter2.apply(articles)
        print(f"✓ Content quality filter: {len(quality)} articles")

        # Test date filter
        filter3 = ArticleFilter().add_date_filter(days=7)
        recent = filter3.apply(articles)
        print(f"✓ Last 7 days: {len(recent)} articles")

        # Combined filters
        filter_chain = (
            ArticleFilter()
            .add_language_filter("es")
            .add_content_filter(min_content_length=20)
            .add_date_filter(days=7)
        )
        combined = filter_chain.apply(articles)
        print(f"✓ Combined (es + quality + 7 days): {len(combined)} articles")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deduplication():
    """Test deduplication."""
    print("\n" + "=" * 60)
    print("Testing Deduplication")
    print("=" * 60)

    from src.connectors.base import RawArticle
    from src.processing.deduplication import ArticleDeduplicator

    try:
        # Create articles with some near-duplicates
        articles = [
            RawArticle(
                title="Colombia anuncia nuevo plan de infraestructura",
                source_name="El Tiempo",
                description="El gobierno presenta inversiones"
            ),
            RawArticle(
                title="Gobierno colombiano presenta plan de infraestructura",  # Similar
                source_name="El Espectador",
                description="Nuevas inversiones en carreteras y aeropuertos"
            ),
            RawArticle(
                title="Partido de fútbol termina en empate",
                source_name="ESPN",
                description="El encuentro entre los equipos finaliza sin goles"
            ),
            RawArticle(
                title="Científicos descubren nueva especie",
                source_name="Nature",
                description="Investigadores encuentran animal desconocido"
            ),
        ]

        # Generate embeddings for these articles
        from src.processing.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        article_embeddings = generator.embed_articles(articles)

        print(f"Articles: {len(articles)}")
        print(f"Embeddings shape: {article_embeddings.shape}")

        deduplicator = ArticleDeduplicator(
            duplicate_threshold=0.8,  # Lower threshold for testing
            selection_strategy="longest"
        )

        result = deduplicator.deduplicate(articles, article_embeddings)

        print(f"\n✓ Deduplication complete")
        print(f"  Original: {len(articles)}")
        print(f"  Unique: {len(result.unique_articles)}")
        print(f"  Removed: {result.duplicates_removed}")
        print(f"  Dedup rate: {result.deduplication_rate:.1%}")

        if result.duplicate_groups:
            print(f"  Duplicate groups: {len(result.duplicate_groups)}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test full processing pipeline with real articles."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline (with real articles)")
    print("=" * 60)

    try:
        from src.connectors import create_default_aggregator
        from src.processing import (
            EmbeddingGenerator,
            ArticleFilter,
            NewsClustering,
            ArticleDeduplicator
        )

        # 1. Fetch articles
        print("\n1. Fetching articles...")
        aggregator = create_default_aggregator(
            include_newsapi=False,  # Skip to avoid rate limits
            include_gnews=False,
            include_colombian_rss=True
        )

        articles = aggregator.fetch_all(max_results=20)
        print(f"   Fetched: {len(articles)} articles")

        if len(articles) < 5:
            print("   ⚠ Not enough articles for full pipeline test")
            return True  # Not a failure, just not enough data

        # 2. Filter
        print("\n2. Filtering articles...")
        filter_chain = (
            ArticleFilter()
            .add_content_filter(min_content_length=50)
        )
        filtered = filter_chain.apply(articles)
        print(f"   After filter: {len(filtered)} articles")

        # 3. Generate embeddings
        print("\n3. Generating embeddings...")
        generator = EmbeddingGenerator()
        embeddings = generator.embed_articles(filtered)
        print(f"   Embeddings shape: {embeddings.shape}")

        # 4. Deduplicate
        print("\n4. Deduplicating...")
        deduplicator = ArticleDeduplicator(duplicate_threshold=0.9)
        dedup_result = deduplicator.deduplicate(filtered, embeddings)
        print(f"   Unique: {len(dedup_result.unique_articles)}")
        print(f"   Removed: {dedup_result.duplicates_removed}")

        # 5. Cluster (if enough articles)
        if len(dedup_result.unique_articles) >= 5:
            print("\n5. Clustering...")
            # Re-generate embeddings for unique articles
            unique_embeddings = generator.embed_articles(dedup_result.unique_articles)

            clustering = NewsClustering(
                min_cluster_size=2,
                use_umap=len(unique_embeddings) > 10
            )
            cluster_result = clustering.fit_predict(unique_embeddings, return_2d=False)
            print(f"   Clusters: {cluster_result.n_clusters}")
            print(f"   Noise: {cluster_result.n_noise}")
        else:
            print("\n5. Clustering skipped (not enough articles)")

        print("\n✓ Full pipeline test complete!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all processing tests."""
    print("\n🔬 Processing Pipeline Test Suite\n")

    results = []

    # Test 1: Embeddings
    success, embeddings = test_embeddings()
    results.append(("Embeddings", success))

    if success and embeddings is not None:
        # Test 2: Similarity
        results.append(("Similarity", test_similarity(embeddings)))

        # Test 3: Clustering
        results.append(("Clustering", test_clustering(embeddings)))

        # Test 4: Deduplication
        results.append(("Deduplication", test_deduplication()))

    # Test 5: Filters (independent)
    results.append(("Filters", test_filters()))

    # Test 6: Full pipeline
    results.append(("Full Pipeline", test_full_pipeline()))

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

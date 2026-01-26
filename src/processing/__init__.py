"""
Text processing pipeline for news articles.

Includes:
- Embedding generation (SentenceTransformers / LM Studio)
- Clustering (UMAP + HDBSCAN)
- Similarity calculation
- Filtering
- Deduplication

Usage:
    from src.processing import (
        EmbeddingGenerator,
        NewsClustering,
        SimilarityCalculator,
        ArticleDeduplicator,
        ArticleFilter
    )

    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(texts)

    # Cluster articles
    clustering = NewsClustering()
    result = clustering.fit_predict(embeddings)

    # Deduplicate
    deduplicator = ArticleDeduplicator()
    unique = deduplicator.deduplicate(articles, embeddings)
"""

from src.processing.embeddings import (
    EmbeddingGenerator,
    LMStudioEmbeddingGenerator,
    get_embedding_generator,
)
from src.processing.clustering import (
    NewsClustering,
    ClusterResult,
    cluster_articles,
)
from src.processing.similarity import (
    SimilarityCalculator,
    SimilarityResult,
    calculate_similarity,
)
from src.processing.filters import (
    ArticleFilter,
    filter_articles,
)
from src.processing.deduplication import (
    ArticleDeduplicator,
    DeduplicationResult,
    deduplicate_articles,
)

__all__ = [
    # Embeddings
    "EmbeddingGenerator",
    "LMStudioEmbeddingGenerator",
    "get_embedding_generator",
    # Clustering
    "NewsClustering",
    "ClusterResult",
    "cluster_articles",
    # Similarity
    "SimilarityCalculator",
    "SimilarityResult",
    "calculate_similarity",
    # Filters
    "ArticleFilter",
    "filter_articles",
    # Deduplication
    "ArticleDeduplicator",
    "DeduplicationResult",
    "deduplicate_articles",
]

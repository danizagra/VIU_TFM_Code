"""
Similarity calculation for news articles.

Supports:
- Cosine similarity using embeddings
- TF-IDF similarity for text comparison
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

from src.config.settings import settings

logger = structlog.get_logger()


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""

    # Similarity matrix (n x n)
    matrix: np.ndarray

    # Pairs above threshold: [(idx1, idx2, score), ...]
    similar_pairs: list[tuple[int, int, float]]

    # Duplicate pairs (above duplicate threshold)
    duplicate_pairs: list[tuple[int, int, float]]


class SimilarityCalculator:
    """
    Calculate similarity between articles using embeddings or TF-IDF.

    Usage:
        calculator = SimilarityCalculator()

        # Using pre-computed embeddings
        result = calculator.calculate_from_embeddings(embeddings)

        # Using text directly (TF-IDF)
        result = calculator.calculate_from_texts(texts)

        # Find duplicates
        duplicates = result.duplicate_pairs
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        duplicate_threshold: Optional[float] = None
    ):
        """
        Initialize similarity calculator.

        Args:
            similarity_threshold: Threshold for similar articles (default from settings).
            duplicate_threshold: Threshold for duplicate detection (default from settings).
        """
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        self.duplicate_threshold = duplicate_threshold or settings.duplicate_threshold

    def calculate_from_embeddings(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> SimilarityResult:
        """
        Calculate similarity matrix from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            metric: Similarity metric ('cosine' or 'euclidean').

        Returns:
            SimilarityResult with matrix and pairs.
        """
        n_samples = len(embeddings)
        logger.info("Calculating similarity matrix", n_samples=n_samples, metric=metric)

        if metric == "cosine":
            matrix = self._cosine_similarity(embeddings)
        elif metric == "euclidean":
            matrix = self._euclidean_similarity(embeddings)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Find similar and duplicate pairs
        similar_pairs = self._find_pairs_above_threshold(
            matrix, self.similarity_threshold
        )
        duplicate_pairs = self._find_pairs_above_threshold(
            matrix, self.duplicate_threshold
        )

        logger.info(
            "Similarity calculation complete",
            similar_pairs=len(similar_pairs),
            duplicate_pairs=len(duplicate_pairs)
        )

        return SimilarityResult(
            matrix=matrix,
            similar_pairs=similar_pairs,
            duplicate_pairs=duplicate_pairs
        )

    def calculate_from_texts(
        self,
        texts: list[str],
        use_tfidf: bool = True
    ) -> SimilarityResult:
        """
        Calculate similarity matrix from texts using TF-IDF.

        Args:
            texts: List of text strings.
            use_tfidf: Use TF-IDF vectorization (vs simple count).

        Returns:
            SimilarityResult with matrix and pairs.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(texts)
        logger.info("Calculating TF-IDF similarity", n_samples=n_samples)

        if use_tfidf:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # Keep stopwords for better Spanish support
                ngram_range=(1, 2)
            )
        else:
            vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )

        tfidf_matrix = vectorizer.fit_transform(texts)
        matrix = cosine_similarity(tfidf_matrix)

        # Find pairs
        similar_pairs = self._find_pairs_above_threshold(
            matrix, self.similarity_threshold
        )
        duplicate_pairs = self._find_pairs_above_threshold(
            matrix, self.duplicate_threshold
        )

        logger.info(
            "TF-IDF similarity complete",
            similar_pairs=len(similar_pairs),
            duplicate_pairs=len(duplicate_pairs)
        )

        return SimilarityResult(
            matrix=matrix,
            similar_pairs=similar_pairs,
            duplicate_pairs=duplicate_pairs
        )

    def _cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings)

    def _euclidean_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity from euclidean distance.
        Converts distance to similarity: sim = 1 / (1 + dist)
        """
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(embeddings)
        return 1 / (1 + distances)

    def _find_pairs_above_threshold(
        self,
        matrix: np.ndarray,
        threshold: float
    ) -> list[tuple[int, int, float]]:
        """Find all pairs above similarity threshold."""
        pairs = []
        n = matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle
                if matrix[i, j] >= threshold:
                    pairs.append((i, j, float(matrix[i, j])))

        # Sort by similarity score (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def find_most_similar(
        self,
        embeddings: np.ndarray,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Find most similar items to a query embedding.

        Args:
            embeddings: Array of embeddings to search.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of (index, similarity_score) tuples.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate similarities
        query = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query, embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


def calculate_similarity(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.65,
    duplicate_threshold: float = 0.95
) -> SimilarityResult:
    """
    Convenience function to calculate similarity.

    Args:
        embeddings: Article embeddings.
        similarity_threshold: Threshold for similar articles.
        duplicate_threshold: Threshold for duplicates.

    Returns:
        SimilarityResult with similarity data.
    """
    calculator = SimilarityCalculator(
        similarity_threshold=similarity_threshold,
        duplicate_threshold=duplicate_threshold
    )
    return calculator.calculate_from_embeddings(embeddings)

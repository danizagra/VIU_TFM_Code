"""
Article deduplication using embeddings and similarity.

Identifies duplicate articles and selects representative articles
from groups of similar content.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

from src.config.settings import settings
from src.connectors.base import RawArticle
from src.processing.similarity import SimilarityCalculator, SimilarityResult

logger = structlog.get_logger()


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""

    # Unique articles (non-duplicates)
    unique_articles: list[RawArticle]

    # Indices of unique articles in original list
    unique_indices: list[int]

    # Duplicate groups: {representative_idx: [duplicate_indices]}
    duplicate_groups: dict[int, list[int]]

    # Total duplicates removed
    duplicates_removed: int

    # Similarity result (for analysis)
    similarity_result: Optional[SimilarityResult] = None

    @property
    def deduplication_rate(self) -> float:
        """Percentage of articles removed as duplicates."""
        total = len(self.unique_articles) + self.duplicates_removed
        if total == 0:
            return 0.0
        return self.duplicates_removed / total


class ArticleDeduplicator:
    """
    Deduplicate articles using embedding similarity.

    Identifies near-duplicate articles and keeps only the best
    representative from each group.

    Usage:
        deduplicator = ArticleDeduplicator()
        result = deduplicator.deduplicate(articles, embeddings)

        unique_articles = result.unique_articles
    """

    def __init__(
        self,
        duplicate_threshold: Optional[float] = None,
        selection_strategy: str = "longest"
    ):
        """
        Initialize deduplicator.

        Args:
            duplicate_threshold: Similarity threshold for duplicates (default from settings).
            selection_strategy: How to select representative article:
                - 'longest': Article with most content
                - 'newest': Most recently published
                - 'first': First article encountered
        """
        self.duplicate_threshold = duplicate_threshold or settings.duplicate_threshold
        self.selection_strategy = selection_strategy

    def deduplicate(
        self,
        articles: list[RawArticle],
        embeddings: np.ndarray
    ) -> DeduplicationResult:
        """
        Deduplicate articles based on embedding similarity.

        Args:
            articles: List of articles to deduplicate.
            embeddings: Corresponding embedding vectors.

        Returns:
            DeduplicationResult with unique articles and metadata.
        """
        n_articles = len(articles)
        logger.info(
            "Starting deduplication",
            n_articles=n_articles,
            threshold=self.duplicate_threshold
        )

        if n_articles == 0:
            return DeduplicationResult(
                unique_articles=[],
                unique_indices=[],
                duplicate_groups={},
                duplicates_removed=0
            )

        # Calculate similarity
        calculator = SimilarityCalculator(
            duplicate_threshold=self.duplicate_threshold
        )
        similarity_result = calculator.calculate_from_embeddings(embeddings)

        # Build duplicate groups using Union-Find
        duplicate_groups = self._build_duplicate_groups(
            n_articles,
            similarity_result.duplicate_pairs
        )

        # Select representatives
        unique_indices = []
        representative_groups = {}

        for group in duplicate_groups:
            if len(group) == 1:
                # Single article, no duplicates
                unique_indices.append(group[0])
            else:
                # Select representative
                rep_idx = self._select_representative(articles, group)
                unique_indices.append(rep_idx)
                representative_groups[rep_idx] = [i for i in group if i != rep_idx]

        # Sort by original order
        unique_indices.sort()
        unique_articles = [articles[i] for i in unique_indices]
        duplicates_removed = n_articles - len(unique_articles)

        logger.info(
            "Deduplication complete",
            original=n_articles,
            unique=len(unique_articles),
            removed=duplicates_removed,
            groups=len(representative_groups)
        )

        return DeduplicationResult(
            unique_articles=unique_articles,
            unique_indices=unique_indices,
            duplicate_groups=representative_groups,
            duplicates_removed=duplicates_removed,
            similarity_result=similarity_result
        )

    def _build_duplicate_groups(
        self,
        n_articles: int,
        duplicate_pairs: list[tuple[int, int, float]]
    ) -> list[list[int]]:
        """
        Build groups of duplicate articles using Union-Find.

        Returns list of groups, where each group contains indices
        of articles that are duplicates of each other.
        """
        # Union-Find data structure
        parent = list(range(n_articles))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union duplicate pairs
        for idx1, idx2, _ in duplicate_pairs:
            union(idx1, idx2)

        # Group by root
        groups: dict[int, list[int]] = {}
        for i in range(n_articles):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        return list(groups.values())

    def _select_representative(
        self,
        articles: list[RawArticle],
        group: list[int]
    ) -> int:
        """
        Select the best representative article from a duplicate group.

        Args:
            articles: Full list of articles.
            group: Indices of articles in the duplicate group.

        Returns:
            Index of the selected representative.
        """
        if self.selection_strategy == "longest":
            return self._select_longest(articles, group)
        elif self.selection_strategy == "newest":
            return self._select_newest(articles, group)
        elif self.selection_strategy == "first":
            return group[0]
        else:
            return group[0]

    def _select_longest(
        self,
        articles: list[RawArticle],
        group: list[int]
    ) -> int:
        """Select article with most content."""
        def content_length(idx: int) -> int:
            article = articles[idx]
            content = article.content or ""
            description = article.description or ""
            return len(content) + len(description)

        return max(group, key=content_length)

    def _select_newest(
        self,
        articles: list[RawArticle],
        group: list[int]
    ) -> int:
        """Select most recently published article."""
        from datetime import datetime

        def pub_date(idx: int) -> datetime:
            article = articles[idx]
            return article.published_at or datetime.min

        return max(group, key=pub_date)


def deduplicate_articles(
    articles: list[RawArticle],
    embeddings: np.ndarray,
    duplicate_threshold: float = 0.95
) -> DeduplicationResult:
    """
    Convenience function to deduplicate articles.

    Args:
        articles: List of articles to deduplicate.
        embeddings: Article embedding vectors.
        duplicate_threshold: Similarity threshold for duplicates.

    Returns:
        DeduplicationResult with unique articles.
    """
    deduplicator = ArticleDeduplicator(
        duplicate_threshold=duplicate_threshold
    )
    return deduplicator.deduplicate(articles, embeddings)

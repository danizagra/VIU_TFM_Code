"""
Agent state definitions for the journalist workflow.

Defines the TypedDict that flows through the LangGraph nodes,
containing articles, embeddings, clusters, and generated content.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

import numpy as np

from src.connectors.base import RawArticle


@dataclass
class ProcessedArticle:
    """
    An article with all generated content.

    Contains the original article plus summaries, headlines,
    angles, and quality scores.
    """

    # Original data
    raw: RawArticle
    embedding: np.ndarray | None = None

    # Generated content
    summary: str = ""
    headlines: dict[str, str] = field(default_factory=dict)
    angles: list[dict] = field(default_factory=list)

    # Metadata
    cluster_id: int = -1  # -1 = noise/unclustered
    quality_score: float = 0.0
    processed_at: datetime | None = None

    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = datetime.now()

    @property
    def title(self) -> str:
        return self.raw.title

    @property
    def source(self) -> str:
        return self.raw.source_name

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.raw.title,
            "source": self.raw.source_name,
            "url": self.raw.url,
            "published_at": self.raw.published_at.isoformat() if self.raw.published_at else None,
            "summary": self.summary,
            "headlines": self.headlines,
            "angles": self.angles,
            "cluster_id": self.cluster_id,
            "quality_score": self.quality_score,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


@dataclass
class ClusterInfo:
    """Information about a news cluster."""

    cluster_id: int
    article_indices: list[int] = field(default_factory=list)
    representative_idx: int | None = None
    topic_summary: str = ""


class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph agent.

    Each node reads from and writes to this state.
    Using total=False makes all fields optional.
    """

    # Configuration
    query: str  # Optional search query
    max_articles: int  # Max articles to fetch
    use_few_shot: bool  # Use few-shot prompts
    sources: list[str]  # News sources to use: rss, newsapi, gnews

    # Raw data
    raw_articles: list[RawArticle]  # Fetched articles

    # Processing results
    filtered_articles: list[RawArticle]  # After filtering
    embeddings: np.ndarray  # Article embeddings matrix
    cluster_labels: list[int]  # Cluster assignment per article
    clusters: list[ClusterInfo]  # Cluster metadata
    deduplicated_articles: list[RawArticle]  # After deduplication
    duplicate_groups: list[list[int]]  # Groups of duplicate indices

    # Generated content
    processed_articles: list[ProcessedArticle]  # Final processed articles

    # Quality and metrics
    quality_results: dict[int, dict]  # Quality check per article index
    failed_articles: list[int]  # Indices of articles that failed quality

    # Persistence (Option B - cache inteligente)
    existing_urls: set[str]  # URLs already in database
    skipped_count: int  # Articles skipped (already processed)
    session_id: str  # Database session ID
    saved_article_ids: list[str]  # IDs of saved articles

    # Historical context (Option C - búsqueda híbrida)
    historical_articles: list[RawArticle]  # Similar articles from DB
    historical_embeddings: np.ndarray  # Embeddings of historical articles
    historical_indices: list[int]  # Indices of historical articles in merged list
    all_articles: list[RawArticle]  # Merged: current + historical
    all_embeddings: np.ndarray  # Merged embeddings

    # Execution metadata
    current_step: str  # Current node being executed
    errors: list[str]  # Errors encountered
    start_time: datetime
    end_time: datetime


def create_initial_state(
    query: str = "",
    max_articles: int = 20,
    use_few_shot: bool = True,
    sources: list[str] | None = None,
) -> AgentState:
    """
    Create initial state for the agent.

    Args:
        query: Optional search query for news.
        max_articles: Maximum articles to fetch.
        use_few_shot: Whether to use few-shot prompts.
        sources: News sources to use. Options: 'rss', 'newsapi', 'gnews'.
                 Default is ['rss'] if None.

    Returns:
        Initial AgentState ready for execution.
    """
    if sources is None:
        sources = ["rss"]

    return AgentState(
        query=query,
        max_articles=max_articles,
        use_few_shot=use_few_shot,
        sources=sources,
        raw_articles=[],
        filtered_articles=[],
        deduplicated_articles=[],
        processed_articles=[],
        cluster_labels=[],
        clusters=[],
        duplicate_groups=[],
        quality_results={},
        failed_articles=[],
        errors=[],
        current_step="initialized",
        start_time=datetime.now(),
    )

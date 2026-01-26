"""
News article clustering using UMAP for dimensionality reduction
and HDBSCAN for density-based clustering.

HDBSCAN advantages:
- No need to specify number of clusters (K)
- Automatically detects noise points
- Works well with varying cluster densities
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    # Cluster labels (-1 = noise/outlier)
    labels: np.ndarray

    # Number of clusters found (excluding noise)
    n_clusters: int

    # Number of noise points
    n_noise: int

    # Cluster probabilities (confidence scores)
    probabilities: Optional[np.ndarray] = None

    # Reduced embeddings (2D for visualization)
    embeddings_2d: Optional[np.ndarray] = None

    # Centroid for each cluster (in original embedding space)
    centroids: dict[int, np.ndarray] = field(default_factory=dict)

    # Articles per cluster
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def get_cluster_indices(self, label: int) -> np.ndarray:
        """Get indices of articles in a specific cluster."""
        return np.where(self.labels == label)[0]

    def get_noise_indices(self) -> np.ndarray:
        """Get indices of noise points (label = -1)."""
        return np.where(self.labels == -1)[0]


class NewsClustering:
    """
    Cluster news articles based on their embeddings.

    Uses UMAP for dimensionality reduction (optional but recommended)
    and HDBSCAN for clustering.

    Usage:
        clustering = NewsClustering()
        result = clustering.fit_predict(embeddings)

        print(f"Found {result.n_clusters} clusters")
        print(f"Noise points: {result.n_noise}")
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        use_umap: bool = True,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean"
    ):
        """
        Initialize clustering.

        Args:
            min_cluster_size: Minimum number of articles to form a cluster.
            min_samples: Number of samples in neighborhood for core points.
            use_umap: Whether to reduce dimensions with UMAP before clustering.
            umap_n_components: Number of dimensions after UMAP reduction.
            umap_n_neighbors: UMAP neighborhood size.
            umap_min_dist: UMAP minimum distance parameter.
            cluster_selection_epsilon: Distance threshold for cluster merging.
            metric: Distance metric ('euclidean', 'cosine', etc.).
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

        self._umap_model = None
        self._hdbscan_model = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
        return_2d: bool = True
    ) -> ClusterResult:
        """
        Cluster embeddings and return results.

        Args:
            embeddings: Numpy array of shape (n_samples, embedding_dim).
            return_2d: Whether to include 2D embeddings for visualization.

        Returns:
            ClusterResult with labels, centroids, and metadata.
        """
        import hdbscan

        n_samples = len(embeddings)
        logger.info(
            "Starting clustering",
            n_samples=n_samples,
            embedding_dim=embeddings.shape[1] if len(embeddings) > 0 else 0
        )

        if n_samples < self.min_cluster_size:
            logger.warning("Too few samples for clustering")
            return ClusterResult(
                labels=np.full(n_samples, -1),
                n_clusters=0,
                n_noise=n_samples
            )

        # Dimensionality reduction with UMAP
        if self.use_umap and embeddings.shape[1] > self.umap_n_components:
            embeddings_reduced = self._apply_umap(embeddings)
        else:
            embeddings_reduced = embeddings

        # HDBSCAN clustering
        logger.info("Running HDBSCAN", min_cluster_size=self.min_cluster_size)

        self._hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            prediction_data=True
        )

        labels = self._hdbscan_model.fit_predict(embeddings_reduced)
        probabilities = self._hdbscan_model.probabilities_

        # Calculate statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)

        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = np.sum(labels == label)

        # Calculate centroids in original embedding space
        centroids = {}
        for label in unique_labels:
            if label != -1:
                mask = labels == label
                centroids[label] = embeddings[mask].mean(axis=0)

        logger.info(
            "Clustering complete",
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_sizes=cluster_sizes
        )

        # Get 2D embeddings for visualization
        embeddings_2d = None
        if return_2d:
            embeddings_2d = self._get_2d_embeddings(embeddings)

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            probabilities=probabilities,
            embeddings_2d=embeddings_2d,
            centroids=centroids,
            cluster_sizes=cluster_sizes
        )

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        import umap

        logger.info(
            "Applying UMAP",
            input_dim=embeddings.shape[1],
            output_dim=self.umap_n_components
        )

        self._umap_model = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.metric,
            random_state=42
        )

        return self._umap_model.fit_transform(embeddings)

    def _get_2d_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Get 2D embeddings for visualization."""
        import umap

        if embeddings.shape[1] <= 2:
            return embeddings

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric=self.metric,
            random_state=42
        )

        return reducer.fit_transform(embeddings)


def cluster_articles(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    use_umap: bool = True
) -> ClusterResult:
    """
    Convenience function to cluster article embeddings.

    Args:
        embeddings: Article embeddings array.
        min_cluster_size: Minimum articles per cluster.
        use_umap: Whether to use UMAP reduction.

    Returns:
        ClusterResult with clustering results.
    """
    clustering = NewsClustering(
        min_cluster_size=min_cluster_size,
        use_umap=use_umap
    )
    return clustering.fit_predict(embeddings)

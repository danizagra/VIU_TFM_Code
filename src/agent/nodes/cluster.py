"""
Cluster node - Groups similar articles into clusters.
"""

import structlog

from src.agent.state import AgentState, ClusterInfo
from src.processing.clustering import NewsClustering

logger = structlog.get_logger()


def cluster_articles(state: AgentState) -> AgentState:
    """
    Cluster articles based on embeddings similarity.

    Reads:
        - embeddings: Article embeddings
        - filtered_articles: Articles to cluster

    Writes:
        - cluster_labels: Cluster ID per article (-1 = noise)
        - clusters: List of ClusterInfo objects
        - current_step: Updated to 'clustered'
    """
    state["current_step"] = "clustering"

    embeddings = state.get("embeddings")
    articles = state.get("filtered_articles", [])

    if embeddings is None or len(articles) == 0:
        logger.warning("No embeddings or articles to cluster")
        state["cluster_labels"] = []
        state["clusters"] = []
        state["current_step"] = "clustered"
        return state

    logger.info("Clustering articles", count=len(articles))

    try:
        # Configure clustering based on dataset size
        min_cluster_size = 2 if len(articles) < 10 else 3
        use_umap = len(articles) > 15  # UMAP needs more samples

        clustering = NewsClustering(
            min_cluster_size=min_cluster_size,
            use_umap=use_umap,
        )

        result = clustering.fit_predict(embeddings, return_2d=False)

        # Build cluster info
        clusters = []
        for cluster_id in range(result.n_clusters):
            indices = [i for i, label in enumerate(result.labels) if label == cluster_id]
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                article_indices=indices,
                representative_idx=indices[0] if indices else None,
            )
            clusters.append(cluster_info)

        state["cluster_labels"] = list(result.labels)
        state["clusters"] = clusters
        state["current_step"] = "clustered"

        logger.info(
            "Clustering complete",
            n_clusters=result.n_clusters,
            n_noise=result.n_noise,
            cluster_sizes=result.cluster_sizes,
        )

    except Exception as e:
        error_msg = f"Error clustering articles: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        # Assign all to noise on error
        state["cluster_labels"] = [-1] * len(articles)
        state["clusters"] = []
        state["current_step"] = "cluster_failed"

    return state

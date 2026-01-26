"""
Save node - Persists processed articles to database.
"""

import structlog

from src.agent.state import AgentState
from src.storage.database import get_db
from src.storage.repositories.article import ArticleRepository
from src.storage.repositories.processed import ProcessedArticleRepository
from src.storage.repositories.session import SessionRepository, ClusterRepository

logger = structlog.get_logger()


def save_to_database(state: AgentState) -> AgentState:
    """
    Save processed articles and session data to database.

    Reads:
        - raw_articles: Original fetched articles
        - processed_articles: Articles with generated content
        - clusters: Cluster information
        - embeddings: Article embeddings
        - quality_results: Quality check results

    Writes:
        - session_id: Database session ID
        - saved_article_ids: List of saved article IDs
        - current_step: Updated to 'saved'
    """
    state["current_step"] = "saving"

    processed_articles = state.get("processed_articles", [])

    if not processed_articles:
        logger.warning("No processed articles to save")
        state["current_step"] = "saved"
        return state

    logger.info("Saving to database", count=len(processed_articles))

    try:
        with get_db() as db:
            # Initialize repositories
            article_repo = ArticleRepository(db)
            processed_repo = ProcessedArticleRepository(db)
            session_repo = SessionRepository(db)
            cluster_repo = ClusterRepository(db)

            # Create agent session
            agent_session = session_repo.create(
                query=state.get("query", ""),
                filters={
                    "max_articles": state.get("max_articles", 20),
                    "use_few_shot": state.get("use_few_shot", True),
                },
            )
            state["session_id"] = agent_session.id

            # Get embeddings if available
            embeddings = state.get("embeddings")

            # Save clusters first
            clusters = state.get("clusters", [])
            cluster_id_map = {}  # Map cluster label to DB cluster ID

            for cluster_info in clusters:
                db_cluster = cluster_repo.create(
                    session_id=agent_session.id,
                    cluster_label=cluster_info.cluster_id,
                    article_count=len(cluster_info.article_indices),
                )
                cluster_id_map[cluster_info.cluster_id] = db_cluster.id

            # Save articles and processed content
            saved_article_ids = []
            quality_results = state.get("quality_results", {})

            for i, processed in enumerate(processed_articles):
                raw = processed.raw

                # Get embedding for this article
                embedding = None
                if embeddings is not None and i < len(embeddings):
                    embedding = embeddings[i]

                # Check if article already exists by URL
                existing = article_repo.get_by_url(raw.source_url) if raw.source_url else None

                if existing:
                    article = existing
                    # Update embedding if we have a new one
                    if embedding is not None and existing.embedding is None:
                        article_repo.update_embedding(existing.id, embedding)
                else:
                    # Create new article
                    article = article_repo.create(raw, embedding)

                saved_article_ids.append(article.id)

                # Get cluster DB ID
                db_cluster_id = cluster_id_map.get(processed.cluster_id) if processed.cluster_id >= 0 else None

                # Get quality info
                quality_info = quality_results.get(i, {})
                quality_score = quality_info.get("overall_score", processed.quality_score)
                quality_passed = quality_score >= 0.5 if quality_score else None

                # Create processed article record
                processed_repo.create(
                    article_id=article.id,
                    session_id=agent_session.id,
                    summary=processed.summary,
                    headlines=processed.headlines,
                    angles=processed.angles,
                    cluster_id=db_cluster_id,
                    quality_score=quality_score,
                    quality_passed=quality_passed,
                    llm_provider="lm_studio",  # TODO: get from config
                )

            # Update session metrics
            session_repo.update_metrics(
                agent_session.id,
                articles_fetched=len(state.get("raw_articles", [])),
                articles_after_filter=len(state.get("filtered_articles", [])),
                articles_after_dedup=len(state.get("deduplicated_articles", [])),
                clusters_found=len(clusters),
            )

            # Mark session complete
            session_repo.complete(agent_session.id)

            state["saved_article_ids"] = saved_article_ids
            state["current_step"] = "saved"

            logger.info(
                "Saved to database",
                session_id=str(agent_session.id),
                articles_saved=len(saved_article_ids),
                clusters_saved=len(clusters),
            )

    except Exception as e:
        error_msg = f"Error saving to database: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["current_step"] = "save_failed"

    return state

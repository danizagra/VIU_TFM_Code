"""
Embed node - Generates embeddings for articles.
"""

import structlog

from src.agent.state import AgentState
from src.processing.embeddings import get_embedding_generator

logger = structlog.get_logger()


def embed_articles(state: AgentState) -> AgentState:
    """
    Generate embeddings for filtered articles.

    Reads:
        - filtered_articles: Articles to embed

    Writes:
        - embeddings: numpy array of embeddings
        - current_step: Updated to 'embedded'
    """
    state["current_step"] = "embedding"

    articles = state.get("filtered_articles", [])

    if not articles:
        logger.warning("No articles to embed")
        state["embeddings"] = None
        state["current_step"] = "embedded"
        return state

    logger.info("Generating embeddings", count=len(articles))

    try:
        generator = get_embedding_generator()
        embeddings = generator.embed_articles(articles)

        state["embeddings"] = embeddings
        state["current_step"] = "embedded"

        logger.info(
            "Embeddings generated",
            shape=embeddings.shape,
            dimension=generator.embedding_dimension,
        )

    except Exception as e:
        error_msg = f"Error generating embeddings: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["embeddings"] = None
        state["current_step"] = "embed_failed"

    return state

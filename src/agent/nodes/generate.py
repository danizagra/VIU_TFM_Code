"""
Generate node - Creates summaries, headlines, and angles.
"""

import structlog

from src.agent.state import AgentState, ProcessedArticle
from src.generation.summarizer import ArticleSummarizer
from src.generation.headlines import HeadlineGenerator
from src.generation.angles import AngleGenerator
from src.llm.factory import get_llm_client

logger = structlog.get_logger()


def generate_content(state: AgentState) -> AgentState:
    """
    Generate summaries, headlines, and angles for articles.

    Reads:
        - deduplicated_articles: Articles to process
        - cluster_labels: Cluster assignments
        - use_few_shot: Whether to use few-shot prompts

    Writes:
        - processed_articles: List of ProcessedArticle with generated content
        - current_step: Updated to 'generated'
    """
    state["current_step"] = "generating"

    articles = state.get("deduplicated_articles", [])
    cluster_labels = state.get("cluster_labels", [])
    use_few_shot = state.get("use_few_shot", True)

    if not articles:
        logger.warning("No articles to generate content for")
        state["processed_articles"] = []
        state["current_step"] = "generated"
        return state

    logger.info("Generating content", count=len(articles), use_few_shot=use_few_shot)

    try:
        # Initialize LLM client and generators
        llm_client = get_llm_client()
        summarizer = ArticleSummarizer(llm_client, use_few_shot=use_few_shot)
        headline_gen = HeadlineGenerator(llm_client, use_few_shot=use_few_shot)
        angle_gen = AngleGenerator(llm_client, use_few_shot=use_few_shot)

        processed_articles = []

        for i, article in enumerate(articles):
            logger.info(
                "Processing article",
                index=i + 1,
                total=len(articles),
                title=article.title[:50],
            )

            try:
                # Get cluster label (if available)
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1

                # Generate summary
                summary_result = summarizer.summarize_article(article)
                summary = summary_result.summary

                # Generate headlines
                headline_result = headline_gen.generate(
                    title=article.title,
                    summary=summary,
                )
                headlines = {
                    "informativo": headline_result.informativo,
                    "engagement": headline_result.engagement,
                    "seo": headline_result.seo,
                }

                # Generate angles
                angle_result = angle_gen.generate(
                    title=article.title,
                    summary=summary,
                )
                angles = [a.to_dict() for a in angle_result.angles]

                # Create processed article
                processed = ProcessedArticle(
                    raw=article,
                    summary=summary,
                    headlines=headlines,
                    angles=angles,
                    cluster_id=cluster_id,
                )
                processed_articles.append(processed)

            except Exception as e:
                logger.error(
                    "Error processing article",
                    index=i,
                    title=article.title[:50],
                    error=str(e),
                )
                # Create minimal processed article on error
                processed = ProcessedArticle(
                    raw=article,
                    summary="",
                    headlines={},
                    angles=[],
                    cluster_id=-1,
                )
                processed_articles.append(processed)

        state["processed_articles"] = processed_articles
        state["current_step"] = "generated"

        logger.info(
            "Content generation complete",
            processed=len(processed_articles),
        )

    except Exception as e:
        error_msg = f"Error in content generation: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["processed_articles"] = []
        state["current_step"] = "generate_failed"

    return state

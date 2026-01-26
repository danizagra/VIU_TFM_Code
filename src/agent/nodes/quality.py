"""
Quality node - Validates generated content quality.
"""

import structlog

from src.agent.state import AgentState
from src.generation.quality_checker import QualityChecker

logger = structlog.get_logger()


def check_quality(state: AgentState) -> AgentState:
    """
    Check quality of generated content.

    Reads:
        - processed_articles: Articles with generated content

    Writes:
        - quality_results: Quality scores per article
        - failed_articles: Indices of articles that failed quality
        - current_step: Updated to 'quality_checked'
    """
    state["current_step"] = "checking_quality"

    processed_articles = state.get("processed_articles", [])

    if not processed_articles:
        logger.warning("No processed articles to check")
        state["quality_results"] = {}
        state["failed_articles"] = []
        state["current_step"] = "quality_checked"
        return state

    logger.info("Checking quality", count=len(processed_articles))

    try:
        checker = QualityChecker()
        quality_results = {}
        failed_articles = []

        for i, article in enumerate(processed_articles):
            # Check summary
            summary_result = checker.check_summary(article.summary)

            # Check headlines
            headline_results = {}
            if article.headlines:
                headline_results = checker.check_headlines_set(
                    article.headlines.get("informativo", ""),
                    article.headlines.get("engagement", ""),
                    article.headlines.get("seo", ""),
                )

            # Check angles
            angles_result = checker.check_angles(article.angles)

            # Calculate overall score
            scores = [summary_result.score]
            if headline_results:
                scores.extend([r.score for r in headline_results.values()])
            scores.append(angles_result.score)

            overall_score = sum(scores) / len(scores)

            # Update article quality score
            article.quality_score = overall_score

            # Store results
            quality_results[i] = {
                "summary": {
                    "passed": summary_result.passed,
                    "score": summary_result.score,
                    "issues": [
                        {"code": iss.code, "message": iss.message}
                        for iss in summary_result.issues
                    ],
                },
                "headlines": {
                    htype: {
                        "passed": hresult.passed,
                        "score": hresult.score,
                    }
                    for htype, hresult in headline_results.items()
                } if headline_results else {},
                "angles": {
                    "passed": angles_result.passed,
                    "score": angles_result.score,
                },
                "overall_score": overall_score,
            }

            # Track failures (score < 0.5)
            if overall_score < 0.5:
                failed_articles.append(i)

        state["quality_results"] = quality_results
        state["failed_articles"] = failed_articles
        state["current_step"] = "quality_checked"

        # Calculate stats
        avg_score = sum(r["overall_score"] for r in quality_results.values()) / len(quality_results)

        logger.info(
            "Quality check complete",
            articles_checked=len(processed_articles),
            failed=len(failed_articles),
            avg_score=f"{avg_score:.2f}",
        )

    except Exception as e:
        error_msg = f"Error checking quality: {e}"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["quality_results"] = {}
        state["failed_articles"] = []
        state["current_step"] = "quality_failed"

    return state

"""
LangGraph workflow for the journalist agent.

Defines the graph that orchestrates the complete pipeline:
fetch → check_existing → filter → embed → load_similar → cluster → deduplicate → generate → quality → save → finalize

Implements:
- Option B (cache inteligente): Skip articles already in database
- Option C (búsqueda híbrida): Load similar historical articles for context
"""

from datetime import datetime

import structlog
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState, create_initial_state
from src.agent.nodes.fetch import fetch_articles
from src.agent.nodes.check_existing import check_existing_articles
from src.agent.nodes.filter import filter_articles
from src.agent.nodes.embed import embed_articles
from src.agent.nodes.load_similar import load_similar_from_db
from src.agent.nodes.cluster import cluster_articles
from src.agent.nodes.deduplicate import deduplicate_articles
from src.agent.nodes.generate import generate_content
from src.agent.nodes.quality import check_quality
from src.agent.nodes.save import save_to_database

logger = structlog.get_logger()


def should_continue_after_fetch(state: AgentState) -> str:
    """Decide whether to continue after fetching."""
    if not state.get("raw_articles"):
        logger.warning("No articles fetched, ending workflow")
        return "end"
    return "continue"


def should_continue_after_check(state: AgentState) -> str:
    """Decide whether to continue after checking existing."""
    if not state.get("raw_articles"):
        logger.warning("All articles already in database, ending workflow")
        return "end"
    return "continue"


def should_continue_after_filter(state: AgentState) -> str:
    """Decide whether to continue after filtering."""
    if not state.get("filtered_articles"):
        logger.warning("No articles after filtering, ending workflow")
        return "end"
    return "continue"


def should_continue_after_dedup(state: AgentState) -> str:
    """Decide whether to continue after deduplication."""
    if not state.get("deduplicated_articles"):
        logger.warning("No articles after deduplication, ending workflow")
        return "end"
    return "continue"


def finalize(state: AgentState) -> AgentState:
    """Final node - records end time and summary."""
    state["end_time"] = datetime.now()
    state["current_step"] = "completed"

    # Calculate duration
    start = state.get("start_time")
    end = state.get("end_time")
    if start and end:
        duration = (end - start).total_seconds()
    else:
        duration = 0

    # Log summary
    processed = state.get("processed_articles", [])
    errors = state.get("errors", [])
    skipped = state.get("skipped_count", 0)
    saved = state.get("saved_article_ids", [])

    logger.info(
        "Workflow completed",
        articles_processed=len(processed),
        articles_saved=len(saved),
        articles_skipped=skipped,
        errors=len(errors),
        duration_seconds=f"{duration:.1f}",
    )

    return state


def create_journalist_agent(use_persistence: bool = True) -> StateGraph:
    """
    Create the journalist agent workflow graph.

    Args:
        use_persistence: Whether to include DB persistence nodes.

    The graph follows this flow:
    fetch → check_existing → filter → embed → load_similar → cluster → deduplicate → generate → quality → save → finalize

    Returns:
        Compiled StateGraph ready to invoke.
    """
    # Create graph with state type
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("fetch", fetch_articles)
    workflow.add_node("filter", filter_articles)
    workflow.add_node("embed", embed_articles)
    workflow.add_node("cluster", cluster_articles)
    workflow.add_node("deduplicate", deduplicate_articles)
    workflow.add_node("generate", generate_content)
    workflow.add_node("quality", check_quality)
    workflow.add_node("finalize", finalize)

    if use_persistence:
        workflow.add_node("check_existing", check_existing_articles)
        workflow.add_node("load_similar", load_similar_from_db)
        workflow.add_node("save", save_to_database)

    # Set entry point
    workflow.set_entry_point("fetch")

    if use_persistence:
        # Flow with persistence: fetch → check_existing → filter → embed → load_similar → cluster → ...
        workflow.add_conditional_edges(
            "fetch",
            should_continue_after_fetch,
            {
                "continue": "check_existing",
                "end": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "check_existing",
            should_continue_after_check,
            {
                "continue": "filter",
                "end": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "filter",
            should_continue_after_filter,
            {
                "continue": "embed",
                "end": "finalize",
            },
        )

        workflow.add_edge("embed", "load_similar")
        workflow.add_edge("load_similar", "cluster")
        workflow.add_edge("cluster", "deduplicate")

        workflow.add_conditional_edges(
            "deduplicate",
            should_continue_after_dedup,
            {
                "continue": "generate",
                "end": "finalize",
            },
        )

        workflow.add_edge("generate", "quality")
        workflow.add_edge("quality", "save")
        workflow.add_edge("save", "finalize")

    else:
        # Flow without persistence (original)
        workflow.add_conditional_edges(
            "fetch",
            should_continue_after_fetch,
            {
                "continue": "filter",
                "end": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "filter",
            should_continue_after_filter,
            {
                "continue": "embed",
                "end": "finalize",
            },
        )

        workflow.add_edge("embed", "cluster")
        workflow.add_edge("cluster", "deduplicate")

        workflow.add_conditional_edges(
            "deduplicate",
            should_continue_after_dedup,
            {
                "continue": "generate",
                "end": "finalize",
            },
        )

        workflow.add_edge("generate", "quality")
        workflow.add_edge("quality", "finalize")

    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_journalist_agent(
    query: str = "",
    max_articles: int = 20,
    use_few_shot: bool = True,
    use_persistence: bool = True,
    sources: list[str] | None = None,
) -> AgentState:
    """
    Run the journalist agent workflow.

    Args:
        query: Optional search query for news.
        max_articles: Maximum articles to fetch.
        use_few_shot: Whether to use few-shot prompts.
        use_persistence: Whether to use database persistence.
        sources: News sources to use. Options: 'rss', 'newsapi', 'gnews', 'all'.
                 Default is ['rss'].

    Returns:
        Final AgentState with all results.
    """
    if sources is None:
        sources = ["rss"]

    logger.info(
        "Starting journalist agent",
        query=query or "(no query)",
        max_articles=max_articles,
        use_persistence=use_persistence,
        sources=sources,
    )

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        max_articles=max_articles,
        use_few_shot=use_few_shot,
        sources=sources,
    )

    # Create and run graph
    graph = create_journalist_agent(use_persistence=use_persistence)
    final_state = graph.invoke(initial_state)

    return final_state

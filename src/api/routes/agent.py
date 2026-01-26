"""
Agent routes for triggering and monitoring the journalist agent pipeline.

Endpoints:
- POST /agent/run - Execute the agent pipeline
- GET /agent/sessions - List recent sessions
- GET /agent/sessions/{id} - Get session details
"""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    SessionListResponse,
    SessionResponse,
)
from src.storage.repositories.session import SessionRepository

logger = structlog.get_logger()

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/run", response_model=AgentRunResponse)
def run_agent(
    request: AgentRunRequest,
    db: Session = Depends(get_db),
) -> AgentRunResponse:
    """
    Execute the journalist agent pipeline.

    This runs the complete workflow:
    fetch -> filter -> embed -> cluster -> deduplicate -> generate -> quality -> save

    Note: This is a synchronous operation that may take several minutes
    depending on the number of articles and LLM response times.
    """
    from src.agent.graph import run_journalist_agent

    logger.info(
        "Starting agent run",
        query=request.query or "(no query)",
        max_articles=request.max_articles,
        sources=request.sources,
    )

    try:
        # Run the agent
        final_state = run_journalist_agent(
            query=request.query,
            max_articles=request.max_articles,
            use_persistence=request.use_persistence,
            sources=request.sources,
        )

        # Extract results from state
        processed = final_state.get("processed_articles", [])
        saved_ids = final_state.get("saved_article_ids", [])
        errors = final_state.get("errors", [])

        # Calculate duration
        start_time = final_state.get("start_time")
        end_time = final_state.get("end_time")
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()

        # Get session ID if persistence was enabled
        session_id_str = final_state.get("session_id")
        session_id: Optional[UUID] = UUID(session_id_str) if session_id_str else None

        return AgentRunResponse(
            status="completed",
            session_id=session_id,
            articles_fetched=len(final_state.get("raw_articles", [])),
            articles_processed=len(processed),
            articles_saved=len(saved_ids),
            errors=errors,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.exception("Agent run failed", error=str(e))
        return AgentRunResponse(
            status="failed",
            errors=[str(e)],
        )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(10, ge=1, le=100),
    status: str | None = Query(None, description="Filter by status: completed, failed, running"),
    db: Session = Depends(get_db),
) -> SessionListResponse:
    """
    List recent agent execution sessions.
    """
    logger.info("Listing sessions", limit=limit, status=status)

    session_repo = SessionRepository(db)

    if status == "completed":
        sessions = session_repo.get_successful(limit=limit)
    else:
        sessions = session_repo.get_recent(limit=limit)

    # Filter by status if specified and not already filtered
    if status and status != "completed":
        sessions = [s for s in sessions if s.status == status]

    items = [
        SessionResponse(
            id=s.id,
            query=s.query,
            status=s.status,
            articles_fetched=s.articles_fetched,
            articles_after_filter=s.articles_after_filter,
            articles_after_dedup=s.articles_after_dedup,
            clusters_found=s.clusters_found,
            started_at=s.started_at,
            completed_at=s.completed_at,
            error_message=s.error_message,
        )
        for s in sessions
    ]

    return SessionListResponse(
        items=items,
        total=len(items),
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: UUID,
    db: Session = Depends(get_db),
) -> SessionResponse:
    """
    Get details of a specific agent session.
    """
    logger.info("Fetching session", session_id=str(session_id))

    session_repo = SessionRepository(db)
    session = session_repo.get_by_id(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session.id,
        query=session.query,
        status=session.status,
        articles_fetched=session.articles_fetched,
        articles_after_filter=session.articles_after_filter,
        articles_after_dedup=session.articles_after_dedup,
        clusters_found=session.clusters_found,
        started_at=session.started_at,
        completed_at=session.completed_at,
        error_message=session.error_message,
    )

"""
RAG-powered Q&A routes.

Endpoints:
- POST /ask - Ask a question and get an answer with citations
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import AskRequest, AskResponse, SourceCitation
from src.rag.engine import RAGEngine

logger = structlog.get_logger()

router = APIRouter(prefix="/ask", tags=["RAG Q&A"])


@router.post("/", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    Ask a question about news and get an AI-generated answer with citations.

    The system uses RAG (Retrieval-Augmented Generation) to:
    1. Find relevant articles via semantic search
    2. Generate an answer based only on those articles
    3. Include citations [1], [2], etc. to the sources

    Example request:
    ```json
    {
        "question": "Cuales son las ultimas noticias sobre economia?",
        "max_sources": 5,
        "use_reranking": false
    }
    ```
    """
    logger.info(
        "Received RAG question",
        question=request.question[:50],
        max_sources=request.max_sources,
        use_reranking=request.use_reranking,
    )

    try:
        # Initialize RAG engine
        engine = RAGEngine(
            db=db,
            use_reranking=request.use_reranking,
        )

        # Get answer
        rag_response = engine.ask(
            question=request.question,
            max_sources=request.max_sources,
            category=request.category,
        )

        # Convert to API response
        sources = [
            SourceCitation(
                index=s.index,
                title=s.title,
                url=s.url,
                source_name=s.source_name,
                published_at=s.published_at,
                relevance_score=s.relevance_score,
            )
            for s in rag_response.sources
        ]

        return AskResponse(
            answer=rag_response.answer,
            sources=sources,
            confidence=rag_response.confidence,
            question=request.question,
        )

    except Exception as e:
        logger.exception("RAG query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


@router.post("/context")
def get_context(
    request: AskRequest,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get relevant articles for a question without generating an answer.

    Useful for debugging or inspecting what articles would be used
    to answer a question.
    """
    logger.info("Getting context for question", question=request.question[:50])

    try:
        engine = RAGEngine(db=db)

        articles = engine.get_context_only(
            question=request.question,
            max_sources=request.max_sources,
            category=request.category,
        )

        return {
            "question": request.question,
            "articles": [
                {
                    "title": a.title,
                    "source_name": a.source_name,
                    "source_url": a.source_url,
                    "description": a.description,
                    "published_at": a.published_at.isoformat() if a.published_at else None,
                    "similarity_score": score,
                }
                for a, score in articles
            ],
            "count": len(articles),
        }

    except Exception as e:
        logger.exception("Context retrieval failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}",
        )

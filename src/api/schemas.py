"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

# =============================================================================
# Article Schemas
# =============================================================================


class ArticleBase(BaseModel):
    """Base article fields."""

    title: str
    source_name: str
    source_url: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    published_at: Optional[datetime] = None
    category: Optional[str] = None
    language: Optional[str] = None


class ArticleResponse(ArticleBase):
    """Article response with all fields."""

    id: UUID
    content: Optional[str] = None
    fetched_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class ProcessedArticleResponse(BaseModel):
    """Processed article with generated content."""

    id: UUID
    article_id: UUID
    summary: Optional[str] = None
    headlines: Optional[dict] = None
    angles: Optional[list] = None
    quality_score: Optional[float] = None
    quality_passed: Optional[bool] = None
    created_at: datetime

    # Source article info (joined)
    title: str
    source_name: str
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ArticleListResponse(BaseModel):
    """Paginated list of articles."""

    items: list[ProcessedArticleResponse]
    total: int
    limit: int
    offset: int


class ArticleSearchRequest(BaseModel):
    """Search request parameters."""

    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(10, ge=1, le=100)
    category: Optional[str] = None


# =============================================================================
# RAG Schemas
# =============================================================================


class SourceCitation(BaseModel):
    """Source citation for RAG responses."""

    index: int = Field(..., description="Citation number [1], [2], etc.")
    title: str
    url: Optional[str] = None
    source_name: str
    published_at: Optional[datetime] = None
    relevance_score: float = Field(..., ge=0, le=1)


class AskRequest(BaseModel):
    """RAG question request."""

    question: str = Field(
        ..., min_length=3, max_length=500, description="Question to ask"
    )
    max_sources: int = Field(5, ge=1, le=20, description="Maximum sources to use")
    category: Optional[str] = Field(None, description="Filter by category")
    use_reranking: bool = Field(True, description="Use LLM reranking for better accuracy")


class AskResponse(BaseModel):
    """RAG question response."""

    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    question: str


# =============================================================================
# Agent Schemas
# =============================================================================


class AgentRunRequest(BaseModel):
    """Request to run the journalist agent."""

    query: str = Field("", description="Search query for news")
    max_articles: int = Field(20, ge=1, le=100)
    sources: list[str] = Field(
        default=["rss"],
        description="News sources: rss, newsapi, gnews",
    )
    use_persistence: bool = Field(True, description="Save results to database")


class AgentRunResponse(BaseModel):
    """Response from agent execution."""

    status: str = Field(..., description="completed, failed, or running")
    session_id: Optional[UUID] = None
    articles_fetched: int = 0
    articles_processed: int = 0
    articles_saved: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None


class SessionResponse(BaseModel):
    """Agent session details."""

    id: UUID
    query: Optional[str] = None
    status: str
    articles_fetched: int
    articles_after_filter: int
    articles_after_dedup: int
    clusters_found: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    """List of agent sessions."""

    items: list[SessionResponse]
    total: int

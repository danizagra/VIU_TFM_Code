"""
FastAPI application for the Journalist Agent API.

Provides REST endpoints for:
- News retrieval and search
- RAG-powered Q&A
- Agent execution
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.api.routes import agent, ask, news
from src.llm.factory import get_available_client
from src.storage.database import engine

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Startup: verify database and LLM connectivity.
    Shutdown: cleanup resources.
    """
    # Startup
    logger.info("Starting Journalist Agent API")

    # Verify database connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
    except Exception as e:
        logger.error("Database connection failed", error=str(e))

    # Check LLM availability (optional, non-blocking)
    try:
        llm = get_available_client()
        if llm:
            logger.info("LLM client available", provider=llm.__class__.__name__)
        else:
            logger.warning("No LLM client available - RAG features will be limited")
    except Exception as e:
        logger.warning("LLM check failed", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down Journalist Agent API")


app = FastAPI(
    title="Journalist Agent API",
    description="AI-powered news processing and Q&A system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(news.router)
app.include_router(ask.router)
app.include_router(agent.router)


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "journalist-agent-api"}


@app.get("/", tags=["Health"])
def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "Journalist Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

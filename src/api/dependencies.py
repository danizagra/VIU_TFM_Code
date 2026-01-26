"""
FastAPI dependencies for dependency injection.
"""

from typing import Generator

from sqlalchemy.orm import Session

from src.llm.base import LLMClient
from src.llm.factory import get_available_client
from src.processing.embeddings import EmbeddingGenerator
from src.storage.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session.

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Dependency to get an embedding generator.

    Returns a cached singleton for efficiency.
    """
    return EmbeddingGenerator()


def get_llm() -> LLMClient | None:
    """
    Dependency to get an LLM client.

    Uses get_available_client() which tries LM Studio first,
    then falls back to DeepSeek.

    Returns None if no LLM is available.
    """
    return get_available_client()

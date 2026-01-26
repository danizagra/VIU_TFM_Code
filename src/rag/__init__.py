"""
RAG (Retrieval-Augmented Generation) module for journalism Q&A.

Components:
- retriever: Vector search over news articles
- generator: Response generation with citations
- engine: Orchestrator for the RAG pipeline
"""

from src.rag.engine import RAGEngine
from src.rag.generator import CitedResponseGenerator, RAGResponse, SourceCitation
from src.rag.retriever import NewsRetriever

__all__ = [
    "NewsRetriever",
    "CitedResponseGenerator",
    "RAGResponse",
    "SourceCitation",
    "RAGEngine",
]

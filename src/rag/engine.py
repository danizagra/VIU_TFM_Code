"""
RAG Engine - Orchestrates retrieval and generation.

Main entry point for the RAG system.
"""

import re
from typing import Optional

import structlog
from sqlalchemy.orm import Session

from src.llm.base import LLMClient
from src.llm.factory import get_available_client
from src.processing.embeddings import (
    EmbeddingGenerator,
    LMStudioEmbeddingGenerator,
    get_embedding_generator,
)
from src.rag.generator import CitedResponseGenerator, RAGResponse
from src.rag.retriever import NewsRetriever

logger = structlog.get_logger()


class RAGEngine:
    """
    Main RAG engine for journalism Q&A.

    Orchestrates the retrieval-augmented generation pipeline:
    1. Retrieve relevant articles via vector search
    2. (Optional) Rerank results for better accuracy
    3. Generate response with citations

    Usage:
        from src.storage.database import get_db

        with get_db() as db:
            engine = RAGEngine(db)
            response = engine.ask("Que esta pasando con la economia?")
            print(response.answer)
            for source in response.sources:
                print(f"[{source.index}] {source.title}")
    """

    def __init__(
        self,
        db: Session,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingGenerator | LMStudioEmbeddingGenerator] = None,
        use_reranking: bool = False,
    ):
        """
        Initialize the RAG engine.

        Args:
            db: Database session.
            llm_client: LLM client (auto-detected if not provided).
            embedder: Embedding generator (created if not provided).
            use_reranking: Whether to use LLM reranking (more accurate but slower).
        """
        self.db = db
        self.llm = llm_client or get_available_client()
        self.embedder = embedder or get_embedding_generator()
        self.use_reranking = use_reranking

        # Initialize components
        self.retriever = NewsRetriever(db, self.embedder)
        self.generator = (
            CitedResponseGenerator(self.llm) if self.llm else None
        )

        logger.info(
            "RAG engine initialized",
            llm_available=self.llm is not None,
            use_reranking=use_reranking,
        )

    def _rerank(
        self,
        query: str,
        articles: list,
        top_k: int = 5,
        max_candidates: int = 10,
    ) -> list:
        """
        Rerank articles using LLM in a single batch call.

        Uses retry logic to handle LM Studio/MLX segfaults, and
        max_tokens=300 to ensure reasoning models have enough
        token budget for both thinking and output.

        Args:
            query: User's question.
            articles: List of (Article, score) tuples.
            top_k: Number of articles to return.
            max_candidates: Maximum articles to rerank.

        Returns:
            Reranked list of (Article, score) tuples.
        """
        if not self.llm or not articles:
            return articles[:top_k]

        # Limit candidates to rerank
        candidates = articles[:max_candidates]

        logger.info("Reranking articles (batch)", candidates=len(candidates))

        # Build batch prompt with all articles
        articles_text = ""
        for i, (article, _) in enumerate(candidates, 1):
            articles_text += f"""
[{i}] {article.title}
    {(article.description or '')[:200]}
"""

        batch_prompt = f"""Evalua la relevancia de cada articulo para responder esta pregunta:

PREGUNTA: {query}

ARTICULOS:
{articles_text}

Para cada articulo, responde con su numero y puntuacion (0-10).
Formato: [numero]: puntuacion
Ejemplo:
[1]: 8
[2]: 3

Responde SOLO con las puntuaciones:"""

        # Retry once on failure (LM Studio MLX segfaults are transient)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(
                    batch_prompt,
                    max_tokens=300,
                    temperature=0
                )
                scores_text = response.content.strip()

                logger.info("Rerank LLM raw response", response=scores_text[:300])

                # Parse scores: try strict format first [1]: 8
                llm_scores = {}
                for match in re.finditer(r"\[(\d+)\]:\s*(\d+)", scores_text):
                    idx = int(match.group(1))
                    score = min(int(match.group(2)), 10) / 10.0
                    llm_scores[idx] = score

                # Fallback: try without brackets (1: 8 or 1 : 8)
                if not llm_scores:
                    for match in re.finditer(
                        r"^(\d+)\s*:\s*(\d+)", scores_text, re.MULTILINE
                    ):
                        idx = int(match.group(1))
                        score = min(int(match.group(2)), 10) / 10.0
                        llm_scores[idx] = score

                if not llm_scores:
                    logger.warning(
                        "No scores parsed from rerank response",
                        raw_response=scores_text[:300],
                    )

                # Combine scores
                scored = []
                for i, (article, original_score) in enumerate(candidates, 1):
                    llm_score = llm_scores.get(i, original_score)
                    combined_score = (original_score + llm_score) / 2
                    scored.append((article, combined_score))

                # Sort by combined score
                scored.sort(key=lambda x: x[1], reverse=True)

                logger.info(
                    "Reranking completed",
                    top_score=scored[0][1] if scored else 0,
                    scores_parsed=len(llm_scores),
                )

                return scored[:top_k]

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Rerank attempt failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    import time
                    time.sleep(3)
                    continue
                logger.warning(
                    "Batch rerank failed after retries, using original scores",
                    error=str(e),
                )
                return articles[:top_k]

        return articles[:top_k]

    def ask(
        self,
        question: str,
        max_sources: int = 5,
        category: Optional[str] = None,
        similarity_threshold: float = 0.5,
    ) -> RAGResponse:
        """
        Ask a question and get an answer with citations.

        Args:
            question: Natural language question.
            max_sources: Maximum number of sources to use.
            category: Optional category filter.
            similarity_threshold: Minimum similarity for retrieval.

        Returns:
            RAGResponse with answer, sources, and confidence.

        Raises:
            ValueError: If no LLM is available.
        """
        if not self.llm:
            return RAGResponse(
                answer="No hay un LLM disponible para generar respuestas. Configure LM Studio o DeepSeek.",
                sources=[],
                confidence=0.0,
            )

        logger.info(
            "Processing RAG query",
            question=question[:50],
            max_sources=max_sources,
            category=category,
        )

        # Step 1: Retrieve candidates
        over_fetch = 2 if self.use_reranking else 1
        candidates = self.retriever.search(
            query=question,
            limit=max_sources * over_fetch,
            threshold=similarity_threshold,
            category=category,
        )

        if not candidates:
            logger.warning("No articles found for query")
            return RAGResponse(
                answer="No encontre articulos relevantes para responder esta pregunta. Intenta con otra consulta o verifica que hay noticias en la base de datos.",
                sources=[],
                confidence=0.0,
            )

        # Step 2: Rerank if enabled
        if self.use_reranking:
            candidates = self._rerank(question, list(candidates), max_sources)
        else:
            candidates = list(candidates)[:max_sources]

        # Step 3: Generate response
        if not self.generator:
            return RAGResponse(
                answer="No hay generador disponible.",
                sources=[],
                confidence=0.0,
            )
        response = self.generator.generate(question, candidates)

        logger.info(
            "RAG query completed",
            sources_found=len(candidates),
            sources_cited=len(response.sources),
            confidence=response.confidence,
        )

        return response

    def get_context_only(
        self,
        question: str,
        max_sources: int = 5,
        category: Optional[str] = None,
    ) -> list:
        """
        Get relevant articles without generating a response.

        Useful for debugging or when you want to handle generation separately.

        Args:
            question: Natural language question.
            max_sources: Maximum number of sources.
            category: Optional category filter.

        Returns:
            List of (Article, similarity_score) tuples.
        """
        return list(
            self.retriever.search(
                query=question,
                limit=max_sources,
                category=category,
            )
        )

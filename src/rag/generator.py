"""
Response generator for RAG system with source citations.

Generates answers based on retrieved articles with proper citations.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import structlog

from src.llm.base import LLMClient
from src.storage.models import Article

logger = structlog.get_logger()


@dataclass
class SourceCitation:
    """A source citation for a RAG response."""

    index: int  # Citation number [1], [2], etc.
    title: str
    url: Optional[str]
    source_name: str
    published_at: Optional[datetime]
    relevance_score: float


@dataclass
class RAGResponse:
    """Response from the RAG system."""

    answer: str
    sources: list[SourceCitation]
    confidence: float


class CitedResponseGenerator:
    """
    Generates responses with source citations.

    Uses an LLM to generate answers based on provided articles,
    with mandatory source citations in [1], [2] format.

    Usage:
        generator = CitedResponseGenerator(llm_client)
        response = generator.generate(
            query="Que esta pasando con la economia?",
            articles=[(article1, 0.9), (article2, 0.8)],
        )
        print(response.answer)
        for source in response.sources:
            print(f"[{source.index}] {source.title}")
    """

    SYSTEM_PROMPT = """Eres un asistente periodistico experto que responde preguntas basandose UNICAMENTE en los articulos de noticias proporcionados.

REGLAS ESTRICTAS:
1. Responde SOLO con informacion de los articulos proporcionados
2. Cita las fuentes usando [1], [2], etc. correspondientes al numero del articulo
3. Si no hay informacion suficiente para responder, di "No tengo informacion suficiente sobre este tema en los articulos disponibles"
4. NO inventes datos ni especules
5. Responde en espanol de manera clara y concisa
6. Si hay informacion contradictoria entre fuentes, mencionalo

FORMATO:
- Usa parrafos cortos y claros
- Cita fuentes al final de cada afirmacion relevante
- Maximo 3-4 parrafos"""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the generator.

        Args:
            llm_client: LLM client for generation.
        """
        self.llm = llm_client

    def _build_context(
        self,
        articles: Sequence[tuple[Article, float]],
    ) -> tuple[str, list[SourceCitation]]:
        """
        Build context string from articles with citation indices.

        Args:
            articles: List of (Article, similarity_score) tuples.

        Returns:
            Tuple of (context_string, list of SourceCitation).
        """
        context_parts = []
        sources = []

        for i, (article, score) in enumerate(articles, start=1):
            # Build article text
            text_parts = [f"[{i}] {article.title}"]

            if article.description:
                text_parts.append(article.description)

            if article.content:
                # Limit content length
                content = article.content[:1000]
                if len(article.content) > 1000:
                    content += "..."
                text_parts.append(content)

            text_parts.append(f"Fuente: {article.source_name}")
            if article.published_at:
                text_parts.append(f"Fecha: {article.published_at.strftime('%Y-%m-%d')}")

            context_parts.append("\n".join(text_parts))

            # Create source citation
            sources.append(
                SourceCitation(
                    index=i,
                    title=article.title,
                    url=article.source_url,
                    source_name=article.source_name,
                    published_at=article.published_at,
                    relevance_score=score,
                )
            )

        return "\n\n---\n\n".join(context_parts), sources

    def _extract_used_citations(
        self,
        answer: str,
        all_sources: list[SourceCitation],
    ) -> list[SourceCitation]:
        """
        Extract only the citations that were actually used in the answer.

        Args:
            answer: The generated answer text.
            all_sources: All available sources.

        Returns:
            List of sources that were cited in the answer.
        """
        # Find all citation numbers in the answer [1], [2], etc.
        pattern = r"\[(\d+)\]"
        cited_indices = set(int(m) for m in re.findall(pattern, answer))

        # Filter to only used sources
        used_sources = [s for s in all_sources if s.index in cited_indices]

        return used_sources

    def _calculate_confidence(
        self,
        answer: str,
        sources: list[SourceCitation],
        used_sources: list[SourceCitation],
    ) -> float:
        """
        Calculate confidence score based on answer quality.

        Args:
            answer: The generated answer.
            sources: All available sources.
            used_sources: Sources that were cited.

        Returns:
            Confidence score between 0 and 1.
        """
        # Base confidence on several factors
        score = 0.5

        # More citations = higher confidence (up to a point)
        citation_ratio = len(used_sources) / max(len(sources), 1)
        score += min(citation_ratio * 0.2, 0.2)

        # Higher average relevance = higher confidence
        if used_sources:
            avg_relevance = sum(s.relevance_score for s in used_sources) / len(
                used_sources
            )
            score += avg_relevance * 0.2

        # Longer answers with citations = more confident
        if len(answer) > 200 and len(used_sources) >= 2:
            score += 0.1

        # Check for "no information" type responses
        no_info_phrases = [
            "no tengo informacion",
            "no hay informacion",
            "no puedo responder",
            "no encuentro",
        ]
        if any(phrase in answer.lower() for phrase in no_info_phrases):
            score = 0.2

        return min(score, 1.0)

    def generate(
        self,
        query: str,
        articles: Sequence[tuple[Article, float]],
    ) -> RAGResponse:
        """
        Generate a response with citations.

        Args:
            query: User's question.
            articles: List of (Article, similarity_score) tuples.

        Returns:
            RAGResponse with answer, sources, and confidence.
        """
        if not articles:
            return RAGResponse(
                answer="No tengo articulos disponibles para responder esta pregunta.",
                sources=[],
                confidence=0.0,
            )

        logger.info("Generating RAG response", query=query[:50], articles=len(articles))

        # Build context
        context, sources = self._build_context(articles)

        # Build prompt
        user_prompt = f"""ARTICULOS DISPONIBLES:

{context}

---

PREGUNTA: {query}

Responde basandote UNICAMENTE en los articulos anteriores, citando las fuentes con [1], [2], etc."""

        # Generate response
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for factual responses
            )
            answer = response.content.strip()
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return RAGResponse(
                answer=f"Error al generar respuesta: {str(e)}",
                sources=[],
                confidence=0.0,
            )

        # Extract used citations
        used_sources = self._extract_used_citations(answer, sources)

        # Calculate confidence
        confidence = self._calculate_confidence(answer, sources, used_sources)

        logger.info(
            "RAG response generated",
            answer_length=len(answer),
            sources_used=len(used_sources),
            confidence=confidence,
        )

        return RAGResponse(
            answer=answer,
            sources=used_sources if used_sources else sources[:3],  # Fallback to top 3
            confidence=confidence,
        )

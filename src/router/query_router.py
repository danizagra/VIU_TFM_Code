"""
LLM-based query router for deciding search strategy.

Analyzes user queries and decides whether to search locally (RAG),
externally (news APIs), or both (combined approach).
"""

import re
from dataclasses import dataclass

import structlog

from src.llm.base import LLMClient

logger = structlog.get_logger()

ROUTER_SYSTEM_PROMPT = """\
Eres un clasificador de consultas de noticias. Tu trabajo es decidir la mejor estrategia de búsqueda.

Responde ÚNICAMENTE con una de estas tres opciones seguida de una breve razón (máximo 15 palabras):

LOCAL_RAG - La consulta es sobre temas generales, históricos o ya cubiertos en nuestra base de datos.
EXTERNAL_SEARCH - La consulta pide noticias recientes, menciona "hoy", "ayer", "última hora", "ahora", o temas muy específicos/emergentes.
COMBINED - No está claro, o la consulta podría beneficiarse de ambas fuentes.

Ejemplos:
- "¿Qué es la inflación?" → LOCAL_RAG: tema general ya cubierto
- "¿Qué pasó hoy con Trump?" → EXTERNAL_SEARCH: pide información de hoy
- "Últimas noticias de economía" → EXTERNAL_SEARCH: pide lo más reciente
- "Noticias sobre cambio climático" → COMBINED: tema amplio, puede haber local y nuevo
- "¿Quién es el presidente de Colombia?" → LOCAL_RAG: información factual estable
- "Noticias de última hora" → EXTERNAL_SEARCH: pide lo más reciente
- "Explica la reforma tributaria" → LOCAL_RAG: tema analítico ya cubierto

Formato de respuesta: RUTA: razón breve"""

ROUTER_PROMPT_TEMPLATE = "Clasifica esta consulta: {query}"

# Pattern to extract route from LLM response
ROUTE_PATTERN = re.compile(
    r"\b(LOCAL_RAG|EXTERNAL_SEARCH|COMBINED)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    """Result of query routing decision."""

    route: str  # "LOCAL_RAG" | "EXTERNAL_SEARCH" | "COMBINED"
    reasoning: str  # Brief explanation from the LLM


class QueryRouter:
    """
    LLM-based query router that decides search strategy.

    Uses prompt-based routing (no function calling required)
    to classify queries into LOCAL_RAG, EXTERNAL_SEARCH, or COMBINED.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def route(self, query: str) -> RouteDecision:
        """
        Analyze a query and decide the best search strategy.

        Args:
            query: The user's question or search query.

        Returns:
            RouteDecision with route and reasoning.
            Falls back to COMBINED if parsing fails.
        """
        try:
            prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)
            response = self._llm.generate(
                prompt=prompt,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=60,
            )

            return self._parse_response(response.content)

        except Exception as e:
            logger.warning("Router LLM call failed, defaulting to COMBINED", error=str(e))
            return RouteDecision(route="COMBINED", reasoning=f"Fallback: error del router ({e})")

    def _parse_response(self, text: str) -> RouteDecision:
        """Parse LLM response to extract route decision."""
        match = ROUTE_PATTERN.search(text)

        if match:
            route = match.group(1).upper()
            # Extract reasoning: everything after the route keyword
            reasoning = text[match.end():].strip().lstrip(":").lstrip("-").strip()
            if not reasoning:
                reasoning = text.strip()
            # Limit reasoning length
            reasoning = reasoning[:100]
            return RouteDecision(route=route, reasoning=reasoning)

        logger.warning("Could not parse router response, defaulting to COMBINED", response=text[:100])
        return RouteDecision(route="COMBINED", reasoning=f"Fallback: respuesta no parseable")

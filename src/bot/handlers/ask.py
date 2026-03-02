"""
RAG question handler for free-form queries.
"""

import asyncio

import structlog
from aiogram import F, Router
from aiogram.types import Message

from src.agent.graph import run_journalist_agent
from src.bot.config import EMOJI, MAX_MESSAGE_LENGTH, MESSAGES
from src.bot.utils import escape_md
from src.llm.factory import get_available_client
from src.rag.engine import RAGEngine
from src.rag.generator import RAGResponse
from src.router import QueryRouter, RouteDecision
from src.storage.database import get_db

logger = structlog.get_logger()
router = Router(name="ask")


def _route_query(question: str) -> RouteDecision:
    """Route a query using the LLM router. Falls back to COMBINED on failure."""
    llm = get_available_client()
    if not llm:
        return RouteDecision(route="COMBINED", reasoning="No LLM disponible para routing")
    query_router = QueryRouter(llm)
    return query_router.route(question)


def _run_rag_query(question: str, max_sources: int = 5) -> RAGResponse:
    """Run RAG query synchronously. Creates its own DB session (thread-safe)."""
    with get_db() as db:
        engine = RAGEngine(db, use_reranking=True)
        return engine.ask(question=question, max_sources=max_sources)


def format_rag_response(answer: str, sources: list, confidence: float) -> str:
    """Format RAG response for Telegram."""
    lines = []

    # Answer
    lines.append(f"{EMOJI['robot']} *Respuesta:*\n")
    lines.append(escape_md(answer))

    # Confidence
    confidence_pct = int(confidence * 100)
    lines.append(f"\n\n{EMOJI['check']} _Confianza: {confidence_pct}%_")

    # Sources
    if sources:
        lines.append(f"\n\n{EMOJI['source']} *Fuentes:*")
        for src in sources:
            title = escape_md(src.title[:50]) if src.title else "Sin titulo"
            source_name = escape_md(src.source_name) if src.source_name else ""
            lines.append(f"\n\\[{src.index}\\] {title}")
            if source_name:
                lines.append(f" \\- _{source_name}_")

    return "\n".join(lines)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_question(message: Message) -> None:
    """
    Handle free-form questions using RAG.

    Any message that doesn't start with / is treated as a question.
    If local RAG confidence is low, searches external sources and retries.
    """
    question = message.text.strip()

    if len(question) < 3:
        return  # Ignore very short messages

    logger.info("Processing RAG question", user_id=message.from_user.id, question=question[:50])

    # Send "thinking" message
    thinking_msg = await message.answer(f"{EMOJI['robot']} {MESSAGES['thinking']}")

    try:
        # Step 1: Route the query using LLM
        route = await asyncio.to_thread(_route_query, question)
        logger.info(
            "Router decision",
            route=route.route,
            reasoning=route.reasoning,
            query=question[:50],
        )

        response = None

        if route.route == "LOCAL_RAG":
            # Only local RAG, no external fallback
            response = await asyncio.to_thread(_run_rag_query, question, 5)

        elif route.route == "EXTERNAL_SEARCH":
            # Go directly to external sources, then RAG
            await thinking_msg.edit_text(
                f"{EMOJI['robot']} Buscando en fuentes externas \\(NewsAPI, GNews\\)\\.\\.\\.",
                parse_mode="MarkdownV2",
            )

            try:
                agent_result = await asyncio.to_thread(
                    run_journalist_agent,
                    query=question,
                    max_articles=10,
                    sources=["newsapi", "gnews"],
                    use_persistence=True,
                )
                fetched = len(agent_result.get("raw_articles", []))
                logger.info("Agent completed for external search", fetched=fetched)

                if fetched > 0:
                    await thinking_msg.edit_text(
                        f"{EMOJI['check']} Encontre {fetched} articulos nuevos\\. Generando respuesta\\.\\.\\.",
                        parse_mode="MarkdownV2",
                    )
            except Exception as agent_err:
                logger.error("Agent failed during external search", error=str(agent_err))

            # RAG over whatever we have (new + existing)
            response = await asyncio.to_thread(_run_rag_query, question, 5)

        else:
            # COMBINED: local RAG first, external if confidence is low
            response = await asyncio.to_thread(_run_rag_query, question, 5)

            needs_external = response.confidence <= 0.3 or len(response.sources) == 0

            if needs_external:
                logger.info(
                    "Combined route: low RAG confidence, trying external sources",
                    confidence=response.confidence,
                    sources=len(response.sources),
                )

                try:
                    await thinking_msg.edit_text(
                        f"{EMOJI['robot']} No encontre suficiente informacion local\\.\n"
                        f"Buscando en fuentes externas \\(NewsAPI, GNews\\)\\.\\.\\.",
                        parse_mode="MarkdownV2",
                    )

                    agent_result = await asyncio.to_thread(
                        run_journalist_agent,
                        query=question,
                        max_articles=10,
                        sources=["newsapi", "gnews"],
                        use_persistence=True,
                    )

                    fetched = len(agent_result.get("raw_articles", []))
                    saved = len(agent_result.get("saved_article_ids", []))
                    logger.info("Agent completed for RAG", fetched=fetched, saved=saved)

                    if fetched > 0:
                        await thinking_msg.edit_text(
                            f"{EMOJI['check']} Encontre {fetched} articulos nuevos\\. Generando respuesta\\.\\.\\.",
                            parse_mode="MarkdownV2",
                        )

                        new_response = await asyncio.to_thread(_run_rag_query, question, 5)

                        if new_response.confidence > response.confidence:
                            response = new_response
                            logger.info(
                                "Using improved RAG response after external fetch",
                                confidence=response.confidence,
                                sources=len(response.sources),
                            )

                except Exception as agent_err:
                    logger.error("Agent failed during RAG fallback", error=str(agent_err))

        # Delete thinking message
        await thinking_msg.delete()

        if response.confidence <= 0.3 and not response.sources:
            await message.answer(
                f"{EMOJI['warning']} {MESSAGES['no_results']}\n\n"
                f"Intenta con otra pregunta o usa /buscar para explorar temas."
            )
            return

        # Format response
        formatted = format_rag_response(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
        )

        # Truncate if needed
        if len(formatted) > MAX_MESSAGE_LENGTH:
            formatted = formatted[:MAX_MESSAGE_LENGTH - 100] + "\n\n_\\.\\.\\. mensaje truncado_"

        await message.answer(
            formatted,
            parse_mode="MarkdownV2",
        )

        logger.info(
            "RAG response sent",
            user_id=message.from_user.id,
            sources=len(response.sources),
            confidence=response.confidence,
        )

    except Exception as e:
        error_str = str(e)
        logger.error("Error processing RAG question", error=error_str, question=question[:50])
        try:
            await thinking_msg.delete()
        except Exception:
            pass

        # Provide specific error messages for common issues
        if "Connection error" in error_str or "ConnectError" in error_str:
            await message.answer(
                f"{EMOJI['warning']} No se pudo conectar al servicio de embeddings "
                f"(LM Studio). Verifica que este encendido o reinicia el bot."
            )
        else:
            await message.answer(f"{EMOJI['warning']} {MESSAGES['error']}")

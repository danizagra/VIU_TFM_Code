"""
RAG question handler for free-form queries.
"""

import structlog
from aiogram import F, Router
from aiogram.types import Message

from src.bot.config import EMOJI, MAX_MESSAGE_LENGTH, MESSAGES
from src.bot.utils import escape_md
from src.rag.engine import RAGEngine
from src.storage.database import get_db

logger = structlog.get_logger()
router = Router(name="ask")


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
    """
    question = message.text.strip()

    if len(question) < 3:
        return  # Ignore very short messages

    logger.info("Processing RAG question", user_id=message.from_user.id, question=question[:50])

    # Send "thinking" message
    thinking_msg = await message.answer(f"{EMOJI['robot']} {MESSAGES['thinking']}")

    try:
        with get_db() as db:
            engine = RAGEngine(db, use_reranking=True)
            response = engine.ask(
                question=question,
                max_sources=5,
            )

        # Delete thinking message
        await thinking_msg.delete()

        if not response.sources and response.confidence == 0:
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
        logger.error("Error processing RAG question", error=str(e), question=question[:50])
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await message.answer(f"{EMOJI['warning']} {MESSAGES['error']}")

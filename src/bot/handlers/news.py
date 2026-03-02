"""
News command handlers.
"""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from src.bot.config import (
    DEFAULT_NEWS_LIMIT,
    EMOJI,
    HIGH_SIMILARITY_THRESHOLD,
    MAX_MESSAGE_LENGTH,
    MAX_NEWS_LIMIT,
    MESSAGES,
    MIN_ARTICLES_FOR_LOCAL_ONLY,
    MIN_RELEVANCE_THRESHOLD,
    VECTOR_SEARCH_THRESHOLD,
)
from src.bot.keyboards import article_detail_keyboard, news_pagination_keyboard
from src.bot.utils import escape_md
from src.storage.database import get_db
from src.storage.models import Article
from src.storage.repositories.article import ArticleRepository
from src.storage.repositories.processed import ProcessedArticleRepository
from src.storage.repositories.session import SessionRepository
from src.processing.embeddings import get_embedding_generator
from src.agent.graph import run_journalist_agent
from src.generation.summarizer import ArticleSummarizer
from src.llm.factory import get_available_client
from src.router import QueryRouter, RouteDecision

logger = structlog.get_logger()
router = Router(name="news")


def _is_truncated_content(text: str | None) -> bool:
    """
    Check if content is truncated by news APIs (NewsAPI/GNews).

    These APIs truncate content and append markers like "[+1234 chars]" or "[1234 chars]".

    Args:
        text: The content text to check

    Returns:
        True if content appears to be truncated
    """
    import re

    if not text:
        return False

    # Pattern matches "[+1234 chars]" or "[1234 chars]" at end of text
    truncation_pattern = r'\[\+?\d+\s*chars?\]$'
    return bool(re.search(truncation_pattern, text.strip()))


def _get_usable_content(article: Article) -> str | None:
    """
    Get the best available content for summarization.

    If content is truncated by news APIs, combines available content
    with description to maximize information for the LLM.

    Args:
        article: The article to extract content from

    Returns:
        Combined content string or None if insufficient
    """
    import re

    content = article.content
    description = article.description

    # If content is truncated, strip the truncation marker
    if content and _is_truncated_content(content):
        # Remove truncation marker
        content = re.sub(r'\s*\[\+?\d+\s*chars?\]$', '', content.strip())

    # Combine content and description for maximum context
    parts = []
    if content and len(content.strip()) > 20:
        parts.append(content.strip())
    if description and len(description.strip()) > 20:
        # Only add description if it adds new information
        if not content or description.strip() not in content:
            parts.append(description.strip())

    if not parts:
        return None

    return "\n\n".join(parts)


def _has_sufficient_content(article: Article, min_length: int = 100) -> bool:
    """
    Check if an article has sufficient content for summarization.

    Args:
        article: The article to check
        min_length: Minimum usable content length required (increased from 50)

    Returns:
        True if article has sufficient content, False otherwise
    """
    usable = _get_usable_content(article)
    if usable and len(usable) >= min_length:
        return True
    return False


def _is_bad_summary(summary: str | None) -> bool:
    """
    Check if a summary indicates the LLM couldn't generate a proper summary.

    These occur when articles don't have enough content.
    """
    if not summary:
        return True

    bad_phrases = [
        "no es posible",
        "no dispongo",
        "lo siento",
        "no puedo generar",
        "no tengo suficiente",
        "contenido insuficiente",
        "sin contar con",
        "información completa",
        "texto completo",
        "sin conocer el contenido",
        "sin el artículo completo",
        "no cuento con",
        "información suficiente",
        "artículo completo",
        "necesito más información",
        "no tengo acceso",
        "contenido completo",
    ]

    summary_lower = summary.lower()
    return any(phrase in summary_lower for phrase in bad_phrases)


def _normalize_text(text: str) -> str:
    """Normalize text by removing accents and converting to lowercase."""
    import unicodedata

    # Normalize to NFD (decomposed form) then remove combining characters (accents)
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return without_accents.lower()


def _extract_proper_nouns(query: str) -> list[str]:
    """
    Extract proper nouns (capitalized words) from a query.
    These are considered high-priority keywords that SHOULD match.

    Examples:
        "Seleccion Colombia de futbol" -> ["Seleccion", "Colombia"]
        "economia en 2026" -> []
        "Petro y la reforma" -> ["Petro"]
    """
    words = query.split()
    # A proper noun is a word that starts with uppercase and is not at the start
    # of a sentence (we assume queries don't start sentences, so first word counts too)
    proper_nouns = []
    for word in words:
        # Skip short words and common words that might be capitalized
        if len(word) < 3:
            continue
        # Check if it starts with uppercase
        if word[0].isupper():
            # Normalize but keep track it's a proper noun
            proper_nouns.append(_normalize_text(word))
    return proper_nouns


def _is_relevant_to_query(article: Article, query: str, min_keyword_match: int = 1) -> bool:
    """
    Check if an article is actually relevant to the query using keyword matching.

    This helps filter out false positives from vector similarity search.
    For example, "Petro" matching "perro" due to embedding similarity.

    IMPORTANT: If the query contains proper nouns (capitalized words like "Colombia",
    "Petro"), at least one proper noun must match for the article to be relevant.
    This prevents completely unrelated articles while allowing partial matches
    (e.g., "Anthropic y Trump" matches articles about Anthropic OR about Trump).

    Args:
        article: The article to check
        query: The search query
        min_keyword_match: Minimum keywords that must match

    Returns:
        True if article appears relevant to query
    """
    import re

    # Normalize query into keywords (words with 3+ chars)
    query_keywords = [_normalize_text(w) for w in query.split() if len(w) >= 3]

    # Extract proper nouns - these are HIGH priority
    proper_nouns = _extract_proper_nouns(query)

    if not query_keywords:
        return True  # No keywords to match, allow all

    # Build searchable text from article (keep original case for proper noun detection)
    searchable_original = ""
    if article.title:
        searchable_original += article.title + " "
    if article.description:
        searchable_original += article.description + " "
    if article.content:
        searchable_original += article.content[:1000]  # First 1000 chars of content

    # Normalized version for keyword matching (no accents, lowercase)
    searchable_normalized = _normalize_text(searchable_original)

    def keyword_matches(kw: str) -> bool:
        """Check if a single keyword matches in the article."""
        # Pattern 1: Exact word match (accent-insensitive)
        exact_pattern = rf'\b{re.escape(kw)}\b'
        if re.search(exact_pattern, searchable_normalized):
            return True

        # Pattern 2: Proper noun compound - keyword at start of capitalized word
        for m in re.finditer(rf'\b({re.escape(kw)}[a-z]*)\b', searchable_normalized):
            start_pos = m.start(1)
            if start_pos < len(searchable_original):
                if searchable_original[start_pos].isupper():
                    return True
        return False

    # Count total keyword matches
    matches = sum(1 for kw in query_keywords if keyword_matches(kw))

    # If query has proper nouns, at least one must match
    # This prevents completely unrelated articles from being included
    # while still allowing articles that cover part of the query
    # (e.g., "Anthropic y Donald Trump" should match articles about Anthropic OR Trump)
    if proper_nouns:
        proper_noun_matches = sum(1 for pn in proper_nouns if keyword_matches(pn))
        if proper_noun_matches == 0:
            return False  # No proper nouns matched at all - reject article

    return matches >= min_keyword_match


def _filter_relevant_articles(
    articles: list[tuple[Article, float]],
    query: str,
    min_similarity: float = MIN_RELEVANCE_THRESHOLD,
    high_similarity: float = HIGH_SIMILARITY_THRESHOLD,
) -> list[tuple[Article, float]]:
    """
    Filter articles to only include truly relevant ones.

    Uses OR logic (not AND) to avoid filtering out valid articles:
    - If article has keyword match AND similarity >= min_similarity, include it
    - If article has HIGH similarity (>= high_similarity), include even without keyword match
      (these are strong semantic matches that may use synonyms or related terms)

    The key insight: keyword matching is the primary filter for relevance,
    similarity score is secondary. High similarity without keywords is allowed
    because the embedding model may find semantically related content.

    Args:
        articles: List of (Article, similarity) tuples
        query: Search query
        min_similarity: Minimum similarity score to consider (default from config)
        high_similarity: Threshold for high confidence semantic matches (default from config)

    Returns:
        Filtered list of relevant articles
    """
    relevant = []
    for article, score in articles:
        # Skip if below minimum threshold
        if score < min_similarity:
            continue

        has_keyword = _is_relevant_to_query(article, query)

        # Include if:
        # 1. Has keyword match (primary criterion) - don't care about exact similarity
        # 2. High similarity without keyword (strong semantic match)
        if has_keyword:
            relevant.append((article, score))
        elif score >= high_similarity:
            # High similarity but no keyword - strong semantic match
            # Trust the embedding model for these high-confidence matches
            relevant.append((article, score))

    return relevant


def process_articles_without_summary(
    db,
    articles: list[tuple[Article, float]],
    query: str,
) -> list[dict]:
    """
    Generate summaries for articles that don't have them.

    Args:
        db: Database session
        articles: List of (Article, score) tuples
        query: Search query (for session tracking)

    Returns:
        List of result dicts with summaries
    """
    session_repo = SessionRepository(db)
    processed_repo = ProcessedArticleRepository(db)

    # Filter out articles without sufficient content
    valid_articles = [
        (article, score) for article, score in articles
        if _has_sufficient_content(article)
    ]

    skipped_count = len(articles) - len(valid_articles)
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} articles without sufficient content")

    if not valid_articles:
        logger.warning("No articles with sufficient content to process")
        return []

    # Get LLM client
    llm = get_available_client()
    if not llm:
        logger.warning("No LLM available for summary generation")
        return []

    summarizer = ArticleSummarizer(llm)

    # Create a session for this processing
    session = session_repo.create(query=f"bot_search: {query}")

    results = []
    processed_count = 0

    for article, score in valid_articles:
        # Check if already has a processed version
        existing = processed_repo.get_by_article_id(article.id)

        # Use existing summary only if it's a GOOD summary
        if existing and not _is_bad_summary(existing[-1].summary):
            processed = existing[-1]
            results.append({
                "title": article.title,
                "source_name": article.source_name,
                "published_at": article.published_at,
                "summary": processed.summary[:200] if processed.summary else None,
                "score": score,
            })
        else:
            # Delete bad summary if exists
            if existing:
                for bad in existing:
                    if _is_bad_summary(bad.summary):
                        db.delete(bad)
                db.commit()
            # Generate new summary
            try:
                # Get usable content (handles truncated content from APIs)
                usable_content = _get_usable_content(article)

                if not usable_content or len(usable_content) < 100:
                    logger.info(f"Skipping article with insufficient content: {article.title[:50]}...")
                    continue

                # Create RawArticle-like object for summarizer
                from src.connectors.base import RawArticle
                raw = RawArticle(
                    title=article.title,
                    content=usable_content,  # Use combined/cleaned content
                    description=article.description,
                    source_name=article.source_name,
                    source_url=article.source_url,
                )

                summary_result = summarizer.summarize_article(raw)
                summary = summary_result.summary

                # Save to processed_articles
                processed_repo.create(
                    article_id=article.id,
                    session_id=session.id,
                    summary=summary,
                    quality_score=0.7,  # Default score for on-demand processing
                )
                processed_count += 1

                results.append({
                    "title": article.title,
                    "source_name": article.source_name,
                    "published_at": article.published_at,
                    "summary": summary[:200] if summary else None,
                    "score": score,
                })

            except Exception as e:
                logger.warning(f"Failed to generate summary for article: {e}")
                # Still include article with description as fallback
                results.append({
                    "title": article.title,
                    "source_name": article.source_name,
                    "published_at": article.published_at,
                    "summary": article.description[:200] if article.description else None,
                    "score": score,
                })

    # Complete session
    session_repo.complete(session.id)
    db.commit()

    logger.info(f"Processed {processed_count} articles without summaries")

    return results


def format_article_preview(
    title: str,
    source_name: str,
    published_at: Optional[datetime],
    summary: Optional[str] = None,
) -> str:
    """Format article for Telegram message."""
    lines = [f"{EMOJI['news']} *{escape_md(title)}*"]

    if source_name:
        lines.append(f"{EMOJI['source']} {escape_md(source_name)}")

    if published_at:
        date_str = published_at.strftime("%d/%m/%Y %H:%M")
        lines.append(f"{EMOJI['calendar']} {date_str}")

    if summary:
        # Truncate summary if needed
        max_summary = 300
        if len(summary) > max_summary:
            summary = summary[:max_summary] + "..."
        lines.append(f"\n{escape_md(summary)}")

    return "\n".join(lines)


# Use centralized escape_md from src.bot.utils


@router.message(Command("ultimas"))
async def cmd_latest_news(message: Message, command: CommandObject) -> None:
    """
    Handle /ultimas command - show latest news.

    Usage: /ultimas [limit]
    """
    # Parse limit from command - only accepts a number
    limit = DEFAULT_NEWS_LIMIT
    if command.args:
        try:
            limit = max(1, int(command.args))
        except ValueError:
            await message.answer(
                f"{EMOJI['warning']} /ultimas solo acepta un numero\\.\n"
                f"Ejemplo: `/ultimas 10`\n\n"
                f"Para buscar por tema usa: `/buscar {escape_md(command.args)}`",
                parse_mode="MarkdownV2",
            )
            return

    await message.answer(f"{EMOJI['robot']} Buscando las ultimas noticias...")

    try:
        with get_db() as db:
            processed_repo = ProcessedArticleRepository(db)
            results = list(processed_repo.get_recent_summaries(limit=limit))

            if not results:
                await message.answer(MESSAGES["no_results"])
                return

            # Build response as list of article entries
            header = f"{EMOJI['news']} *Ultimas {len(results)} noticias:*\n"
            article_entries = []

            for i, (processed, article) in enumerate(results, 1):
                preview = format_article_preview(
                    title=article.title,
                    source_name=article.source_name,
                    published_at=article.published_at,
                    summary=processed.summary[:200] if processed.summary else None,
                )
                article_entries.append(f"\n{i}\\. {preview}\n")

        # Split into multiple messages at article boundaries
        messages = []
        current = header
        for entry in article_entries:
            if len(current) + len(entry) > MAX_MESSAGE_LENGTH - 50:
                messages.append(current)
                current = ""
            current += entry
        if current:
            messages.append(current)

        for msg in messages:
            await message.answer(
                msg,
                parse_mode="MarkdownV2",
            )

    except Exception as e:
        logger.error("Error fetching latest news", error=str(e))
        await message.answer(MESSAGES["error"])


def _route_search_query(query: str) -> RouteDecision:
    """Route a search query using the LLM router. Falls back to COMBINED on failure."""
    llm = get_available_client()
    if not llm:
        return RouteDecision(route="COMBINED", reasoning="No LLM disponible para routing")
    query_router = QueryRouter(llm)
    return query_router.route(query)


def _search_local_articles(
    query: str,
    limit: int = DEFAULT_NEWS_LIMIT,
    process_results: bool = True,
) -> tuple[list[dict], int]:
    """
    Search local DB for articles and optionally process them.

    Runs synchronously (meant to be called via asyncio.to_thread).
    Creates its own DB session for thread safety.

    Args:
        query: Search query.
        limit: Maximum number of results.
        process_results: Whether to generate summaries for found articles.

    Returns:
        (results, local_count)
    """
    embedder = get_embedding_generator()
    query_embedding = embedder.embed_single(query)

    with get_db() as db:
        article_repo = ArticleRepository(db)

        similar_articles = list(article_repo.find_similar(
            embedding=query_embedding,
            limit=limit * 2,
            threshold=VECTOR_SEARCH_THRESHOLD,
        ))

        if similar_articles:
            original_count = len(similar_articles)
            similar_articles = _filter_relevant_articles(
                similar_articles, query, min_similarity=MIN_RELEVANCE_THRESHOLD
            )
            filtered_out = original_count - len(similar_articles)
            if filtered_out > 0:
                logger.info(f"Filtered out {filtered_out} irrelevant articles", query=query)

        local_count = len(similar_articles)
        if local_count > 0:
            logger.info(f"Found {local_count} relevant articles in DB", query=query)

        results = []
        if process_results and local_count > 0:
            results = process_articles_without_summary(db, similar_articles[:limit], query)

    return results, local_count


def _research_after_agent(
    query: str,
    limit: int = DEFAULT_NEWS_LIMIT,
) -> list[dict]:
    """
    Re-search DB after agent fetch and process articles.

    Runs synchronously (meant to be called via asyncio.to_thread).
    Creates its own DB session for thread safety.

    Returns:
        List of result dicts with summaries.
    """
    embedder = get_embedding_generator()
    query_embedding = embedder.embed_single(query)

    with get_db() as db:
        article_repo = ArticleRepository(db)
        similar_articles = list(article_repo.find_similar(
            embedding=query_embedding,
            limit=limit * 2,
            threshold=VECTOR_SEARCH_THRESHOLD,
        ))
        if similar_articles:
            similar_articles = _filter_relevant_articles(
                similar_articles, query, min_similarity=MIN_RELEVANCE_THRESHOLD
            )
        if similar_articles:
            return process_articles_without_summary(db, similar_articles[:limit], query)
    return []


@router.message(Command("buscar"))
async def cmd_search_news(message: Message, command: CommandObject) -> None:
    """
    Handle /buscar command - search news by topic.

    Flow:
    1. Search for similar articles in DB (by embedding)
    2. If found but without summaries → generate summaries on-demand
    3. If NO articles found → fetch from external sources (agent)

    Usage: /buscar <tema>
    """
    if not command.args:
        await message.answer(
            f"{EMOJI['warning']} Uso: /buscar [tema]\n"
            f"Ejemplo: /buscar economia colombiana"
        )
        return

    query = command.args
    status_msg = await message.answer(f"{EMOJI['search']} Buscando noticias sobre: {query}...")

    # Use thresholds from config - these control precision/recall trade-off
    # VECTOR_SEARCH_THRESHOLD (0.40): Cast wide net initially
    # MIN_RELEVANCE_THRESHOLD (0.45): After keyword filtering
    # HIGH_SIMILARITY_THRESHOLD (0.65): Strong semantic match (no keyword needed)

    try:
        # Step 1: Route the query using LLM
        route = await asyncio.to_thread(_route_search_query, query)
        logger.info(
            "Router decision",
            route=route.route,
            reasoning=route.reasoning,
            query=query[:50],
        )

        results = []
        local_count = 0

        if route.route == "LOCAL_RAG":
            # Only search local DB
            results, local_count = await asyncio.to_thread(
                _search_local_articles, query, DEFAULT_NEWS_LIMIT, True
            )
            if local_count > 0:
                await status_msg.edit_text(
                    f"{EMOJI['robot']} Encontre {local_count} articulos relevantes\\. Procesados\\.",
                    parse_mode="MarkdownV2",
                )

        elif route.route == "EXTERNAL_SEARCH":
            # Go directly to external sources
            await status_msg.edit_text(
                f"{EMOJI['robot']} Buscando en fuentes externas \\(NewsAPI, GNews\\)\\.\\.\\.",
                parse_mode="MarkdownV2",
            )
            logger.info("Router chose external search", query=query)

            try:
                agent_result = await asyncio.to_thread(
                    run_journalist_agent,
                    query=query,
                    max_articles=15,
                    sources=["newsapi", "gnews"],
                    use_persistence=True,
                )
                fetched = len(agent_result.get("raw_articles", []))
                logger.info("Agent completed", fetched=fetched)

                if fetched > 0:
                    await status_msg.edit_text(
                        f"{EMOJI['check']} Descargue {fetched} articulos nuevos\\. Procesando resultados\\.\\.\\.",
                        parse_mode="MarkdownV2",
                    )
            except Exception as agent_err:
                logger.error("Agent failed", error=str(agent_err))

            # Search DB for combined results (existing + newly fetched)
            results = await asyncio.to_thread(
                _research_after_agent, query, DEFAULT_NEWS_LIMIT
            )

        else:
            # COMBINED: local first, external if insufficient
            results, local_count = await asyncio.to_thread(
                _search_local_articles, query, DEFAULT_NEWS_LIMIT, True
            )

            needs_external = local_count < MIN_ARTICLES_FOR_LOCAL_ONLY

            if not needs_external:
                await status_msg.edit_text(
                    f"{EMOJI['robot']} Encontre {local_count} articulos relevantes\\. Procesados\\.",
                    parse_mode="MarkdownV2",
                )

            if needs_external:
                if local_count == 0:
                    await status_msg.edit_text(
                        f"{EMOJI['robot']} No encontre noticias en la base de datos\\.\n"
                        f"Buscando en fuentes externas \\(NewsAPI, GNews\\)\\.\\.\\.",
                        parse_mode="MarkdownV2",
                    )
                    logger.info("Combined route: no local articles, running agent", query=query)
                else:
                    await status_msg.edit_text(
                        f"{EMOJI['robot']} Solo encontre {local_count} articulo\\(s\\) en la base de datos\\.\n"
                        f"Buscando mas en fuentes externas para mejor cobertura\\.\\.\\.",
                        parse_mode="MarkdownV2",
                    )
                    logger.info(f"Combined route: only {local_count} local articles, running agent", query=query)

                try:
                    agent_result = await asyncio.to_thread(
                        run_journalist_agent,
                        query=query,
                        max_articles=15,
                        sources=["newsapi", "gnews"],
                        use_persistence=True,
                    )
                    fetched = len(agent_result.get("raw_articles", []))
                    saved = len(agent_result.get("saved_article_ids", []))

                    logger.info("Agent completed", fetched=fetched, saved=saved)

                    if fetched == 0 and local_count == 0:
                        await status_msg.edit_text(
                            f"{EMOJI['warning']} No encontre noticias sobre '{escape_md(query)}' en ninguna fuente\\.",
                            parse_mode="MarkdownV2",
                        )
                        return
                    elif fetched > 0:
                        await status_msg.edit_text(
                            f"{EMOJI['check']} Descargue {fetched} articulos nuevos\\. Procesando resultados\\.\\.\\.",
                            parse_mode="MarkdownV2",
                        )

                except Exception as agent_err:
                    logger.error("Agent failed", error=str(agent_err))
                    if local_count == 0:
                        await status_msg.edit_text(
                            f"{EMOJI['warning']} Error buscando en fuentes externas: {escape_md(str(agent_err)[:100])}",
                            parse_mode="MarkdownV2",
                        )
                        return
                    logger.info("Agent failed but we have local articles, continuing", local_count=local_count)

                # Re-search to get combined results (local + newly fetched)
                results = await asyncio.to_thread(
                    _research_after_agent, query, DEFAULT_NEWS_LIMIT
                )

        # No results after everything
        if not results:
            await status_msg.edit_text(
                f"{EMOJI['warning']} No encontre noticias relevantes sobre '{escape_md(query)}'\\.",
                parse_mode="MarkdownV2",
            )
            return

        # Build response
        response_parts = [f"{EMOJI['search']} *Resultados para:* {escape_md(query)}\n"]

        for i, item in enumerate(results, 1):
            preview = format_article_preview(
                title=item["title"],
                source_name=item["source_name"],
                published_at=item["published_at"],
                summary=item["summary"],
            )
            score_pct = int(item["score"] * 100)
            response_parts.append(f"\n{i}\\. \\[{score_pct}%\\] {preview}\n")

        response = "\n".join(response_parts)

        if len(response) > MAX_MESSAGE_LENGTH:
            response = response[:MAX_MESSAGE_LENGTH - 100] + "\n\n_\\.\\.\\. mensaje truncado_"

        await status_msg.edit_text(
            response,
            parse_mode="MarkdownV2",
        )

    except Exception as e:
        logger.error("Error searching news", error=str(e), query=query)
        await message.answer(MESSAGES["error"])


@router.message(Command("digest"))
async def cmd_digest(message: Message) -> None:
    """
    Handle /digest command - daily news summary.
    """
    await message.answer(f"{EMOJI['robot']} Generando resumen del dia...")

    try:
        with get_db() as db:
            processed_repo = ProcessedArticleRepository(db)
            # Get today's processed articles
            results = list(processed_repo.get_recent_summaries(limit=10))

            if not results:
                await message.answer("No hay noticias procesadas hoy.")
                return

            # Build digest inside session context
            response_parts = [f"{EMOJI['news']} *Resumen del dia*\n"]
            response_parts.append(f"_Total: {len(results)} noticias procesadas_\n")

            for i, (processed, article) in enumerate(results, 1):
                title = escape_md(article.title)
                source = escape_md(article.source_name or "Desconocido")
                response_parts.append(f"\n{i}\\. *{title}*\n   {EMOJI['source']} {source}")

                if processed.summary:
                    summary = escape_md(processed.summary[:150])
                    response_parts.append(f"\n   {summary}\\.\\.\\.\n")

            response = "\n".join(response_parts)

        if len(response) > MAX_MESSAGE_LENGTH:
            response = response[:MAX_MESSAGE_LENGTH - 100] + "\n\n_\\.\\.\\. mensaje truncado_"

        await message.answer(
            response,
            parse_mode="MarkdownV2",
        )

    except Exception as e:
        logger.error("Error generating digest", error=str(e))
        await message.answer(MESSAGES["error"])


@router.message(Command("categorias"))
async def cmd_categories(message: Message) -> None:
    """
    Handle /categorias command - list available categories.
    """
    try:
        with get_db() as db:
            article_repo = ArticleRepository(db)
            # Get distinct categories
            from sqlalchemy import distinct, func
            from src.storage.models import Article

            categories = db.query(
                Article.category,
                func.count(Article.id).label('count')
            ).filter(
                Article.category.isnot(None)
            ).group_by(
                Article.category
            ).order_by(
                func.count(Article.id).desc()
            ).all()

        if not categories:
            await message.answer("No hay categorias disponibles.")
            return

        response_parts = [f"{EMOJI['news']} *Categorias disponibles:*\n"]
        for cat, count in categories:
            response_parts.append(f"\\- {escape_md(cat)} \\({count} articulos\\)")

        response_parts.append(f"\n\nUsa /buscar \\<categoria\\> para filtrar\\.")

        await message.answer(
            "\n".join(response_parts),
            parse_mode="MarkdownV2",
        )

    except Exception as e:
        logger.error("Error fetching categories", error=str(e))
        await message.answer(MESSAGES["error"])


@router.callback_query(F.data.startswith("news_page:"))
async def callback_news_page(callback: CallbackQuery) -> None:
    """Handle pagination callbacks."""
    offset = int(callback.data.split(":")[1])

    try:
        with get_db() as db:
            processed_repo = ProcessedArticleRepository(db)
            total = processed_repo.count()
            results = list(processed_repo.get_recent_summaries(
                limit=DEFAULT_NEWS_LIMIT + offset
            ))[offset:offset + DEFAULT_NEWS_LIMIT]

            if not results:
                await callback.answer("No hay mas noticias.")
                return

            # Build response inside session context
            response_parts = [f"{EMOJI['news']} *Noticias \\({offset + 1}\\-{offset + len(results)} de {total}\\):*\n"]

            for i, (processed, article) in enumerate(results, offset + 1):
                preview = format_article_preview(
                    title=article.title,
                    source_name=article.source_name,
                    published_at=article.published_at,
                )
                response_parts.append(f"\n{i}\\. {preview}\n")

            response = "\n".join(response_parts)
            keyboard = news_pagination_keyboard(offset, total, DEFAULT_NEWS_LIMIT)

        await callback.message.edit_text(
            response,
            parse_mode="MarkdownV2",
            reply_markup=keyboard,
        )
        await callback.answer()

    except Exception as e:
        logger.error("Error paginating news", error=str(e))
        await callback.answer("Error al cargar pagina.")

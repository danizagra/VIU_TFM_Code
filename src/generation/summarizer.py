"""
News article summarizer using LLM.

Generates concise summaries of news articles
following journalistic standards.
"""

from dataclasses import dataclass

from src.connectors.base import RawArticle
from src.generation.prompts.summary import get_summary_prompt
from src.llm.base import LLMClient


@dataclass
class SummaryResult:
    """Result of summarization."""

    original_title: str
    original_content: str
    summary: str
    token_count: int | None = None


def _clean_truncated_content(content: str | None) -> str | None:
    """
    Clean content that may be truncated by news APIs.

    NewsAPI and GNews truncate content with markers like "[+1234 chars]".
    This removes those markers to avoid confusing the LLM.

    Args:
        content: The content to clean

    Returns:
        Cleaned content without truncation markers
    """
    import re

    if not content:
        return None

    # Remove truncation markers like "[+1234 chars]" or "[1234 chars]"
    cleaned = re.sub(r'\s*\[\+?\d+\s*chars?\]$', '', content.strip())
    # Also handle "... [1234 chars]" format
    cleaned = re.sub(r'\.\.\.\s*\[\+?\d+\s*chars?\]$', '...', cleaned)

    return cleaned if cleaned else None


class ArticleSummarizer:
    """
    Generates summaries for news articles using LLM.

    Attributes:
        llm_client: LLM client for generation.
        use_few_shot: Whether to include few-shot examples in prompts.
        max_content_length: Maximum content length to send to LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
        max_content_length: int = 4000,
    ):
        """
        Initialize the summarizer.

        Args:
            llm_client: LLM client for generation.
            use_few_shot: Whether to use few-shot examples.
            max_content_length: Max chars of content to process.
        """
        self.llm_client = llm_client
        self.use_few_shot = use_few_shot
        self.max_content_length = max_content_length

    def summarize(
        self,
        title: str,
        content: str,
        use_few_shot: bool | None = None,
    ) -> SummaryResult:
        """
        Generate a summary for an article.

        Args:
            title: Article title.
            content: Article content/body.
            use_few_shot: Override default few-shot setting.

        Returns:
            SummaryResult with the generated summary.
        """
        # Truncate content if too long
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "..."

        # Build prompt
        few_shot = use_few_shot if use_few_shot is not None else self.use_few_shot
        messages = get_summary_prompt(
            title=title,
            content=content,
            use_few_shot=few_shot,
        )

        # Generate summary
        response = self.llm_client.chat(messages)

        # Clean up response
        summary = response.content.strip()

        return SummaryResult(
            original_title=title,
            original_content=content,
            summary=summary,
        )

    def summarize_article(
        self,
        article: RawArticle,
        use_few_shot: bool | None = None,
    ) -> SummaryResult:
        """
        Generate a summary for a RawArticle.

        Args:
            article: RawArticle to summarize.
            use_few_shot: Override default few-shot setting.

        Returns:
            SummaryResult with the generated summary.
        """
        # Clean truncated content from news APIs
        content = _clean_truncated_content(article.content)
        description = _clean_truncated_content(article.description)

        # Combine content and description for maximum context
        # This helps when content is truncated by the API
        parts = []
        if content and len(content) > 20:
            parts.append(content)
        if description and len(description) > 20:
            # Only add description if it provides new information
            if not content or description not in content:
                parts.append(description)

        combined_content = "\n\n".join(parts) if parts else ""

        return self.summarize(
            title=article.title,
            content=combined_content,
            use_few_shot=use_few_shot,
        )

    def summarize_batch(
        self,
        articles: list[RawArticle],
        use_few_shot: bool | None = None,
    ) -> list[SummaryResult]:
        """
        Generate summaries for multiple articles.

        Args:
            articles: List of RawArticles to summarize.
            use_few_shot: Override default few-shot setting.

        Returns:
            List of SummaryResults.
        """
        results = []
        for article in articles:
            result = self.summarize_article(article, use_few_shot)
            results.append(result)
        return results

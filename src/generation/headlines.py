"""
News headline generator using LLM.

Generates multiple headline variations with different styles
(informative, engaging, SEO-optimized).
"""

from dataclasses import dataclass

from src.connectors.base import RawArticle
from src.generation.prompts.headline import get_headline_prompt, parse_headlines
from src.llm.base import LLMClient


@dataclass
class HeadlineResult:
    """Result of headline generation."""

    original_title: str
    informativo: str
    engagement: str
    seo: str
    raw_response: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "original": self.original_title,
            "informativo": self.informativo,
            "engagement": self.engagement,
            "seo": self.seo,
        }

    def all_headlines(self) -> list[str]:
        """Return all generated headlines as a list."""
        return [self.informativo, self.engagement, self.seo]


class HeadlineGenerator:
    """
    Generates alternative headlines for news articles.

    Produces three headline variations:
    - Informativo: Objective, traditional journalism style
    - Engagement: Attention-grabbing without clickbait
    - SEO: Optimized for search engines with keywords
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
    ):
        """
        Initialize the headline generator.

        Args:
            llm_client: LLM client for generation.
            use_few_shot: Whether to use few-shot examples.
        """
        self.llm_client = llm_client
        self.use_few_shot = use_few_shot

    def generate(
        self,
        title: str,
        summary: str,
        use_few_shot: bool | None = None,
    ) -> HeadlineResult:
        """
        Generate alternative headlines for an article.

        Args:
            title: Original article title.
            summary: Article summary or description.
            use_few_shot: Override default few-shot setting.

        Returns:
            HeadlineResult with three headline variations.
        """
        # Build prompt
        few_shot = use_few_shot if use_few_shot is not None else self.use_few_shot
        messages = get_headline_prompt(
            title=title,
            summary=summary,
            use_few_shot=few_shot,
        )

        # Generate headlines
        response = self.llm_client.chat(messages)

        # Parse response
        headlines = parse_headlines(response.content)

        return HeadlineResult(
            original_title=title,
            informativo=headlines.get("informativo", ""),
            engagement=headlines.get("engagement", ""),
            seo=headlines.get("seo", ""),
            raw_response=response.content,
        )

    def generate_for_article(
        self,
        article: RawArticle,
        summary: str | None = None,
        use_few_shot: bool | None = None,
    ) -> HeadlineResult:
        """
        Generate headlines for a RawArticle.

        Args:
            article: RawArticle to generate headlines for.
            summary: Optional pre-generated summary. Uses description if not provided.
            use_few_shot: Override default few-shot setting.

        Returns:
            HeadlineResult with three headline variations.
        """
        # Use provided summary, or fall back to description
        article_summary = summary or article.description or ""

        return self.generate(
            title=article.title,
            summary=article_summary,
            use_few_shot=use_few_shot,
        )

    def generate_batch(
        self,
        articles: list[tuple[str, str]],
        use_few_shot: bool | None = None,
    ) -> list[HeadlineResult]:
        """
        Generate headlines for multiple articles.

        Args:
            articles: List of (title, summary) tuples.
            use_few_shot: Override default few-shot setting.

        Returns:
            List of HeadlineResults.
        """
        results = []
        for title, summary in articles:
            result = self.generate(title, summary, use_few_shot)
            results.append(result)
        return results

"""
Journalistic angle generator using LLM.

Suggests different perspectives and approaches
for covering a news story or topic cluster.
"""

from dataclasses import dataclass, field

from src.connectors.base import RawArticle
from src.generation.prompts.angle import get_angle_prompt, parse_angles
from src.llm.base import LLMClient


@dataclass
class Angle:
    """A single journalistic angle."""

    tipo: str
    enfoque: str
    pregunta_clave: str
    fuentes: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "tipo": self.tipo,
            "enfoque": self.enfoque,
            "pregunta_clave": self.pregunta_clave,
            "fuentes": self.fuentes,
        }


@dataclass
class AngleResult:
    """Result of angle generation."""

    original_title: str
    original_summary: str
    angles: list[Angle] = field(default_factory=list)
    related_articles: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original_title": self.original_title,
            "original_summary": self.original_summary,
            "angles": [a.to_dict() for a in self.angles],
            "related_articles": self.related_articles,
        }


class AngleGenerator:
    """
    Generates journalistic angles for news coverage.

    Suggests different perspectives for deeper coverage:
    - HUMANO: Human interest stories
    - EXPLICATIVO: Context and explanations
    - DATOS: Data-driven analysis
    - INVESTIGATIVO: Investigative angles
    - LOCAL: Local impact
    - PROSPECTIVO: Future outlook
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
    ):
        """
        Initialize the angle generator.

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
        related_articles: list[str] | None = None,
        use_few_shot: bool | None = None,
    ) -> AngleResult:
        """
        Generate journalistic angles for an article.

        Args:
            title: Article title.
            summary: Article summary.
            related_articles: Summaries of related articles (cluster context).
            use_few_shot: Override default few-shot setting.

        Returns:
            AngleResult with suggested angles.
        """
        # Build prompt
        few_shot = use_few_shot if use_few_shot is not None else self.use_few_shot
        messages = get_angle_prompt(
            title=title,
            summary=summary,
            related_articles=related_articles,
            use_few_shot=few_shot,
        )

        # Generate angles
        response = self.llm_client.chat(messages)

        # Parse response
        parsed_angles = parse_angles(response.content)

        # Convert to Angle objects
        angles = []
        for angle_dict in parsed_angles:
            angle = Angle(
                tipo=angle_dict.get("tipo", ""),
                enfoque=angle_dict.get("enfoque", ""),
                pregunta_clave=angle_dict.get("pregunta_clave", ""),
                fuentes=angle_dict.get("fuentes", ""),
            )
            angles.append(angle)

        return AngleResult(
            original_title=title,
            original_summary=summary,
            angles=angles,
            related_articles=related_articles or [],
            raw_response=response.content,
        )

    def generate_for_article(
        self,
        article: RawArticle,
        summary: str | None = None,
        related_articles: list[str] | None = None,
        use_few_shot: bool | None = None,
    ) -> AngleResult:
        """
        Generate angles for a RawArticle.

        Args:
            article: RawArticle to generate angles for.
            summary: Optional pre-generated summary.
            related_articles: Summaries of related articles.
            use_few_shot: Override default few-shot setting.

        Returns:
            AngleResult with suggested angles.
        """
        article_summary = summary or article.description or ""

        return self.generate(
            title=article.title,
            summary=article_summary,
            related_articles=related_articles,
            use_few_shot=use_few_shot,
        )

    def generate_for_cluster(
        self,
        articles: list[RawArticle],
        summaries: list[str] | None = None,
        use_few_shot: bool | None = None,
    ) -> AngleResult:
        """
        Generate angles for a cluster of related articles.

        Uses the first article as the main one and others as context.

        Args:
            articles: List of related articles (first is main).
            summaries: Optional pre-generated summaries for each article.
            use_few_shot: Override default few-shot setting.

        Returns:
            AngleResult with suggested angles for the cluster.
        """
        if not articles:
            raise ValueError("At least one article is required")

        main_article = articles[0]
        main_summary = summaries[0] if summaries else main_article.description or ""

        # Build context from related articles
        related_summaries = []
        for i, article in enumerate(articles[1:], start=1):
            if summaries and i < len(summaries):
                related_summaries.append(summaries[i])
            else:
                related_summaries.append(article.description or article.title)

        return self.generate(
            title=main_article.title,
            summary=main_summary,
            related_articles=related_summaries if related_summaries else None,
            use_few_shot=use_few_shot,
        )

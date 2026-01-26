"""
Quality checker for generated content.

Validates summaries, headlines, and angles
for length, language, and basic quality criteria.
"""

from dataclasses import dataclass, field
from enum import Enum


class QualityLevel(Enum):
    """Quality assessment levels."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class QualityIssue:
    """A single quality issue."""

    code: str
    message: str
    level: QualityLevel


@dataclass
class QualityResult:
    """Result of quality check."""

    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0

    def add_issue(self, code: str, message: str, level: QualityLevel) -> None:
        """Add an issue to the result."""
        self.issues.append(QualityIssue(code, message, level))
        if level == QualityLevel.FAIL:
            self.passed = False
            self.score -= 0.3
        elif level == QualityLevel.WARNING:
            self.score -= 0.1
        self.score = max(0.0, self.score)


class QualityChecker:
    """
    Checks quality of generated content.

    Validates:
    - Length constraints
    - Language consistency
    - Content presence
    - Format compliance
    """

    def __init__(
        self,
        min_summary_words: int = 20,
        max_summary_words: int = 150,
        min_headline_words: int = 5,
        max_headline_words: int = 20,
        max_headline_chars: int = 100,
    ):
        """
        Initialize the quality checker.

        Args:
            min_summary_words: Minimum words for summaries.
            max_summary_words: Maximum words for summaries.
            min_headline_words: Minimum words for headlines.
            max_headline_words: Maximum words for headlines.
            max_headline_chars: Maximum characters for headlines.
        """
        self.min_summary_words = min_summary_words
        self.max_summary_words = max_summary_words
        self.min_headline_words = min_headline_words
        self.max_headline_words = max_headline_words
        self.max_headline_chars = max_headline_chars

    def check_summary(self, summary: str, original_content: str = "") -> QualityResult:
        """
        Check quality of a generated summary.

        Args:
            summary: Generated summary text.
            original_content: Original article content for reference.

        Returns:
            QualityResult with issues found.
        """
        result = QualityResult(passed=True)

        # Check for empty content
        if not summary or not summary.strip():
            result.add_issue(
                "EMPTY_SUMMARY",
                "El resumen está vacío",
                QualityLevel.FAIL,
            )
            return result

        words = summary.split()
        word_count = len(words)

        # Check minimum length
        if word_count < self.min_summary_words:
            result.add_issue(
                "SUMMARY_TOO_SHORT",
                f"Resumen muy corto ({word_count} palabras, mínimo {self.min_summary_words})",
                QualityLevel.WARNING,
            )

        # Check maximum length
        if word_count > self.max_summary_words:
            result.add_issue(
                "SUMMARY_TOO_LONG",
                f"Resumen muy largo ({word_count} palabras, máximo {self.max_summary_words})",
                QualityLevel.WARNING,
            )

        # Check for meta-commentary
        meta_phrases = [
            "el artículo habla de",
            "en resumen",
            "el texto menciona",
            "se puede concluir",
            "el autor dice",
        ]
        summary_lower = summary.lower()
        for phrase in meta_phrases:
            if phrase in summary_lower:
                result.add_issue(
                    "META_COMMENTARY",
                    f"El resumen contiene meta-comentario: '{phrase}'",
                    QualityLevel.WARNING,
                )
                break

        # Check for repetition with original title (if available)
        if original_content:
            # Simple check: summary shouldn't be identical to content start
            content_start = original_content[:200].lower()
            if summary.lower().startswith(content_start[:100]):
                result.add_issue(
                    "COPY_PASTE",
                    "El resumen parece ser copia directa del contenido original",
                    QualityLevel.FAIL,
                )

        return result

    def check_headline(self, headline: str) -> QualityResult:
        """
        Check quality of a generated headline.

        Args:
            headline: Generated headline text.

        Returns:
            QualityResult with issues found.
        """
        result = QualityResult(passed=True)

        # Check for empty content
        if not headline or not headline.strip():
            result.add_issue(
                "EMPTY_HEADLINE",
                "El titular está vacío",
                QualityLevel.FAIL,
            )
            return result

        words = headline.split()
        word_count = len(words)
        char_count = len(headline)

        # Check minimum length
        if word_count < self.min_headline_words:
            result.add_issue(
                "HEADLINE_TOO_SHORT",
                f"Titular muy corto ({word_count} palabras)",
                QualityLevel.WARNING,
            )

        # Check maximum length
        if word_count > self.max_headline_words:
            result.add_issue(
                "HEADLINE_TOO_LONG",
                f"Titular muy largo ({word_count} palabras, máximo {self.max_headline_words})",
                QualityLevel.WARNING,
            )

        # Check character limit
        if char_count > self.max_headline_chars:
            result.add_issue(
                "HEADLINE_CHARS_EXCEEDED",
                f"Titular excede límite de caracteres ({char_count}, máximo {self.max_headline_chars})",
                QualityLevel.WARNING,
            )

        # Check for clickbait patterns
        clickbait_patterns = [
            "no vas a creer",
            "increíble",
            "impactante",
            "esto es lo que",
            "te sorprenderá",
            "la razón te dejará",
        ]
        headline_lower = headline.lower()
        for pattern in clickbait_patterns:
            if pattern in headline_lower:
                result.add_issue(
                    "CLICKBAIT",
                    f"El titular contiene patrón clickbait: '{pattern}'",
                    QualityLevel.WARNING,
                )
                break

        return result

    def check_headlines_set(
        self,
        informativo: str,
        engagement: str,
        seo: str,
    ) -> dict[str, QualityResult]:
        """
        Check quality of a complete headlines set.

        Args:
            informativo: Informative headline.
            engagement: Engagement headline.
            seo: SEO-optimized headline.

        Returns:
            Dict mapping headline type to QualityResult.
        """
        return {
            "informativo": self.check_headline(informativo),
            "engagement": self.check_headline(engagement),
            "seo": self.check_headline(seo),
        }

    def check_angles(self, angles: list[dict]) -> QualityResult:
        """
        Check quality of generated angles.

        Args:
            angles: List of angle dictionaries.

        Returns:
            QualityResult with issues found.
        """
        result = QualityResult(passed=True)

        # Check count
        if len(angles) < 3:
            result.add_issue(
                "INSUFFICIENT_ANGLES",
                f"Se generaron solo {len(angles)} ángulos (se esperaban 3)",
                QualityLevel.WARNING,
            )

        # Check each angle has required fields
        required_fields = ["tipo", "enfoque", "pregunta_clave", "fuentes"]
        for i, angle in enumerate(angles):
            for field in required_fields:
                if not angle.get(field):
                    result.add_issue(
                        "MISSING_FIELD",
                        f"Ángulo {i + 1} no tiene '{field}'",
                        QualityLevel.WARNING,
                    )

        # Check for variety in types
        tipos = [a.get("tipo", "").upper() for a in angles if a.get("tipo")]
        if len(tipos) != len(set(tipos)):
            result.add_issue(
                "DUPLICATE_TYPES",
                "Hay tipos de ángulo repetidos",
                QualityLevel.WARNING,
            )

        return result

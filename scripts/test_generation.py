#!/usr/bin/env python3
"""
Script to test the content generation pipeline.

Tests summarization, headline generation, angle suggestions,
and quality checking using the LLM.

Usage:
    poetry run python scripts/test_generation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_llm_connection():
    """Test LLM connection."""
    print("=" * 60)
    print("Testing LLM Connection")
    print("=" * 60)

    from src.llm.factory import get_llm_client

    try:
        client = get_llm_client()
        print(f"Provider: {client.__class__.__name__}")

        # Simple test
        response = client.chat([
            {"role": "user", "content": "Responde solo con 'OK' si puedes leer esto."}
        ])
        print(f"Response: {response.content.strip()}")
        print("✓ LLM connection successful")
        return True, client

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_summarizer(client):
    """Test article summarization."""
    print("\n" + "=" * 60)
    print("Testing Summarization")
    print("=" * 60)

    from src.generation.summarizer import ArticleSummarizer

    try:
        summarizer = ArticleSummarizer(client, use_few_shot=True)

        # Test article
        title = "Colombia anuncia inversión de $5.000 millones en infraestructura"
        content = """El presidente Gustavo Petro anunció hoy un ambicioso plan de infraestructura
        valorado en $5.000 millones de dólares. El plan incluye la construcción de 500 kilómetros
        de nuevas carreteras y la modernización de 20 aeropuertos en todo el país. Según el mandatario,
        esta inversión generará aproximadamente 100.000 nuevos empleos en los próximos cinco años.
        La oposición ha cuestionado las fuentes de financiamiento del proyecto, señalando que
        el gobierno no ha explicado de dónde saldrán los recursos. El ministro de Hacienda
        afirmó que parte de la inversión vendrá de créditos internacionales y otra parte
        del presupuesto nacional."""

        print(f"\nTítulo: {title}")
        print(f"Contenido: {len(content)} caracteres")
        print("\nGenerando resumen...")

        result = summarizer.summarize(title, content)

        print(f"\n✓ Resumen generado:")
        print(f"  {result.summary}")

        return True, result.summary

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_headlines(client, summary: str):
    """Test headline generation."""
    print("\n" + "=" * 60)
    print("Testing Headline Generation")
    print("=" * 60)

    from src.generation.headlines import HeadlineGenerator

    try:
        generator = HeadlineGenerator(client, use_few_shot=True)

        title = "Colombia anuncia inversión de $5.000 millones en infraestructura"

        print(f"\nTítulo original: {title}")
        print(f"Resumen: {summary[:100]}...")
        print("\nGenerando titulares...")

        result = generator.generate(title, summary)

        print(f"\n✓ Titulares generados:")
        print(f"  INFORMATIVO: {result.informativo}")
        print(f"  ENGAGEMENT:  {result.engagement}")
        print(f"  SEO:         {result.seo}")

        return True, result

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_angles(client, summary: str):
    """Test angle generation."""
    print("\n" + "=" * 60)
    print("Testing Angle Generation")
    print("=" * 60)

    from src.generation.angles import AngleGenerator

    try:
        generator = AngleGenerator(client, use_few_shot=True)

        title = "Colombia anuncia inversión de $5.000 millones en infraestructura"

        print(f"\nTítulo: {title}")
        print(f"Resumen: {summary[:100]}...")
        print("\nGenerando ángulos periodísticos...")

        result = generator.generate(title, summary)

        print(f"\n✓ Ángulos generados: {len(result.angles)}")
        for i, angle in enumerate(result.angles, 1):
            print(f"\n  ÁNGULO {i}:")
            print(f"    Tipo: {angle.tipo}")
            print(f"    Enfoque: {angle.enfoque}")
            print(f"    Pregunta: {angle.pregunta_clave}")
            print(f"    Fuentes: {angle.fuentes}")

        return True, result

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_quality_checker(summary: str, headlines):
    """Test quality checker."""
    print("\n" + "=" * 60)
    print("Testing Quality Checker")
    print("=" * 60)

    from src.generation.quality_checker import QualityChecker

    try:
        checker = QualityChecker()

        # Check summary
        print("\n1. Verificando resumen...")
        summary_result = checker.check_summary(summary)
        print(f"   Pasó: {summary_result.passed}")
        print(f"   Score: {summary_result.score:.2f}")
        if summary_result.issues:
            for issue in summary_result.issues:
                print(f"   - [{issue.level.value}] {issue.message}")

        # Check headlines
        print("\n2. Verificando titulares...")
        headline_results = checker.check_headlines_set(
            headlines.informativo,
            headlines.engagement,
            headlines.seo,
        )
        for htype, hresult in headline_results.items():
            status = "✓" if hresult.passed else "✗"
            print(f"   {status} {htype}: score={hresult.score:.2f}")
            for issue in hresult.issues:
                print(f"      [{issue.level.value}] {issue.message}")

        print("\n✓ Quality check complete")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_articles(client):
    """Test generation with real fetched articles."""
    print("\n" + "=" * 60)
    print("Testing with Real Articles")
    print("=" * 60)

    try:
        from src.connectors import create_default_aggregator
        from src.generation.summarizer import ArticleSummarizer
        from src.generation.headlines import HeadlineGenerator
        from src.generation.quality_checker import QualityChecker

        # Fetch articles
        print("\n1. Fetching articles...")
        aggregator = create_default_aggregator(
            include_newsapi=False,
            include_gnews=False,
            include_colombian_rss=True,
        )
        articles = aggregator.fetch_all(max_results=3)
        print(f"   Fetched: {len(articles)} articles")

        if not articles:
            print("   ⚠ No articles fetched, skipping test")
            return True

        # Process first article
        article = articles[0]
        print(f"\n2. Processing: {article.title[:60]}...")

        # Summarize
        summarizer = ArticleSummarizer(client, use_few_shot=False)
        summary_result = summarizer.summarize_article(article)
        print(f"\n   Summary: {summary_result.summary[:150]}...")

        # Generate headlines
        generator = HeadlineGenerator(client, use_few_shot=False)
        headline_result = generator.generate(
            article.title,
            summary_result.summary,
        )
        print(f"\n   Headlines:")
        print(f"   - {headline_result.informativo}")
        print(f"   - {headline_result.engagement}")

        # Quality check
        checker = QualityChecker()
        quality = checker.check_summary(summary_result.summary)
        print(f"\n   Quality: score={quality.score:.2f}, passed={quality.passed}")

        print("\n✓ Real article processing complete")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all generation tests."""
    print("\n📝 Content Generation Test Suite\n")

    results = []

    # Test 1: LLM Connection
    success, client = test_llm_connection()
    results.append(("LLM Connection", success))

    if not success or not client:
        print("\n⚠ Cannot continue without LLM connection")
        return 1

    # Test 2: Summarization
    success, summary = test_summarizer(client)
    results.append(("Summarization", success))

    if not success or not summary:
        summary = "El gobierno colombiano presentó un plan de infraestructura de $5.000 millones."

    # Test 3: Headlines
    success, headlines = test_headlines(client, summary)
    results.append(("Headlines", success))

    # Test 4: Angles
    success, _ = test_angles(client, summary)
    results.append(("Angles", success))

    # Test 5: Quality Checker
    if headlines:
        success = test_quality_checker(summary, headlines)
        results.append(("Quality Checker", success))

    # Test 6: Real Articles
    success = test_with_real_articles(client)
    results.append(("Real Articles", success))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

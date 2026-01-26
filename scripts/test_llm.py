#!/usr/bin/env python3
"""
Script to test LLM connection and basic generation.

Usage:
    poetry run python scripts/test_llm.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import get_llm_client, LMStudioClient


def test_lm_studio_connection():
    """Test connection to LM Studio."""
    print("=" * 60)
    print("Testing LM Studio Connection")
    print("=" * 60)

    try:
        client = LMStudioClient()
        print(f"Base URL: {client._base_url}")
        print(f"Model: {client.model_name}")

        # Check availability
        print("\nChecking availability...")
        if client.is_available():
            print("✓ LM Studio is running and responsive")

            # List models
            models = client.list_models()
            if models:
                print(f"\nAvailable models:")
                for model in models:
                    print(f"  - {model}")
        else:
            print("✗ LM Studio is not responding")
            print("  Make sure LM Studio is running and the server is started.")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True


def test_generation():
    """Test basic text generation."""
    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)

    try:
        client = get_llm_client()
        print(f"Using provider: {client.model_name}")

        # Simple test prompt
        prompt = "What is 2 + 2? Answer in one word."
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")

        response = client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=50
        )

        print(f"\nResponse: {response.content}")
        print(f"Model: {response.model}")
        if response.total_tokens:
            print(f"Tokens: {response.total_tokens}")

        return True

    except ConnectionError as e:
        print(f"✗ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_summarization():
    """Test news summarization capability."""
    print("\n" + "=" * 60)
    print("Testing News Summarization")
    print("=" * 60)

    sample_article = """
    The Colombian government announced today a new infrastructure plan
    worth $5 billion dollars. The plan includes the construction of
    500 kilometers of new highways and the modernization of 20 airports
    across the country. President Gustavo Petro stated that this investment
    will create approximately 100,000 new jobs over the next five years.
    Opposition leaders have questioned the funding sources for this
    ambitious project.
    """

    system_prompt = """You are a news summarization assistant.
    Summarize the given article in 2-3 concise sentences.
    Focus on the key facts: who, what, when, where."""

    try:
        client = get_llm_client()
        print("Generating summary...")

        response = client.generate(
            prompt=f"Summarize this article:\n\n{sample_article}",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=200
        )

        print(f"\nOriginal ({len(sample_article)} chars):")
        print(sample_article.strip())
        print(f"\nSummary ({len(response.content)} chars):")
        print(response.content)

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🔬 LLM Integration Test Suite\n")

    results = []

    # Test 1: Connection
    results.append(("LM Studio Connection", test_lm_studio_connection()))

    # Test 2: Basic generation (only if connection works)
    if results[-1][1]:
        results.append(("Text Generation", test_generation()))

        # Test 3: Summarization (only if generation works)
        if results[-1][1]:
            results.append(("News Summarization", test_summarization()))

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

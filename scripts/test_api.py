#!/usr/bin/env python3
"""
Script to test the Journalist Agent API.

Tests all endpoints:
- GET /health
- GET /news/latest
- GET /news/search
- GET /news/{id}
- POST /ask
- POST /agent/run
- GET /agent/sessions

Usage:
    # Start the server first in another terminal:
    poetry run uvicorn src.api.main:app --reload --port 8000

    # Then run this script:
    poetry run python scripts/test_api.py

    # Test specific endpoints:
    poetry run python scripts/test_api.py --endpoint health
    poetry run python scripts/test_api.py --endpoint news
    poetry run python scripts/test_api.py --endpoint ask
    poetry run python scripts/test_api.py --endpoint agent

    # Custom server URL:
    poetry run python scripts/test_api.py --url http://localhost:8080
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

# Default API URL
DEFAULT_URL = "http://localhost:8000"


def print_response(response: httpx.Response, label: str) -> None:
    """Print formatted response."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Status: {response.status_code}")

    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2, ensure_ascii=False, default=str)[:2000]}")
        if len(json.dumps(data)) > 2000:
            print("... (truncated)")
    except Exception:
        print(f"Response: {response.text[:500]}")


def test_health(client: httpx.Client) -> bool:
    """Test health endpoint."""
    print("\n🏥 Testing Health Endpoint...")

    response = client.get("/health")
    print_response(response, "GET /health")

    if response.status_code == 200:
        print("✅ Health check passed")
        return True
    else:
        print("❌ Health check failed")
        return False


def test_news(client: httpx.Client) -> bool:
    """Test news endpoints."""
    print("\n📰 Testing News Endpoints...")

    success = True

    # Test /news/latest
    response = client.get("/news/latest", params={"limit": 5})
    print_response(response, "GET /news/latest?limit=5")

    if response.status_code != 200:
        print("❌ /news/latest failed")
        success = False
    else:
        print("✅ /news/latest passed")

        # If we got articles, test getting one by ID
        data = response.json()
        if data.get("items"):
            article_id = data["items"][0]["article_id"]

            response = client.get(f"/news/{article_id}")
            print_response(response, f"GET /news/{article_id}")

            if response.status_code == 200:
                print("✅ /news/{id} passed")
            else:
                print("❌ /news/{id} failed")
                success = False

    # Test /news/search
    response = client.get("/news/search", params={"q": "economia", "limit": 5})
    print_response(response, "GET /news/search?q=economia&limit=5")

    if response.status_code == 200:
        print("✅ /news/search passed")
    else:
        print("❌ /news/search failed")
        success = False

    return success


def test_ask(client: httpx.Client) -> bool:
    """Test RAG Q&A endpoint."""
    print("\n🤖 Testing RAG Q&A Endpoint...")

    # Test POST /ask
    request_data = {
        "question": "Cuales son las ultimas noticias de economia?",
        "max_sources": 3,
        "use_reranking": False,
    }

    response = client.post("/ask/", json=request_data, timeout=120.0)
    print_response(response, "POST /ask/")

    if response.status_code == 200:
        data = response.json()
        print(f"\n📝 Answer:\n{data.get('answer', 'No answer')[:500]}")
        print(f"\n📚 Sources: {len(data.get('sources', []))}")
        print(f"🎯 Confidence: {data.get('confidence', 0):.2f}")
        print("✅ /ask passed")
        return True
    else:
        print("❌ /ask failed")
        return False


def test_ask_context(client: httpx.Client) -> bool:
    """Test RAG context endpoint."""
    print("\n🔍 Testing RAG Context Endpoint...")

    request_data = {
        "question": "economia",
        "max_sources": 5,
    }

    response = client.post("/ask/context", json=request_data, timeout=60.0)
    print_response(response, "POST /ask/context")

    if response.status_code == 200:
        print("✅ /ask/context passed")
        return True
    else:
        print("❌ /ask/context failed")
        return False


def test_agent(client: httpx.Client) -> bool:
    """Test agent endpoints."""
    print("\n🤖 Testing Agent Endpoints...")

    success = True

    # Test GET /agent/sessions
    response = client.get("/agent/sessions", params={"limit": 5})
    print_response(response, "GET /agent/sessions?limit=5")

    if response.status_code == 200:
        print("✅ /agent/sessions passed")
    else:
        print("❌ /agent/sessions failed")
        success = False

    # Note: POST /agent/run is a long operation, so we skip it by default
    print("\n⚠️  Skipping POST /agent/run (use --run-agent to test)")

    return success


def test_agent_run(client: httpx.Client) -> bool:
    """Test agent run endpoint (long operation)."""
    print("\n🚀 Testing Agent Run Endpoint...")
    print("⚠️  This may take several minutes...")

    request_data = {
        "query": "",
        "max_articles": 5,
        "sources": ["rss"],
        "use_persistence": True,
    }

    response = client.post("/agent/run", json=request_data, timeout=600.0)
    print_response(response, "POST /agent/run")

    if response.status_code == 200:
        data = response.json()
        print(f"\n📊 Results:")
        print(f"   Status: {data.get('status')}")
        print(f"   Articles fetched: {data.get('articles_fetched', 0)}")
        print(f"   Articles processed: {data.get('articles_processed', 0)}")
        print(f"   Articles saved: {data.get('articles_saved', 0)}")
        print(f"   Duration: {data.get('duration_seconds', 0):.1f}s")
        print("✅ /agent/run passed")
        return True
    else:
        print("❌ /agent/run failed")
        return False


def main():
    """Run API tests."""
    parser = argparse.ArgumentParser(description="Test the Journalist Agent API")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"API base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["health", "news", "ask", "agent", "all"],
        default="all",
        help="Specific endpoint to test (default: all)",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Include POST /agent/run test (takes several minutes)",
    )
    args = parser.parse_args()

    print("\n🧪 Journalist Agent API - Test Suite")
    print("=" * 60)
    print(f"API URL: {args.url}")
    print(f"Testing: {args.endpoint}")
    print("=" * 60)

    # Create client
    client = httpx.Client(base_url=args.url, timeout=30.0)

    results = {}

    try:
        # Run tests based on endpoint selection
        if args.endpoint in ["health", "all"]:
            results["health"] = test_health(client)

        if args.endpoint in ["news", "all"]:
            results["news"] = test_news(client)

        if args.endpoint in ["ask", "all"]:
            results["ask"] = test_ask(client)
            results["ask_context"] = test_ask_context(client)

        if args.endpoint in ["agent", "all"]:
            results["agent_sessions"] = test_agent(client)
            if args.run_agent:
                results["agent_run"] = test_agent_run(client)

    except httpx.ConnectError:
        print(f"\n❌ Could not connect to API at {args.url}")
        print("Make sure the server is running:")
        print("  poetry run uvicorn src.api.main:app --reload --port 8000")
        return 1

    finally:
        client.close()

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Test script for Telegram bot.

Usage:
    poetry run python scripts/test_bot.py

This script verifies:
1. Bot token is configured
2. Bot can connect to Telegram
3. Handlers are properly registered
"""

import asyncio
import sys

from aiogram import Bot

from src.config.settings import settings


async def test_bot_connection() -> bool:
    """Test bot token and connection."""
    print("=" * 60)
    print("TELEGRAM BOT TEST")
    print("=" * 60)

    # Check token
    if not settings.telegram_bot_token:
        print("\n[ERROR] TELEGRAM_BOT_TOKEN not configured!")
        print("  1. Create a bot with @BotFather on Telegram")
        print("  2. Copy the token to your .env file:")
        print("     TELEGRAM_BOT_TOKEN=your_token_here")
        return False

    print(f"\n[OK] Token configured: {settings.telegram_bot_token[:10]}...")

    # Test connection
    print("\nTesting connection to Telegram...")
    try:
        bot = Bot(token=settings.telegram_bot_token)
        bot_info = await bot.get_me()
        await bot.session.close()

        print(f"\n[OK] Connected successfully!")
        print(f"     Bot username: @{bot_info.username}")
        print(f"     Bot ID: {bot_info.id}")
        print(f"     Bot name: {bot_info.first_name}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        return False


async def test_handlers() -> bool:
    """Test that handlers are properly imported."""
    print("\n" + "-" * 60)
    print("HANDLER TESTS")
    print("-" * 60)

    try:
        from src.bot.handlers import ask_router, news_router, start_router

        print(f"\n[OK] start_router loaded: {len(list(start_router.message.handlers))} handlers")
        print(f"[OK] news_router loaded: {len(list(news_router.message.handlers))} handlers")
        print(f"[OK] ask_router loaded: {len(list(ask_router.message.handlers))} handlers")

        return True

    except ImportError as e:
        print(f"\n[ERROR] Failed to import handlers: {e}")
        return False


async def test_keyboards() -> bool:
    """Test keyboard builders."""
    print("\n" + "-" * 60)
    print("KEYBOARD TESTS")
    print("-" * 60)

    try:
        from src.bot.keyboards import (
            article_detail_keyboard,
            categories_keyboard,
            news_pagination_keyboard,
        )

        # Test pagination
        kb = news_pagination_keyboard(0, 20, 5)
        print(f"\n[OK] news_pagination_keyboard: {len(kb.inline_keyboard[0]) if kb else 0} buttons")

        # Test categories
        kb = categories_keyboard(["politica", "economia", "deportes"])
        print(f"[OK] categories_keyboard: {len(kb.inline_keyboard)} rows")

        # Test article detail
        kb = article_detail_keyboard("test-id", "https://example.com")
        print(f"[OK] article_detail_keyboard: {len(kb.inline_keyboard)} rows")

        return True

    except Exception as e:
        print(f"\n[ERROR] Keyboard test failed: {e}")
        return False


async def main() -> int:
    """Run all tests."""
    results = []

    results.append(await test_bot_connection())
    results.append(await test_handlers())
    results.append(await test_keyboards())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if all(results):
        print("\n[SUCCESS] All tests passed!")
        print("\nTo start the bot, run:")
        print("  poetry run python -m src.bot.main")
        return 0
    else:
        print("\n[FAILED] Some tests failed. Fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

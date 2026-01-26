"""
Telegram bot main entry point.

Usage:
    poetry run python -m src.bot.main
"""

import asyncio
import sys

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from src.bot.config import MESSAGES
from src.bot.handlers import ask_router, news_router, start_router
from src.config.settings import settings

logger = structlog.get_logger()


async def main() -> None:
    """Initialize and start the bot."""
    # Check token
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not configured")
        print(MESSAGES["no_token"])
        sys.exit(1)

    logger.info("Starting Telegram bot...")

    # Initialize bot with default properties
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # Initialize dispatcher
    dp = Dispatcher()

    # Include routers (order matters - first match wins)
    dp.include_router(start_router)
    dp.include_router(news_router)
    dp.include_router(ask_router)  # Must be last - catches all text

    # Log bot info
    bot_info = await bot.get_me()
    logger.info(
        "Bot initialized",
        username=bot_info.username,
        bot_id=bot_info.id,
    )

    print(f"Bot @{bot_info.username} started. Press Ctrl+C to stop.")

    # Start polling
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        logger.info("Bot stopped")


def run_bot() -> None:
    """Run the bot (blocking)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    run_bot()

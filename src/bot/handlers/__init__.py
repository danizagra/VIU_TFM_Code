"""
Bot command handlers.
"""

from src.bot.handlers.start import router as start_router
from src.bot.handlers.news import router as news_router
from src.bot.handlers.ask import router as ask_router

__all__ = ["start_router", "news_router", "ask_router"]

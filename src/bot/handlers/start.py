"""
Start and help command handlers.
"""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.config import MESSAGES

router = Router(name="start")


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Handle /start command."""
    await message.answer(MESSAGES["welcome"])


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command."""
    await message.answer(MESSAGES["help"])

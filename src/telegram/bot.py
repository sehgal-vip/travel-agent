"""Telegram bot setup — async, python-telegram-bot v20+."""

from __future__ import annotations

import logging

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.config.settings import get_settings
from src.telegram.handlers import process_message

logger = logging.getLogger(__name__)


def create_bot(graph, repo) -> Application:
    """Build and configure the Telegram bot application.

    Args:
        graph: Compiled LangGraph instance.
        repo: TripRepository instance.
    """
    settings = get_settings()
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Store graph and repo in bot_data for access in handlers
    app.bot_data["graph"] = graph
    app.bot_data["repo"] = repo

    # All slash commands route through the same handler (orchestrator decides)
    commands = [
        "start", "research", "library", "priorities", "plan",
        "agenda", "feedback", "costs", "adjust", "status",
        "help", "trips", "trip",
    ]
    for cmd in commands:
        app.add_handler(CommandHandler(cmd, process_message))

    # Plain text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    logger.info("Telegram bot configured with %d command handlers.", len(commands))
    return app

"""Telegram bot setup — async, python-telegram-bot v20+."""

from __future__ import annotations

import logging

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.config.settings import get_settings
from src.telegram.handlers import handle_trip_selection, process_message

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
        "help", "trips", "mytrips", "trip", "join", "summary",
    ]
    for cmd in commands:
        app.add_handler(CommandHandler(cmd, process_message))

    # Inline button callbacks (trip switching)
    app.add_handler(CallbackQueryHandler(handle_trip_selection, pattern="^trip_select:"))

    # Plain text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Register proactive nudge job (runs every 6 hours)
    if app.job_queue is not None:
        from src.telegram.nudge import nudge_check
        app.job_queue.run_repeating(nudge_check, interval=21600, first=60)
        logger.info("Proactive nudge job registered (6h interval).")

    logger.info("Telegram bot configured with %d command handlers.", len(commands))
    return app

"""Entry point — initialise DB, compile graph, start Telegram bot."""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    from src.config.settings import get_settings
    settings = get_settings()

    os.makedirs("data/logs", exist_ok=True)

    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console)

    # File handler (rotating, persistent)
    file_handler = logging.handlers.RotatingFileHandler(
        "data/logs/bot.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)

    async def _start() -> None:
        # 1. Init database
        from src.db.migrations import init_db
        repo = await init_db(settings.DATABASE_URL)
        logger.info("Database ready.")

        # 2. Compile LangGraph with async SQLite checkpointer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from src.graph import compile_graph
        os.makedirs("data", exist_ok=True)

        async with AsyncSqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
            graph = compile_graph(checkpointer=checkpointer)
            logger.info("LangGraph compiled.")

            # 3. Create and run Telegram bot
            from src.telegram.bot import create_bot
            app = create_bot(graph, repo)
            logger.info("Starting Telegram bot polling...")

            async with app:
                await app.initialize()
                await app.start()
                await app.updater.start_polling(drop_pending_updates=True)

                # Run until interrupted
                stop_event = asyncio.Event()

                def _signal_handler():
                    stop_event.set()

                loop = asyncio.get_running_loop()
                for sig_name in ("SIGINT", "SIGTERM"):
                    try:
                        import signal
                        loop.add_signal_handler(getattr(signal, sig_name), _signal_handler)
                    except (NotImplementedError, AttributeError):
                        pass

                logger.info("Bot is running. Press Ctrl+C to stop.")
                await stop_event.wait()

                logger.info("Shutting down...")
                await app.updater.stop()
                await app.stop()
                await repo.close()

    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()

"""Telegram message handlers — bridge between Telegram and LangGraph."""

from __future__ import annotations

import asyncio
import json
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.telegram.formatters import split_message

logger = logging.getLogger(__name__)

# Internal state keys that should not be persisted to the trip repo
_INTERNAL_KEYS = {"_next", "_user_message", "messages"}


async def _keep_typing(chat) -> None:
    """Send typing indicator every 4 seconds until cancelled."""
    try:
        while True:
            await asyncio.sleep(4)
            await chat.send_action(ChatAction.TYPING)
    except asyncio.CancelledError:
        pass


async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any incoming text message or command.

    Flow: get user -> lookup active trip -> load prior state -> invoke graph -> merge & save -> reply.
    """
    if not update.message or not update.message.text:
        await update.message.reply_text("I can only process text messages for now.")
        return

    user_id = str(update.effective_user.id)
    message_text = update.message.text.strip()
    logger.info("Received message from user %s: %.80s", user_id, message_text)

    # Send typing indicator while processing
    await update.message.chat.send_action(ChatAction.TYPING)

    graph = context.bot_data.get("graph")
    repo = context.bot_data.get("repo")

    if not graph:
        await update.message.reply_text("Bot is still starting up. Please try again in a moment.")
        return

    # Determine active trip & thread_id
    trip_id = context.user_data.get("active_trip_id", "default")
    thread_id = f"{user_id}_{trip_id}"

    # Special handling for trip management commands
    if message_text.startswith("/trip") and not message_text.startswith("/trips"):
        result = await _handle_trip_management(update, context, user_id, message_text, repo)
        if result:
            return

    if message_text == "/trips":
        await _list_trips(update, context, user_id, repo)
        return

    # Start a background typing indicator refresh task
    typing_task = asyncio.create_task(_keep_typing(update.message.chat))

    # Invoke the LangGraph
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Load prior state from trip repo as fallback for stale checkpointer
        input_state: dict = {"messages": [{"role": "user", "content": message_text}]}
        if repo:
            try:
                existing_trip = await repo.get_trip(trip_id)
                if existing_trip and existing_trip.state_json:
                    prior_state = json.loads(existing_trip.state_json)
                    # Spread prior state under input; checkpointer takes precedence
                    input_state = {**prior_state, "messages": [{"role": "user", "content": message_text}]}
                    logger.info("Loaded prior state for trip %s (keys: %s)", trip_id, list(prior_state.keys()))
            except Exception:
                logger.exception("Failed to load prior state for trip %s", trip_id)

        logger.info("Invoking graph for thread %s", thread_id)
        result = await graph.ainvoke(input_state, config=config)
        logger.info(
            "Graph returned for thread %s (keys: %s, current_agent: %s)",
            thread_id,
            list(result.keys()) if result else [],
            result.get("current_agent") if result else None,
        )

        # Extract the response to send back
        response_text = _extract_response(result)

        # Save state to DB if we have a repo — merge, don't replace
        if repo and result:
            try:
                state_to_save = {k: v for k, v in result.items() if k not in _INTERNAL_KEYS}
                existing = await repo.get_trip(trip_id)
                if existing:
                    await repo.merge_state(trip_id, state_to_save)
                else:
                    await repo.create_trip(trip_id, user_id, state_to_save)
                logger.info("State saved for trip %s", trip_id)

                # Update active trip id if onboarding just completed
                if result.get("trip_id") and result["trip_id"] != trip_id:
                    context.user_data["active_trip_id"] = result["trip_id"]
            except Exception:
                logger.exception("Failed to save state for trip %s", trip_id)

    except Exception:
        logger.exception("Error processing message for user %s", user_id)
        response_text = "Something went wrong. Please try again or send /help for commands."
    finally:
        typing_task.cancel()

    # Send response, split if needed
    for part in split_message(response_text):
        await update.message.reply_text(part)
    logger.info("Response sent to user %s (%d chars)", user_id, len(response_text))


async def _handle_trip_management(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str,
    message: str,
    repo,
) -> bool:
    """Handle /trip new, /trip switch, /trip archive. Returns True if handled."""
    parts = message.split()
    if len(parts) < 2:
        return False

    sub = parts[1].lower()

    if sub == "new":
        import uuid
        new_id = str(uuid.uuid4())[:8]
        context.user_data["active_trip_id"] = new_id
        await update.message.reply_text(
            f"Starting a new trip (ID: {new_id}). Let's plan your next adventure!\n\n"
            "Where are you dreaming of traveling to?"
        )
        return True

    if sub == "switch" and len(parts) >= 3:
        target_id = parts[2]
        if repo:
            trip = await repo.get_trip(target_id)
            if trip:
                context.user_data["active_trip_id"] = target_id
                state = json.loads(trip.state_json) if trip.state_json else {}
                country = state.get("destination", {}).get("country", "Unknown")
                flag = state.get("destination", {}).get("flag_emoji", "")
                await update.message.reply_text(f"Switched to trip: {flag} {country} ({target_id})")
            else:
                await update.message.reply_text(f"Trip {target_id} not found.")
        return True

    if sub == "archive" and len(parts) >= 3:
        target_id = parts[2]
        if repo:
            try:
                await repo.archive_trip(target_id)
                await update.message.reply_text(f"Trip {target_id} archived.")
            except ValueError:
                await update.message.reply_text(f"Trip {target_id} not found.")
        return True

    return False


async def _list_trips(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str,
    repo,
) -> None:
    """Handle /trips — list all trips for this user."""
    if not repo:
        await update.message.reply_text("No trips found.")
        return

    trips = await repo.list_trips(user_id)
    if not trips:
        await update.message.reply_text("No trips yet. Send /start to plan your first trip!")
        return

    active_id = context.user_data.get("active_trip_id", "default")
    lines = ["Your trips:\n"]

    for trip in trips:
        state = json.loads(trip.state_json) if trip.state_json else {}
        country = state.get("destination", {}).get("country", "Unknown")
        flag = state.get("destination", {}).get("flag_emoji", "")
        status = "📦 Archived" if trip.archived else "✅ Active"
        current = " ← current" if trip.trip_id == active_id else ""
        lines.append(f"  {flag} {country} — {trip.trip_id} {status}{current}")

    lines.append("\nUse /trip switch <id> to change active trip.")
    await update.message.reply_text("\n".join(lines))


def _extract_response(result: dict) -> str:
    """Extract the human-readable response from graph output."""
    # Check for direct response field
    if result.get("response"):
        return result["response"]

    # Check messages for the last AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if msg.type == "ai":
                return msg.content
        elif isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return msg.get("content", "")

    return "I'm not sure how to respond to that. Try /help for available commands."

"""Telegram message handlers — bridge between Telegram and LangGraph."""

from __future__ import annotations

import asyncio
import json
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
        await update.message.reply_text("I work with text messages and commands. Try /help to see what I can do!")
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
    thread_id = trip_id

    # Handle /join command
    if message_text.startswith("/join"):
        await _handle_join(update, context, user_id, message_text, repo)
        return

    # Special handling for trip management commands
    if message_text.startswith("/trip") and not message_text.startswith("/trips"):
        result = await _handle_trip_management(update, context, user_id, message_text, repo)
        if result:
            return

    if message_text in ("/trips", "/mytrips"):
        await _list_mytrips(update, context, user_id, repo)
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
                    new_trip_id = result["trip_id"]
                    context.user_data["active_trip_id"] = new_trip_id
                    response_text += (
                        f"\n\nYour trip ID is: `{new_trip_id}`\n"
                        f"Share this with your travel companions — they can join with /join {new_trip_id}"
                    )
            except Exception:
                logger.exception("Failed to save state for trip %s", trip_id)

    except Exception:
        logger.exception("Error processing message for user %s", user_id)
        response_text = "Something went wrong processing your message. Try again, or use /help to see available commands."
    finally:
        typing_task.cancel()

    # Send response, split if needed
    for part in split_message(response_text):
        await update.message.reply_text(part)
    logger.info("Response sent to user %s (%d chars)", user_id, len(response_text))


async def _handle_join(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str,
    message: str,
    repo,
) -> None:
    """Handle /join <trip_id> — join an existing trip as a member."""
    parts = message.split()
    if len(parts) < 2:
        await update.message.reply_text("Usage: /join <trip_id>")
        return

    trip_id = parts[1]

    if not repo:
        await update.message.reply_text("Database is not available. Please try again later.")
        return

    trip = await repo.get_trip(trip_id)
    if not trip:
        await update.message.reply_text(f"Trip '{trip_id}' not found. Use /trips to see your trips.")
        return
    if trip.archived:
        await update.message.reply_text(f"Trip `{trip_id}` has been archived and can no longer be joined.")
        return

    try:
        await repo.add_member(trip_id, user_id)
    except ValueError:
        await update.message.reply_text(f"Trip '{trip_id}' not found. Use /trips to see your trips.")
        return

    context.user_data["active_trip_id"] = trip_id

    state = json.loads(trip.state_json) if trip.state_json else {}
    country = state.get("destination", {}).get("country", "Unknown")
    flag = state.get("destination", {}).get("flag_emoji", "")
    cities = state.get("cities", [])
    city_names = ", ".join(c.get("name", "") for c in cities) if cities else "not set yet"

    await update.message.reply_text(
        f"You've joined the trip! {flag} {country}\n"
        f"Cities: {city_names}\n\n"
        f"Trip `{trip_id}` is now your active trip.\n"
        f"Use /status to see the full trip plan, or /help to see all commands."
    )
    logger.info("User %s joined trip %s", user_id, trip_id)


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

    if sub == "id":
        active_id = context.user_data.get("active_trip_id", "default")
        await update.message.reply_text(
            f"Your current trip ID is: `{active_id}`\n"
            f"Share this with your travel companions — they can join with /join {active_id}",
            parse_mode="Markdown",
        )
        return True

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
                await update.message.reply_text(f"Trip '{target_id}' not found. Use /trips to see your trips.")
        return True

    if sub == "archive" and len(parts) >= 3:
        target_id = parts[2]
        confirmed = len(parts) >= 4 and parts[3].lower() == "confirm"
        if not confirmed:
            await update.message.reply_text(
                f"Are you sure you want to archive trip {target_id}? "
                f"Send `/trip archive {target_id} confirm` to confirm."
            )
            return True
        if repo:
            try:
                await repo.archive_trip(target_id)
                await update.message.reply_text(f"Trip {target_id} archived.")
            except ValueError:
                await update.message.reply_text(f"Trip '{target_id}' not found. Use /trips to see your trips.")
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
        role = "[owner]" if trip.user_id == user_id else "[member]"
        status = "📦 Archived" if trip.archived else "✅ Active"
        current = " ← current" if trip.trip_id == active_id else ""
        lines.append(f"  {flag} {country} — {trip.trip_id} {role} {status}{current}")

    lines.append("\nUse /trip switch <id> to change active trip.")
    await update.message.reply_text("\n".join(lines))


def _fallback_title_from_state(state: dict) -> str:
    """Deterministic fallback title for trips that lack a trip_title."""
    dest = state.get("destination", {})
    cities = state.get("cities", [])
    country = dest.get("country", "Adventure")
    if cities:
        city_names = " & ".join(c.get("name", "") for c in cities[:2])
        if len(cities) > 2:
            city_names += f" +{len(cities) - 2}"
        return f"{country}: {city_names}"
    return f"{country} Trip"


def _format_date_range(start: str, end: str) -> str:
    """Format ISO date strings into a compact range like 'Apr 1-14' or 'Mar 28 - Apr 5'."""
    if not start or not end:
        return ""
    try:
        from datetime import datetime as _dt
        s = _dt.strptime(start, "%Y-%m-%d")
        e = _dt.strptime(end, "%Y-%m-%d")
        if s.month == e.month:
            return f"{s.strftime('%b')} {s.day}-{e.day}"
        return f"{s.strftime('%b')} {s.day} - {e.strftime('%b')} {e.day}"
    except (ValueError, TypeError):
        return ""


def _trip_progress_indicator(state: dict) -> str:
    """Return a human-readable progress string based on state."""
    if not state.get("onboarding_complete"):
        return "Setting up"
    if state.get("detailed_agenda"):
        return "Scheduled"
    if state.get("high_level_plan"):
        return "Planned"
    if state.get("priorities"):
        return "Researched"
    if state.get("research"):
        return "Ready to prioritize"
    return "Ready to research"


async def _list_mytrips(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: str,
    repo,
) -> None:
    """Handle /mytrips (and /trips) — rich trip listing with inline switch buttons."""
    if not repo:
        await update.message.reply_text("No trips found.")
        return

    trips = await repo.list_trips(user_id)
    if not trips:
        await update.message.reply_text("No trips yet. Send /start to plan your first trip!")
        return

    active_id = context.user_data.get("active_trip_id", "default")
    lines = ["Your trips:\n"]
    buttons = []

    for trip in trips:
        state = json.loads(trip.state_json) if trip.state_json else {}
        dest = state.get("destination", {})
        flag = dest.get("flag_emoji", "")
        title = state.get("trip_title") or _fallback_title_from_state(state)
        is_current = trip.trip_id == active_id
        is_archived = trip.archived

        # Route line: Tokyo > Kyoto > Osaka
        cities = state.get("cities", [])
        route = " > ".join(c.get("name", "") for c in cities) if cities else "No cities yet"

        # Date range
        dates = state.get("dates", {})
        date_range = _format_date_range(dates.get("start", ""), dates.get("end", ""))

        # Progress
        progress = _trip_progress_indicator(state)

        # Role
        role = "owner" if trip.user_id == user_id else "member"

        # Build display line
        current_tag = " (current)" if is_current else ""
        archived_tag = " [archived]" if is_archived else ""
        header = f"{flag} {title}{current_tag}{archived_tag}"
        detail_parts = []
        if route:
            detail_parts.append(route)
        sub_parts = []
        if date_range:
            sub_parts.append(date_range)
        sub_parts.append(progress)
        sub_parts.append(role)
        if sub_parts:
            detail_parts.append(" · ".join(sub_parts))

        lines.append(header)
        for part in detail_parts:
            lines.append(f"   {part}")
        lines.append("")

        # Add inline button for non-current, non-archived trips
        if not is_current and not is_archived:
            buttons.append(
                [InlineKeyboardButton(f"{flag} {title}"[:40], callback_data=f"trip_select:{trip.trip_id}")]
            )

    text = "\n".join(lines).strip()
    reply_markup = InlineKeyboardMarkup(buttons) if buttons else None
    await update.message.reply_text(text, reply_markup=reply_markup)


async def handle_trip_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callback for trip switching."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("trip_select:"):
        return

    trip_id = data.split(":", 1)[1]
    user_id = str(update.effective_user.id)
    repo = context.bot_data.get("repo")

    if not repo:
        await query.edit_message_text("Database is not available. Please try again later.")
        return

    trip = await repo.get_trip(trip_id)
    if not trip:
        await query.edit_message_text(f"Trip '{trip_id}' not found.")
        return

    # Verify user has access (owner or member)
    user_trips = await repo.list_trips(user_id)
    accessible_ids = {t.trip_id for t in user_trips}
    if trip_id not in accessible_ids:
        await query.edit_message_text("You don't have access to this trip.")
        return

    context.user_data["active_trip_id"] = trip_id

    state = json.loads(trip.state_json) if trip.state_json else {}
    dest = state.get("destination", {})
    flag = dest.get("flag_emoji", "")
    title = state.get("trip_title") or _fallback_title_from_state(state)

    await query.edit_message_text(
        f"Switched to: {flag} {title}\n\n"
        f"Use /status to see your trip progress, or just chat to continue planning!"
    )
    logger.info("User %s switched to trip %s via inline button", user_id, trip_id)


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

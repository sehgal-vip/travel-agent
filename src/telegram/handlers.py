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
_INTERNAL_KEYS = {
    "_next", "_user_message", "messages", "_awaiting_input", "_callback",
    "_delegate_to", "_chain", "_routing_echo", "_error_agent", "_error_context",
    "_loopback_depth",
}

COMMAND_DISPATCH = {
    "/start": "onboarding",
    "/research": "research",
    "/priorities": "prioritizer",
    "/plan": "planner",
    "/agenda": "scheduler",
    "/feedback": "feedback",
    "/costs": "cost",
    "/adjust": "feedback",
}


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

        # Pre-graph slash command dispatch (v2)
        if message_text.startswith("/"):
            cmd = message_text.split()[0].lower()

            # Direct handlers (skip graph entirely)
            if cmd == "/status":
                from src.agents.orchestrator import generate_status
                status = generate_status(input_state)
                for part in split_message(status):
                    await update.message.reply_text(part)
                return
            if cmd == "/help":
                from src.agents.orchestrator import generate_help
                for part in split_message(generate_help()):
                    await update.message.reply_text(part)
                return
            if cmd == "/library":
                try:
                    from src.tools.library_sync import library_sync
                    sync_result = await library_sync(input_state)
                    state_to_save = {k: v for k, v in input_state.items() if k not in _INTERNAL_KEYS}
                    if sync_result.get("library"):
                        state_to_save["library"] = sync_result["library"]
                    if repo:
                        existing = await repo.get_trip(trip_id)
                        if existing:
                            await repo.merge_state(trip_id, state_to_save)
                    for part in split_message(sync_result.get("response", "Library synced.")):
                        await update.message.reply_text(part)
                    return
                except Exception:
                    logger.exception("Library sync failed")
                    await update.message.reply_text("Library sync failed. Try again later.")
                    return
            if cmd == "/summary":
                try:
                    from src.tools.trip_synthesis import synthesize_trip
                    summary = await synthesize_trip(input_state)
                    for part in split_message(summary):
                        await update.message.reply_text(part)
                    return
                except Exception:
                    logger.exception("Trip synthesis failed")
                    await update.message.reply_text("Could not generate summary. Try /status instead.")
                    return

            # Pre-set _next for command dispatch
            if cmd in COMMAND_DISPATCH:
                input_state["_next"] = COMMAND_DISPATCH[cmd]

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

                # Auto-trigger library_sync for research/planner/feedback
                responding_agent = result.get("current_agent", "orchestrator")
                if responding_agent in {"research", "planner", "feedback"}:
                    try:
                        from src.tools.library_sync import library_sync
                        sync_result = await library_sync(state_to_save)
                        if sync_result.get("library"):
                            await repo.merge_state(trip_id, {"library": sync_result["library"]})
                    except Exception:
                        logger.exception("Auto library sync failed after %s", responding_agent)

                # Update per-agent memory files
                from src.tools.trip_memory import (
                    append_agent_notes,
                    build_and_write_agent_memory,
                )

                responding_agent = result.get("current_agent", "orchestrator")
                try:
                    await build_and_write_agent_memory(trip_id, responding_agent, state_to_save)
                except Exception:
                    logger.exception("Failed to write agent memory for %s", responding_agent)

                # Generate LLM notes for agents that benefit from accumulated learnings
                if _should_generate_notes(result, responding_agent):
                    try:
                        await _generate_and_append_notes(
                            trip_id, responding_agent, result, append_agent_notes
                        )
                    except Exception:
                        _log_notes_failure(responding_agent)

                # Update active trip id if onboarding just completed
                if result.get("trip_id") and result["trip_id"] != trip_id:
                    new_trip_id = result["trip_id"]
                    context.user_data["active_trip_id"] = new_trip_id

                    # Migrate state to the new trip_id so subsequent messages find it
                    try:
                        new_existing = await repo.get_trip(new_trip_id)
                        if new_existing:
                            await repo.merge_state(new_trip_id, state_to_save)
                        else:
                            await repo.create_trip(new_trip_id, user_id, state_to_save)
                        # Create directory for agent memory files; they'll be written on first agent run
                        try:
                            from pathlib import Path
                            Path(f"./data/trips/{new_trip_id}").mkdir(parents=True, exist_ok=True)
                        except Exception:
                            logger.exception("Failed to create trip directory for %s", new_trip_id)
                        logger.info("State migrated to new trip %s", new_trip_id)
                    except Exception:
                        logger.exception("Failed to migrate state to new trip %s", new_trip_id)

                    trip_name = result.get("trip_title") or _fallback_title_from_state(result)
                    response_text += (
                        f"\n\n✈️ *{trip_name}*\n"
                        f"Trip ID: `{new_trip_id}` — share with /join {new_trip_id}"
                    )

                    # Auto-start research after onboarding completes
                    try:
                        # Send onboarding confirmation first, then start research
                        for part in split_message(response_text):
                            await update.message.reply_text(part)
                        logger.info("Onboarding response sent, auto-starting research for trip %s", new_trip_id)
                        response_text = ""  # Clear so we don't send twice

                        research_input = {**state_to_save, "messages": [{"role": "user", "content": "/research all"}]}
                        research_config = {"configurable": {"thread_id": new_trip_id}}
                        research_result = await graph.ainvoke(research_input, config=research_config)

                        research_response = _extract_response(research_result)
                        response_text = research_response

                        # Save research state
                        research_state = {k: v for k, v in research_result.items() if k not in _INTERNAL_KEYS}
                        await repo.merge_state(new_trip_id, research_state)
                        try:
                            await build_and_write_agent_memory(new_trip_id, "research", research_state)
                        except Exception:
                            logger.exception("Failed to write agent memory after auto-research for %s", new_trip_id)
                        logger.info("Auto-research completed for trip %s", new_trip_id)
                    except Exception:
                        logger.exception("Auto-research failed for trip %s", new_trip_id)
                        if not response_text:
                            response_text = (
                                "Your trip is set up! I tried to auto-start research but hit a snag.\n"
                                "Run /research all to kick it off manually."
                            )
            except Exception:
                logger.exception("Failed to save state for trip %s", trip_id)

    except Exception:
        logger.exception("Error processing message for user %s", user_id)
        response_text = "Something went wrong processing your message. Try again, or use /help to see available commands."
    finally:
        typing_task.cancel()

    # Send response, split if needed (may be empty if already sent during auto-research)
    if response_text:
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

    title = state.get("trip_title") or _fallback_title_from_state(state)
    await update.message.reply_text(
        f"You've joined *{title}*! {flag}\n"
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
        title = active_id
        if repo:
            trip = await repo.get_trip(active_id)
            if trip and trip.state_json:
                st = json.loads(trip.state_json)
                title = st.get("trip_title") or _fallback_title_from_state(st)
        await update.message.reply_text(
            f"✈️ *{title}*\n"
            f"Trip ID: `{active_id}`\n"
            f"Share with: /join {active_id}",
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
                flag = state.get("destination", {}).get("flag_emoji", "")
                title = state.get("trip_title") or _fallback_title_from_state(state)
                await update.message.reply_text(f"Switched to: {flag} *{title}* (`{target_id}`)")
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


# ─── Notes Generation Helpers ────────────────────────────

_notes_failure_counts: dict[str, int] = {}

NOTES_MAX_TOKENS = 300
NOTES_MODEL = "claude-haiku-4-5-20251001"


def _should_generate_notes(result: dict, responding_agent: str) -> bool:
    """Check if this response warrants generating agent notes.

    Notes are generated when a NOTES_ELIGIBLE agent produces state updates
    in research, planning, or feedback domains.
    """
    from src.agents.constants import NOTES_ELIGIBLE_AGENTS

    if responding_agent not in NOTES_ELIGIBLE_AGENTS:
        return False
    state_updates = result.get("state_updates", result)
    return bool(
        state_updates.get("research")
        or state_updates.get("high_level_plan")
        or state_updates.get("feedback_log")
    )


def _extract_notes_context(response_text: str, max_chars: int = 1500) -> str:
    """Take first line (summary) + tail (conclusions), not blind head truncation."""
    if len(response_text) <= max_chars:
        return response_text
    first_line = response_text.split("\n", 1)[0]
    tail = response_text[-(max_chars - len(first_line) - 10):]
    return f"{first_line}\n...\n{tail}"


async def _generate_and_append_notes(
    trip_id: str,
    responding_agent: str,
    result: dict,
    append_fn,
) -> None:
    """Generate LLM notes and append to agent memory.

    Uses structured bullet format for robust parsing. Deduplicates against existing notes.
    Wraps existing notes in XML delimiters for prompt injection protection.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage as HMsg
    from langchain_core.messages import SystemMessage as SMsg
    from src.tools.trip_memory import read_agent_notes

    existing_notes = read_agent_notes(trip_id, responding_agent) or ""

    notes_llm = ChatAnthropic(model=NOTES_MODEL, max_tokens=NOTES_MAX_TOKENS)
    notes_response = await notes_llm.ainvoke([
        SMsg(content=(
            "You are a travel planning agent's memory system. "
            "Write 2-3 brief bullet points about patterns, insights, or things to remember. "
            "Be specific and actionable. No fluff.\n\n"
            "Always format notes as bullet points starting with `- `. "
            "If nothing new, return only: `- NONE`\n"
            "Mark durable preferences (dietary, mobility, strong likes/dislikes) with [pinned]. "
            "Most notes should NOT be pinned.\n"
            "If an insight is relevant to OTHER agents (e.g., user energy level, dietary discovery, "
            "budget surprise), mark with [shared].\n\n"
            "Write NEW patterns only. Do NOT repeat anything in <existing_notes>."
        )),
        HMsg(content=(
            f"Agent: {responding_agent}\n"
            f"Work done: {_extract_notes_context(_extract_response(result))}\n\n"
            f"<existing_notes>\n{existing_notes[-1000:]}\n</existing_notes>"
        )),
    ])

    notes_text = notes_response.content.strip()
    # Extract bullet lines only — ignore any preamble/commentary
    bullet_lines = [l.strip() for l in notes_text.splitlines() if l.strip().startswith("- ")]
    # Filter out the "none" signal
    real_bullets = [l for l in bullet_lines if l.strip().lower() != "- none"]
    if real_bullets:
        await append_fn(trip_id, responding_agent, "\n".join(real_bullets))


def _log_notes_failure(agent_name: str) -> None:
    """Log notes generation failures — first 3, then every 10th."""
    _notes_failure_counts[agent_name] = _notes_failure_counts.get(agent_name, 0) + 1
    count = _notes_failure_counts[agent_name]
    if count <= 3 or count % 10 == 0:
        logger.warning(
            "Failed to generate agent notes for %s (failure #%d)",
            agent_name, count, exc_info=True,
        )


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

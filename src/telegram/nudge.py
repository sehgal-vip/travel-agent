"""Proactive nudge system — periodic checks and notifications."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def nudge_check(context) -> None:
    """Periodic nudge job — runs every 6 hours.

    Checks all active trips for nudge triggers and sends messages.
    Called by python-telegram-bot's job_queue.
    """
    repo = context.bot_data.get("repo")
    if not repo:
        return

    bot = context.bot

    try:
        # Get all trips (we'd need to iterate users; for now check recent trips)
        # In practice, you'd track active user-trip mappings
        # For simplicity, we check trips updated in the last 7 days
        # This is a placeholder — in production you'd have a proper user registry
        logger.info("Running nudge check")
    except Exception:
        logger.exception("Nudge check failed")


def check_nudge_triggers(state: dict) -> list[dict]:
    """Check a trip state for nudge triggers.

    Returns a list of nudge dicts with 'type', 'message', and 'priority'.
    """
    nudges = []
    now = datetime.now(timezone.utc)

    # 1. Booking deadline nudge
    _check_booking_deadlines(state, now, nudges)

    # 2. Research staleness nudge
    _check_research_staleness(state, now, nudges)

    # 3. Budget drift nudge
    _check_budget_drift(state, nudges)

    # 4. Unvisited must-dos nudge
    _check_unvisited_must_dos(state, nudges)

    # 5. Stale trip (no activity) nudge
    _check_stale_trip(state, now, nudges)

    return nudges


def _check_booking_deadlines(state: dict, now: datetime, nudges: list) -> None:
    """Check for must-do items that need advance booking and trip is approaching."""
    dates = state.get("dates", {})
    trip_start = dates.get("start", "")
    if not trip_start:
        return

    try:
        trip_dt = datetime.fromisoformat(trip_start + "T00:00:00+00:00")
        days_until = (trip_dt - now).days
    except (ValueError, TypeError):
        return

    if days_until > 14 or days_until < 0:
        return

    # Check research for items needing advance booking
    research = state.get("research", {})
    for city_name, city_data in research.items():
        for category in ("places", "activities"):
            for item in city_data.get(category, []):
                if item.get("advance_booking"):
                    nudges.append({
                        "type": "booking_deadline",
                        "priority": "high" if days_until <= 7 else "medium",
                        "message": (
                            f"\ud83d\udd34 Booking reminder: {item.get('name', '?')} in {city_name} "
                            f"needs advance booking and your trip starts in {days_until} days!"
                        ),
                    })


def _check_research_staleness(state: dict, now: datetime, nudges: list) -> None:
    """Check if research is stale and trip is approaching."""
    dates = state.get("dates", {})
    trip_start = dates.get("start", "")
    if not trip_start:
        return

    try:
        trip_dt = datetime.fromisoformat(trip_start + "T00:00:00+00:00")
        days_until = (trip_dt - now).days
    except (ValueError, TypeError):
        return

    if days_until > 30 or days_until < 0:
        return

    research = state.get("research", {})
    for city_name, city_data in research.items():
        last_updated = city_data.get("last_updated", "")
        if not last_updated:
            continue
        try:
            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_days = (now - updated_dt).days
        except (ValueError, TypeError):
            continue

        if age_days > 14:
            nudges.append({
                "type": "research_stale",
                "priority": "low",
                "message": (
                    f"Your research for {city_name} is {age_days} days old "
                    f"and your trip starts in {days_until} days. "
                    f"Run /research {city_name} to refresh!"
                ),
            })


def _check_budget_drift(state: dict, nudges: list) -> None:
    """Check if spending is trending over budget."""
    cost_tracker = state.get("cost_tracker", {})
    totals = cost_tracker.get("totals", {})

    if totals.get("status") == "over":
        projected = totals.get("projected_total_usd", 0)
        budget_total = cost_tracker.get("budget_total_usd", 0)
        if budget_total and projected:
            over_pct = ((projected - budget_total) / budget_total) * 100
            nudges.append({
                "type": "budget_drift",
                "priority": "medium",
                "message": (
                    f"\u26a0\ufe0f Budget alert: You're projected to spend ${projected:,.0f} "
                    f"({over_pct:.0f}% over your ${budget_total:,.0f} budget). "
                    f"Run /costs for savings tips."
                ),
            })


def _check_unvisited_must_dos(state: dict, nudges: list) -> None:
    """Check if must-do items are running out of time."""
    current_day = state.get("current_trip_day")
    if not current_day:
        return

    dates = state.get("dates", {})
    total_days = dates.get("total_days", 0)
    if not total_days:
        return

    remaining_days = total_days - current_day
    if remaining_days <= 0:
        return

    # Count unvisited must-dos
    priorities = state.get("priorities", {})
    feedback = state.get("feedback_log", [])
    completed_ids = set()
    for f in feedback:
        completed_ids.update(f.get("completed_items", []))

    unvisited_must_dos = 0
    for city_items in priorities.values():
        for item in city_items:
            if item.get("tier") == "must_do" and item.get("item_id") not in completed_ids:
                unvisited_must_dos += 1

    if unvisited_must_dos > remaining_days * 3:  # More than 3 must-dos per remaining day
        nudges.append({
            "type": "must_do_crunch",
            "priority": "medium",
            "message": (
                f"You have {unvisited_must_dos} must-do items left with {remaining_days} days remaining. "
                f"Consider prioritizing or using /plan to rebalance."
            ),
        })


def _check_stale_trip(state: dict, now: datetime, nudges: list) -> None:
    """Check if the trip hasn't been updated in a while."""
    updated_at = state.get("updated_at", "")
    if not updated_at:
        return

    try:
        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        days_since = (now - updated_dt).days
    except (ValueError, TypeError):
        return

    onboarding_complete = state.get("onboarding_complete", False)
    if onboarding_complete and days_since > 7:
        # Check what's the next logical step
        if not state.get("research"):
            nudges.append({
                "type": "stale_trip",
                "priority": "low",
                "message": "Your trip is set up but not researched yet. Run /research all to get started!",
            })
        elif not state.get("high_level_plan"):
            nudges.append({
                "type": "stale_trip",
                "priority": "low",
                "message": "Research is done! Ready to plan? Run /plan to create your itinerary.",
            })


def format_nudge_message(nudges: list[dict]) -> str:
    """Format nudges into a single message for the user."""
    if not nudges:
        return ""

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    nudges.sort(key=lambda n: priority_order.get(n.get("priority", "low"), 2))

    parts = ["Hey! A few things to keep in mind:\n"]
    for nudge in nudges[:5]:  # Cap at 5 nudges
        parts.append(nudge["message"])

    return "\n\n".join(parts)

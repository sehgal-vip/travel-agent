"""Trip synthesis — one-screen trip overview."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.agents.base import format_money

logger = logging.getLogger(__name__)


async def synthesize_trip(state: dict) -> str:
    """Compress entire trip state into a one-screen summary.

    Returns formatted markdown text suitable for Telegram display.
    """
    dest = state.get("destination", {})
    cities = state.get("cities", [])
    dates = state.get("dates", {})
    travelers = state.get("travelers", {})
    budget = state.get("budget", {})
    plan = state.get("high_level_plan", [])
    agenda = state.get("detailed_agenda", [])
    feedback = state.get("feedback_log", [])
    cost_tracker = state.get("cost_tracker", {})
    priorities = state.get("priorities", {})

    flag = dest.get("flag_emoji", "")
    country = dest.get("country", "Unknown")
    currency_code = dest.get("currency_code", "USD")
    currency_symbol = dest.get("currency_symbol", "$")

    parts = []

    # Header
    title = state.get("trip_title", f"{country} Trip")
    parts.append(f"{flag} {title}")
    parts.append("━" * 30)

    # Route overview
    if cities:
        route = " → ".join(f"{c.get('name', '?')} ({c.get('days', '?')}d)" for c in cities)
        parts.append(f"Route: {route}")

    # Dates
    if dates.get("start"):
        parts.append(f"Dates: {dates['start']} to {dates.get('end', '?')} ({dates.get('total_days', '?')} days)")

    # Travelers
    if travelers.get("type"):
        parts.append(f"Travelers: {travelers.get('count', '?')} ({travelers.get('type', '?')})")

    parts.append("")

    # Key highlights per city (top 3 must-dos)
    if priorities:
        parts.append("Must-Dos:")
        for city_name, items in priorities.items():
            must_dos = [i for i in items if i.get("tier") == "must_do"][:3]
            if must_dos:
                names = ", ".join(i.get("name", "?") for i in must_dos)
                parts.append(f"  {city_name}: {names}")
        parts.append("")

    # Plan overview (if exists)
    if plan:
        parts.append("Plan Overview:")
        for day in plan:
            vibe_emoji = {"easy": "🌿", "active": "🏃", "cultural": "🏛️", "foodie": "🍜",
                         "adventurous": "🧗", "relaxed": "😌", "mixed": "🎯"}.get(day.get("vibe", ""), "📅")
            activities = [a.get("name", "?") for a in day.get("key_activities", [])[:2]]
            acts_str = ", ".join(activities) if activities else "free day"
            parts.append(f"  Day {day.get('day', '?')} {vibe_emoji} {day.get('city', '?')}: {acts_str}")
        parts.append("")

    # Budget summary
    if cost_tracker:
        totals = cost_tracker.get("totals", {})
        budget_total = cost_tracker.get("budget_total_usd", 0)
        spent = totals.get("spent_usd", 0)
        remaining = totals.get("remaining_usd", 0)
        daily_avg = totals.get("daily_avg_usd", 0)

        parts.append("Budget:")
        if budget_total:
            parts.append(f"  Total: ${budget_total:,.0f} USD")
        if spent:
            parts.append(f"  Spent: ${spent:,.0f} | Remaining: ${remaining:,.0f}")
        if daily_avg:
            parts.append(f"  Daily avg: ${daily_avg:,.0f}")
        status = totals.get("status", "")
        if status:
            status_emoji = {"on_track": "✅", "over": "⚠️", "under": "💰"}.get(status, "")
            parts.append(f"  Status: {status_emoji} {status}")
        parts.append("")

    # Booking alerts
    booking_alerts = []
    for day in agenda:
        for alert in day.get("booking_alerts", []):
            booking_alerts.append(alert)
    if booking_alerts:
        parts.append("Booking Alerts:")
        for alert in booking_alerts[:5]:
            urgency = alert.get("urgency", "")
            icon = {"book_now": "🔴", "book_today": "🟡"}.get(urgency, "🟢")
            parts.append(f"  {icon} {alert.get('name', '?')} - {alert.get('note', '')}")
        parts.append("")

    # Feedback summary (if on trip)
    if feedback:
        latest = feedback[-1]
        parts.append(f"Latest check-in: Day {latest.get('day', '?')} in {latest.get('city', '?')}")
        if latest.get("highlight"):
            parts.append(f"  Highlight: {latest['highlight']}")
        if latest.get("energy_level"):
            parts.append(f"  Energy: {latest['energy_level']}")
        parts.append("")

    # Key info
    parts.append("Quick Info:")
    if dest.get("currency_code"):
        rate = dest.get("exchange_rate_to_usd", "?")
        parts.append(f"  Currency: {currency_code} ({currency_symbol}), 1 USD = {rate} {currency_code}")
    if dest.get("language"):
        parts.append(f"  Language: {dest['language']}")
    if dest.get("time_zone"):
        parts.append(f"  Timezone: {dest['time_zone']}")

    return "\n".join(parts)

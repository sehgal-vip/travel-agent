"""Drift detector — analyze feedback trends across trip days."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_drift(feedback_log: list[dict]) -> dict:
    """Analyze trends across feedback entries.

    Returns a dict with drift signals that can be used by the scheduler
    to adjust upcoming days.
    """
    if not feedback_log or len(feedback_log) < 2:
        return {}

    signals: dict = {}

    # Energy trend
    energy_levels = [f.get("energy_level", "medium") for f in feedback_log]
    energy_map = {"low": 1, "medium": 2, "high": 3}
    energy_values = [energy_map.get(e, 2) for e in energy_levels]

    recent_energy = energy_values[-3:] if len(energy_values) >= 3 else energy_values
    low_count = sum(1 for e in recent_energy if e == 1)
    if low_count >= 2:
        signals["energy_drift"] = "declining"
        signals["pace_recommendation"] = "lighten"
    elif all(e >= 3 for e in recent_energy):
        signals["energy_drift"] = "high"
        signals["pace_recommendation"] = "maintain"
    else:
        signals["energy_drift"] = "stable"

    # Budget drift
    budget_statuses = [f.get("budget_status", "on_track") for f in feedback_log]
    over_count = sum(1 for b in budget_statuses[-3:] if b == "over")
    if over_count >= 2:
        signals["budget_drift"] = "overspending"
    elif all(b == "under" for b in budget_statuses[-3:]):
        signals["budget_drift"] = "underspending"
    else:
        signals["budget_drift"] = "on_track"

    # Food rating trend
    food_ratings = [f.get("food_rating", "") for f in feedback_log if f.get("food_rating")]
    food_map = {"amazing": 5, "good": 4, "meh": 2, "bad": 1}
    if food_ratings:
        recent_food = [food_map.get(r, 3) for r in food_ratings[-3:]]
        avg_food = sum(recent_food) / len(recent_food)
        if avg_food < 2.5:
            signals["food_drift"] = "declining"
        elif avg_food >= 4:
            signals["food_drift"] = "excellent"
        else:
            signals["food_drift"] = "stable"

    # Preference shifts accumulation
    all_shifts = []
    for f in feedback_log:
        shifts = f.get("preference_shifts", [])
        all_shifts.extend(shifts)
    if all_shifts:
        signals["accumulated_shifts"] = all_shifts[-5:]  # Last 5 shifts

    # Sentiment trend
    sentiments = [f.get("sentiment", "") for f in feedback_log if f.get("sentiment")]
    if sentiments:
        signals["recent_sentiments"] = sentiments[-3:]

    # Activity completion rate
    total_completed = sum(len(f.get("completed_items", [])) for f in feedback_log)
    total_skipped = sum(len(f.get("skipped_items", [])) for f in feedback_log)
    total_planned = total_completed + total_skipped
    if total_planned > 0:
        completion_rate = total_completed / total_planned
        signals["completion_rate"] = round(completion_rate, 2)
        if completion_rate < 0.5:
            signals["schedule_drift"] = "overloaded"
        elif completion_rate > 0.9:
            signals["schedule_drift"] = "well_paced"

    # Discoveries (organic finds vs planned)
    total_discoveries = sum(len(f.get("discoveries", [])) for f in feedback_log)
    if total_discoveries > len(feedback_log):
        signals["discovery_rate"] = "high"  # More discoveries than days = exploring well

    return signals


def format_drift_for_prompt(drift: dict) -> str:
    """Format drift signals into a string for injection into agent prompts."""
    if not drift:
        return ""

    parts = ["FEEDBACK DRIFT SIGNALS:"]

    if drift.get("energy_drift") == "declining":
        parts.append("  ⚠️ Energy declining — schedule lighter days")
    if drift.get("budget_drift") == "overspending":
        parts.append("  ⚠️ Budget trending over — suggest cheaper options")
    if drift.get("food_drift") == "declining":
        parts.append("  ⚠️ Food experiences declining — try different cuisine types")
    if drift.get("schedule_drift") == "overloaded":
        parts.append("  ⚠️ Too many skipped activities — reduce daily items")
    if drift.get("pace_recommendation"):
        parts.append(f"  Pace: {drift['pace_recommendation']}")
    if drift.get("accumulated_shifts"):
        parts.append(f"  Recent preferences: {', '.join(drift['accumulated_shifts'][:3])}")

    return "\n".join(parts) if len(parts) > 1 else ""

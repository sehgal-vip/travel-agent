"""Tests for proactive nudge system."""

import pytest
from datetime import datetime, timezone, timedelta
from src.telegram.nudge import check_nudge_triggers, format_nudge_message


def _make_state(**overrides):
    base = {
        "dates": {"start": "", "end": "", "total_days": 0},
        "research": {},
        "priorities": {},
        "feedback_log": [],
        "cost_tracker": {},
        "onboarding_complete": False,
    }
    base.update(overrides)
    return base


class TestBookingDeadlines:
    def test_no_nudge_if_no_dates(self):
        state = _make_state()
        assert not [n for n in check_nudge_triggers(state) if n["type"] == "booking_deadline"]

    def test_nudge_for_advance_booking_item(self):
        start = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d")
        state = _make_state(
            dates={"start": start, "end": start, "total_days": 5},
            research={
                "Tokyo": {
                    "places": [{"name": "TeamLab", "advance_booking": True}],
                    "activities": [],
                }
            },
        )
        nudges = check_nudge_triggers(state)
        booking = [n for n in nudges if n["type"] == "booking_deadline"]
        assert len(booking) == 1
        assert "TeamLab" in booking[0]["message"]

    def test_no_nudge_if_trip_far_away(self):
        start = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        state = _make_state(
            dates={"start": start, "end": start, "total_days": 5},
            research={
                "Tokyo": {
                    "places": [{"name": "TeamLab", "advance_booking": True}],
                    "activities": [],
                }
            },
        )
        nudges = check_nudge_triggers(state)
        booking = [n for n in nudges if n["type"] == "booking_deadline"]
        assert len(booking) == 0


class TestResearchStaleness:
    def test_stale_research_nudge(self):
        start = (datetime.now(timezone.utc) + timedelta(days=20)).strftime("%Y-%m-%d")
        old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        state = _make_state(
            dates={"start": start, "end": start, "total_days": 5},
            research={"Tokyo": {"last_updated": old_date, "places": [], "activities": []}},
        )
        nudges = check_nudge_triggers(state)
        stale = [n for n in nudges if n["type"] == "research_stale"]
        assert len(stale) == 1


class TestBudgetDrift:
    def test_over_budget_nudge(self):
        state = _make_state(
            cost_tracker={
                "budget_total_usd": 1000,
                "totals": {
                    "status": "over",
                    "projected_total_usd": 1500,
                },
            }
        )
        nudges = check_nudge_triggers(state)
        budget = [n for n in nudges if n["type"] == "budget_drift"]
        assert len(budget) == 1
        assert "50%" in budget[0]["message"]


class TestUnvisitedMustDos:
    def test_must_do_crunch(self):
        state = _make_state(
            current_trip_day=5,
            dates={"start": "2026-03-01", "end": "2026-03-07", "total_days": 7},
            priorities={
                "Tokyo": [
                    {"item_id": f"tokyo-{i}", "tier": "must_do"} for i in range(10)
                ]
            },
            feedback_log=[{"completed_items": ["tokyo-0"]}],
        )
        nudges = check_nudge_triggers(state)
        crunch = [n for n in nudges if n["type"] == "must_do_crunch"]
        assert len(crunch) == 1


class TestFormatNudge:
    def test_empty_nudges(self):
        assert format_nudge_message([]) == ""

    def test_formats_and_sorts_by_priority(self):
        nudges = [
            {"type": "test", "priority": "low", "message": "Low priority"},
            {"type": "test", "priority": "high", "message": "High priority"},
        ]
        result = format_nudge_message(nudges)
        assert result.index("High priority") < result.index("Low priority")

    def test_caps_at_five(self):
        nudges = [{"type": "test", "priority": "low", "message": f"Msg {i}"} for i in range(10)]
        result = format_nudge_message(nudges)
        # Should only have 5 messages (plus the header)
        assert result.count("Msg") == 5

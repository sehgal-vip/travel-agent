"""Tests for /mytrips — helpers, routing, and display logic."""

from __future__ import annotations

import pytest

from src.agents.onboarding import _fallback_title
from src.agents.orchestrator import COMMAND_MAP, generate_help
from src.telegram.handlers import (
    _fallback_title_from_state,
    _format_date_range,
    _trip_progress_indicator,
)


class TestFallbackTitle:
    def test_fallback_title_with_cities(self):
        dest = {"country": "Japan"}
        cities = [{"name": "Tokyo"}, {"name": "Kyoto"}, {"name": "Osaka"}]
        assert _fallback_title(dest, cities) == "Japan: Tokyo & Kyoto +1"

    def test_fallback_title_no_cities(self):
        dest = {"country": "Japan"}
        assert _fallback_title(dest, []) == "Japan Trip"

    def test_fallback_title_two_cities(self):
        dest = {"country": "Morocco"}
        cities = [{"name": "Marrakech"}, {"name": "Fes"}]
        assert _fallback_title(dest, cities) == "Morocco: Marrakech & Fes"

    def test_fallback_title_empty_dest(self):
        assert _fallback_title({}, []) == "Adventure Trip"

    def test_fallback_title_from_state_with_cities(self):
        state = {
            "destination": {"country": "Japan"},
            "cities": [{"name": "Tokyo"}, {"name": "Kyoto"}, {"name": "Osaka"}],
        }
        assert _fallback_title_from_state(state) == "Japan: Tokyo & Kyoto +1"

    def test_fallback_title_from_state_no_cities(self):
        state = {"destination": {"country": "Japan"}, "cities": []}
        assert _fallback_title_from_state(state) == "Japan Trip"


class TestFormatDateRange:
    def test_same_month(self):
        assert _format_date_range("2026-04-01", "2026-04-14") == "Apr 1-14"

    def test_cross_month(self):
        assert _format_date_range("2026-03-28", "2026-04-05") == "Mar 28 - Apr 5"

    def test_empty_start(self):
        assert _format_date_range("", "2026-04-14") == ""

    def test_empty_end(self):
        assert _format_date_range("2026-04-01", "") == ""

    def test_invalid_dates(self):
        assert _format_date_range("not-a-date", "also-not") == ""


class TestTripProgress:
    def test_not_onboarded(self):
        state = {"onboarding_complete": False}
        assert _trip_progress_indicator(state) == "Setting up"

    def test_ready_to_research(self):
        state = {"onboarding_complete": True}
        assert _trip_progress_indicator(state) == "Ready to research"

    def test_ready_to_prioritize(self):
        state = {"onboarding_complete": True, "research": {"Tokyo": {}}}
        assert _trip_progress_indicator(state) == "Ready to prioritize"

    def test_researched(self):
        state = {"onboarding_complete": True, "research": {"Tokyo": {}}, "priorities": {"Tokyo": []}}
        assert _trip_progress_indicator(state) == "Researched"

    def test_planned(self):
        state = {
            "onboarding_complete": True,
            "research": {"Tokyo": {}},
            "priorities": {"Tokyo": []},
            "high_level_plan": [{"day": 1}],
        }
        assert _trip_progress_indicator(state) == "Planned"

    def test_scheduled(self):
        state = {
            "onboarding_complete": True,
            "research": {"Tokyo": {}},
            "priorities": {"Tokyo": []},
            "high_level_plan": [{"day": 1}],
            "detailed_agenda": [{"day": 1}],
        }
        assert _trip_progress_indicator(state) == "Scheduled"


class TestMytripsRouting:
    def test_mytrips_command_in_command_map(self):
        assert "/mytrips" in COMMAND_MAP
        assert COMMAND_MAP["/mytrips"] == "orchestrator"

    def test_mytrips_in_help_text(self):
        help_text = generate_help()
        assert "/mytrips" in help_text

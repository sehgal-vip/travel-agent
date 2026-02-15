"""Tests for drift detection tool."""

import pytest
from src.tools.drift_detector import detect_drift, format_drift_for_prompt


class TestDetectDrift:
    def test_empty_feedback(self):
        assert detect_drift([]) == {}

    def test_single_entry(self):
        assert detect_drift([{"energy_level": "low"}]) == {}

    def test_energy_declining(self):
        feedback = [
            {"energy_level": "high", "budget_status": "on_track"},
            {"energy_level": "low", "budget_status": "on_track"},
            {"energy_level": "low", "budget_status": "on_track"},
        ]
        drift = detect_drift(feedback)
        assert drift["energy_drift"] == "declining"
        assert drift["pace_recommendation"] == "lighten"

    def test_budget_overspending(self):
        feedback = [
            {"energy_level": "medium", "budget_status": "over"},
            {"energy_level": "medium", "budget_status": "over"},
            {"energy_level": "medium", "budget_status": "on_track"},
        ]
        drift = detect_drift(feedback)
        assert drift["budget_drift"] == "overspending"

    def test_high_energy_maintained(self):
        feedback = [
            {"energy_level": "high", "budget_status": "on_track"},
            {"energy_level": "high", "budget_status": "on_track"},
            {"energy_level": "high", "budget_status": "on_track"},
        ]
        drift = detect_drift(feedback)
        assert drift["energy_drift"] == "high"
        assert drift["pace_recommendation"] == "maintain"

    def test_completion_rate_overloaded(self):
        feedback = [
            {"energy_level": "medium", "budget_status": "on_track",
             "completed_items": ["a"], "skipped_items": ["b", "c", "d"]},
            {"energy_level": "medium", "budget_status": "on_track",
             "completed_items": ["e"], "skipped_items": ["f", "g"]},
        ]
        drift = detect_drift(feedback)
        assert drift["completion_rate"] < 0.5
        assert drift["schedule_drift"] == "overloaded"

    def test_food_drift_declining(self):
        feedback = [
            {"energy_level": "medium", "budget_status": "on_track", "food_rating": "meh"},
            {"energy_level": "medium", "budget_status": "on_track", "food_rating": "bad"},
            {"energy_level": "medium", "budget_status": "on_track", "food_rating": "meh"},
        ]
        drift = detect_drift(feedback)
        assert drift["food_drift"] == "declining"

    def test_preference_shifts_accumulated(self):
        feedback = [
            {"energy_level": "medium", "budget_status": "on_track", "preference_shifts": ["more temples"]},
            {"energy_level": "medium", "budget_status": "on_track", "preference_shifts": ["less shopping", "more food"]},
        ]
        drift = detect_drift(feedback)
        assert "more food" in drift["accumulated_shifts"]

    def test_discovery_rate_high(self):
        feedback = [
            {"energy_level": "medium", "budget_status": "on_track", "discoveries": ["cool bar", "hidden temple"]},
            {"energy_level": "medium", "budget_status": "on_track", "discoveries": ["street food spot"]},
        ]
        drift = detect_drift(feedback)
        assert drift["discovery_rate"] == "high"


class TestFormatDrift:
    def test_empty_drift(self):
        assert format_drift_for_prompt({}) == ""

    def test_energy_warning(self):
        result = format_drift_for_prompt({"energy_drift": "declining", "pace_recommendation": "lighten"})
        assert "Energy declining" in result
        assert "lighten" in result

    def test_budget_warning(self):
        result = format_drift_for_prompt({"budget_drift": "overspending"})
        assert "Budget trending over" in result

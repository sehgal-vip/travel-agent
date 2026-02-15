"""Tests for the library_sync tool function (v2)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.tools.library_sync import library_sync


@pytest.fixture
def sync_state():
    """Minimal state for library sync testing."""
    return {
        "destination": {
            "country": "Japan",
            "country_code": "JP",
            "flag_emoji": "\U0001f1ef\U0001f1f5",
            "currency_code": "JPY",
            "currency_symbol": "\u00a5",
            "exchange_rate_to_usd": 148.0,
            "researched_at": "2026-02-01T10:00:00Z",
            "language": "Japanese",
            "useful_phrases": {"thank you": "arigatou"},
            "payment_norms": "Cash preferred",
            "tipping_culture": "Not customary",
            "safety_notes": [],
            "common_scams": [],
            "emergency_numbers": {"police": "110"},
            "cultural_notes": [],
            "transport_apps": [],
            "common_intercity_transport": [],
            "booking_platforms": [],
            "plug_type": "A/B",
            "voltage": "100V",
            "time_zone": "Asia/Tokyo",
            "climate_type": "temperate",
            "current_season_notes": "",
            "health_advisories": [],
            "daily_budget_benchmarks": {},
            "sim_connectivity": "",
        },
        "cities": [
            {"name": "Tokyo", "country": "Japan", "days": 4, "order": 1},
        ],
        "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
        "research": {
            "Tokyo": {
                "places": [{"id": "t-1", "name": "Senso-ji", "category": "place"}],
                "activities": [],
                "food": [],
                "logistics": [],
                "tips": [],
                "hidden_gems": [],
                "last_updated": "2026-02-01T12:00:00Z",
            }
        },
        "priorities": {},
        "high_level_plan": [],
        "feedback_log": [],
        "cost_tracker": {},
        "library": {},
    }


class TestLibrarySync:
    """Tests for the library_sync tool function."""

    @pytest.mark.asyncio
    async def test_creates_workspace(self, sync_state, tmp_path):
        """Library sync creates workspace folder."""
        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance
            mock_instance.create_workspace.return_value = str(tmp_path / "japan-2026")
            mock_instance.write_city_research.return_value = 1

            result = await library_sync(sync_state)

        assert "library" in result
        assert result["library"]["workspace_path"] is not None
        mock_instance.create_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_destination_guide(self, sync_state, tmp_path):
        """Library sync writes destination guide when researched_at is set."""
        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance
            mock_instance.create_workspace.return_value = str(tmp_path / "japan-2026")
            mock_instance.write_city_research.return_value = 1

            result = await library_sync(sync_state)

        assert result["library"]["guide_written"] is True
        mock_instance.write_destination_guide.assert_called_once()

    @pytest.mark.asyncio
    async def test_syncs_city_research(self, sync_state, tmp_path):
        """Library sync writes city research files."""
        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance
            mock_instance.create_workspace.return_value = str(tmp_path / "japan-2026")
            mock_instance.write_city_research.return_value = 5

            result = await library_sync(sync_state)

        assert "Tokyo" in result["library"]["synced_cities"]
        mock_instance.write_city_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_response_and_library(self, sync_state, tmp_path):
        """Library sync returns both response and library config."""
        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance
            mock_instance.create_workspace.return_value = str(tmp_path / "japan-2026")
            mock_instance.write_city_research.return_value = 1

            result = await library_sync(sync_state)

        assert "response" in result
        assert "library" in result
        assert result["library"]["last_synced"] is not None

    @pytest.mark.asyncio
    async def test_skips_guide_if_already_written(self, sync_state, tmp_path):
        """Library sync skips guide if already written."""
        sync_state["library"] = {
            "workspace_path": str(tmp_path / "japan-2026"),
            "guide_written": True,
            "synced_cities": {},
            "feedback_days_written": [],
        }

        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance
            mock_instance.write_city_research.return_value = 1

            result = await library_sync(sync_state)

        mock_instance.create_workspace.assert_not_called()
        mock_instance.write_destination_guide.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_research_still_updates_index(self, sync_state, tmp_path):
        """Even without research, index is updated."""
        sync_state["research"] = {}
        sync_state["library"] = {
            "workspace_path": str(tmp_path / "japan-2026"),
            "guide_written": True,
            "synced_cities": {},
            "feedback_days_written": [],
        }

        with patch("src.tools.library_sync.MarkdownLibrary") as MockLib:
            mock_instance = MagicMock()
            MockLib.return_value = mock_instance

            result = await library_sync(sync_state)

        mock_instance.write_index.assert_called_once()

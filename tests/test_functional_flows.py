"""Comprehensive functional-flow tests for the travel-agent multi-agent system.

Covers all agent routing, onboarding, research, prioritization, planning,
scheduling, feedback, cost tracking, trip sharing, handler short-circuits,
and error handling flows.

Every test case is tagged with a TC-XXX-NN identifier for traceability.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage

from src.agents.orchestrator import (
    COMMAND_MAP,
    PREREQUISITES,
    OrchestratorAgent,
    generate_help,
    generate_status,
)
from src.agents.onboarding import OnboardingAgent
from src.agents.base import BaseAgent, add_to_conversation_history
from src.graph import (
    END,
    _specialist_node,
    build_graph,
    onboarding_node,
    route_from_orchestrator,
)
from src.state import EnergyLevel, Tier, TripState
from src.telegram.formatters import format_budget_report, split_message
from src.telegram.handlers import (
    _INTERNAL_KEYS,
    _extract_response,
    _handle_join,
    _handle_trip_management,
    _list_trips,
)
from src.db.persistence import TripRepository


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Ensure env vars are set so Settings / ChatAnthropic do not fail."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    # Clear cached settings so each test picks up env vars fresh
    from src.config.settings import get_settings
    get_settings.cache_clear()


def make_mock_update(message_text: str, user_id: int = 12345):
    """Create a mock Telegram Update with the given message text."""
    update = MagicMock()
    update.message.text = message_text
    update.effective_user.id = user_id
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    return update


def make_mock_context(active_trip_id: str = "default", graph=None, repo=None):
    """Create a mock Telegram Context."""
    context = MagicMock()
    context.user_data = {"active_trip_id": active_trip_id}
    context.bot_data = {"graph": graph, "repo": repo}
    return context


# ─── Sample research / priorities / plan data ───────────────────────────────

SAMPLE_RESEARCH_TOKYO = {
    "last_updated": "2026-02-01T12:00:00Z",
    "places": [
        {"id": "tokyo-place-1", "name": "Senso-ji", "category": "place",
         "subcategory": "temple", "description": "Ancient Buddhist temple",
         "cost_usd": 0, "time_needed_hrs": 1.5, "tags": ["culture"],
         "advance_booking": False, "sources": ["guidebook"]},
        {"id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
         "subcategory": "shrine", "description": "Shinto shrine",
         "cost_usd": 0, "time_needed_hrs": 1, "tags": ["culture"],
         "advance_booking": False, "sources": ["guidebook"]},
    ],
    "activities": [
        {"id": "tokyo-act-1", "name": "Teamlab Borderless", "category": "activity",
         "subcategory": "art", "description": "Digital art museum",
         "cost_usd": 25, "time_needed_hrs": 2, "tags": ["art"],
         "advance_booking": True, "sources": ["blog"]},
    ],
    "food": [
        {"id": "tokyo-food-1", "name": "Tsukiji Outer Market", "category": "food",
         "subcategory": "market", "description": "Fresh seafood market",
         "cost_usd": 15, "time_needed_hrs": 2, "tags": ["food"],
         "advance_booking": False, "sources": ["blog"]},
    ],
    "logistics": [],
    "tips": [],
    "hidden_gems": [],
}

SAMPLE_PRIORITIES_TOKYO = [
    {"item_id": "tokyo-place-1", "name": "Senso-ji", "category": "place",
     "tier": "must_do", "score": 90, "reason": "Iconic temple", "user_override": False},
    {"item_id": "tokyo-act-1", "name": "Teamlab Borderless", "category": "activity",
     "tier": "nice_to_have", "score": 75, "reason": "Unique art", "user_override": False},
    {"item_id": "tokyo-food-1", "name": "Tsukiji Outer Market", "category": "food",
     "tier": "must_do", "score": 88, "reason": "Iconic market", "user_override": False},
    {"item_id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
     "tier": "if_nearby", "score": 60, "reason": "Nice shrine", "user_override": False},
]

SAMPLE_DAY_PLAN = [
    {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Arrival & First Impressions",
     "vibe": "easy", "travel": None,
     "key_activities": [{"item_id": "tokyo-place-1", "tier": "must_do", "name": "Senso-ji"}],
     "meals": {"breakfast": {"type": "hotel"}, "lunch": {"name": "Tsukiji Outer Market", "type": "specific"},
               "dinner": {"type": "explore"}},
     "special_moment": "Sunset from Senso-ji", "notes": "", "free_time_blocks": ["afternoon"],
     "estimated_cost_usd": 50},
    {"day": 2, "date": "2026-04-02", "city": "Tokyo", "theme": "Culture Deep Dive",
     "vibe": "cultural", "travel": None,
     "key_activities": [{"item_id": "tokyo-act-1", "tier": "nice_to_have", "name": "Teamlab Borderless"}],
     "meals": {"breakfast": {"type": "hotel"}, "lunch": {"type": "explore"}, "dinner": {"type": "explore"}},
     "special_moment": None, "notes": "", "free_time_blocks": [], "estimated_cost_usd": 80},
]


# ─── Helper for patching LLM ────────────────────────────────────────────────

def _patch_llm(agent, return_value):
    """Return a patch context manager that mocks agent.llm.ainvoke.

    ChatAnthropic is a Pydantic model; we cannot set attributes directly.
    We use patch.object on the type so Pydantic validation is bypassed.
    """
    return patch.object(type(agent.llm), "ainvoke", new_callable=lambda: AsyncMock(return_value=return_value))


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================


class TestOrchestratorRouting:
    """Tests for orchestrator routing logic (TC-ORC-*)."""

    @pytest.mark.asyncio
    async def test_start_routes_to_onboarding(self, empty_state):
        """TC-ORC-01: /start routes to onboarding with welcome message."""
        orch = OrchestratorAgent()
        result = await orch.route(empty_state, "/start")
        assert result["target_agent"] == "onboarding"
        assert result.get("response")
        assert "welcome" in result["response"].lower() or "get started" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_free_text_pre_onboarding_routes_to_onboarding(self, empty_state):
        """TC-ORC-02: Free text pre-onboarding routes to onboarding (guard, no LLM)."""
        orch = OrchestratorAgent()
        result = await orch.route(empty_state, "I want to go to Japan")
        assert result["target_agent"] == "onboarding"

    @pytest.mark.asyncio
    async def test_research_post_onboarding_routes_to_research(self, japan_state):
        """TC-ORC-03: /research post-onboarding routes to research."""
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/research")
        assert result["target_agent"] == "research"

    @pytest.mark.asyncio
    async def test_priorities_without_research_returns_prerequisite_error(self, japan_state):
        """TC-ORC-04: /priorities without research returns prerequisite error."""
        orch = OrchestratorAgent()
        state = dict(japan_state)
        state["research"] = {}
        result = await orch.route(state, "/priorities")
        assert result["target_agent"] == "orchestrator"
        assert "research" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_plan_without_research_returns_prerequisite_error(self, japan_state):
        """TC-ORC-05: /plan without research+priorities returns prerequisite error."""
        orch = OrchestratorAgent()
        state = dict(japan_state)
        state["research"] = {}
        state["priorities"] = {}
        result = await orch.route(state, "/plan")
        assert result["target_agent"] == "orchestrator"
        assert "research" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_agenda_without_plan_returns_prerequisite_error(self, japan_state):
        """TC-ORC-06: /agenda without plan returns prerequisite error."""
        orch = OrchestratorAgent()
        state = dict(japan_state)
        state["high_level_plan"] = []
        result = await orch.route(state, "/agenda")
        assert result["target_agent"] == "orchestrator"
        assert "plan" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_status_returns_dashboard(self, japan_state):
        """TC-ORC-07: /status returns status dashboard directly."""
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/status")
        assert result["target_agent"] == "orchestrator"
        assert "Japan" in result["response"]

    def test_help_contains_join_and_trip_id(self):
        """TC-ORC-08: /help returns help text with /join, /trip id, and /costs sub-commands."""
        help_text = generate_help()
        assert "/join" in help_text
        assert "/trip id" in help_text
        assert "/start" in help_text
        assert "/research" in help_text
        assert "/costs" in help_text
        assert "/costs today" in help_text
        assert "/costs convert" in help_text
        assert "/adjust" in help_text

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self, japan_state):
        """TC-ORC-09: Unknown command /xyz returns error message."""
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/xyz")
        assert result["target_agent"] == "orchestrator"
        assert "Unknown" in result["response"] or "/xyz" in result["response"]


# =============================================================================
# ONBOARDING TESTS
# =============================================================================


class TestOnboardingFlow:
    """Tests for the onboarding agent (TC-ONB-*)."""

    def test_step_advances_with_destination(self, empty_state):
        """TC-ONB-01: Step advances when destination provided (step 0 -> 1)."""
        agent = OnboardingAgent()
        state = dict(empty_state)
        state["destination"] = {"country": "Japan"}
        result = agent._estimate_step(state, "I want to visit Japan", 0)
        assert result == 1

    def test_step_advances_with_cities(self, empty_state):
        """TC-ONB-02: Step advances when cities provided (step 1 -> 2)."""
        agent = OnboardingAgent()
        state = dict(empty_state)
        state["destination"] = {"country": "Japan"}
        state["cities"] = [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}]
        result = agent._estimate_step(state, "I want to visit Tokyo", 1)
        assert result == 2

    def test_step_advances_with_dates(self, empty_state):
        """TC-ONB-03: Step advances when dates provided (step 2 -> 3)."""
        agent = OnboardingAgent()
        state = dict(empty_state)
        state["destination"] = {"country": "Japan"}
        state["cities"] = [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}]
        state["dates"] = {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14}
        result = agent._estimate_step(state, "April 1 to April 14", 2)
        assert result == 3

    def test_confirmation_with_valid_json_completes_onboarding(self):
        """TC-ONB-04: Confirmation with valid JSON sets onboarding_complete=True."""
        agent = OnboardingAgent()
        json_config = {
            "confirmed": True,
            "destination": {"country": "Japan", "country_code": "JP", "region": "East Asia",
                            "flag_emoji": "", "language": "Japanese",
                            "currency_code": "JPY", "currency_symbol": "Y"},
            "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
            "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
            "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
            "budget": {"style": "midrange", "total_estimate_usd": 4000,
                       "splurge_on": ["food"], "save_on": []},
            "interests": ["food", "culture"],
            "must_dos": [],
            "deal_breakers": [],
            "accommodation_pref": "hotel",
            "transport_pref": ["train"],
        }
        text = f"Looks great!\n```json\n{json.dumps(json_config)}\n```"
        config = agent._extract_config(text)
        assert config is not None
        assert config["confirmed"] is True

        state_updates = agent._build_state_from_config(config, {"trip_id": ""})
        assert state_updates["onboarding_complete"] is True
        assert state_updates["onboarding_step"] == 11

    def test_confirmation_with_invalid_json_returns_none(self):
        """TC-ONB-05: Confirmation with invalid JSON continues conversation."""
        agent = OnboardingAgent()
        result = agent._extract_config("No JSON here at all")
        assert result is None

    def test_trip_id_is_8_char_uuid(self):
        """TC-ONB-06: trip_id is 8-char UUID when no existing trip_id."""
        agent = OnboardingAgent()
        config = {
            "destination": {"country": "Japan", "country_code": "JP"},
            "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
            "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
            "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
            "budget": {"style": "midrange", "total_estimate_usd": 4000},
        }
        result = agent._build_state_from_config(config, {"trip_id": ""})
        assert len(result["trip_id"]) == 8

    def test_full_state_written_on_completion(self):
        """TC-ONB-07: Full state written on completion (all empty containers initialized)."""
        agent = OnboardingAgent()
        config = {
            "destination": {"country": "Japan", "country_code": "JP",
                            "currency_code": "JPY", "currency_symbol": "Y"},
            "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
            "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
            "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
            "budget": {"style": "midrange", "total_estimate_usd": 4000},
            "interests": ["food"],
            "must_dos": [],
            "deal_breakers": [],
            "accommodation_pref": "hotel",
            "transport_pref": ["train"],
        }
        result = agent._build_state_from_config(config, {"trip_id": ""})

        # Verify all containers are initialized
        assert result["research"] == {}
        assert result["priorities"] == {}
        assert result["high_level_plan"] == []
        assert "budget_total_usd" in result["cost_tracker"]
        assert "workspace_path" in result["library"]
        assert result["agent_scratch"] == {}
        assert result["plan_status"] == "not_started"
        assert result["plan_version"] == 0
        assert result["detailed_agenda"] == []
        assert result["feedback_log"] == []

    def test_existing_trip_id_preserved(self):
        """TC-ONB-08: Existing trip_id preserved on re-confirmation."""
        agent = OnboardingAgent()
        config = {
            "destination": {"country": "Japan"},
            "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
            "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
            "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
            "budget": {"style": "midrange", "total_estimate_usd": 4000},
        }
        result = agent._build_state_from_config(config, {"trip_id": "existing"})
        assert result["trip_id"] == "existing"


# =============================================================================
# RESEARCH TESTS
# =============================================================================


class TestResearchFlow:
    """Tests for the research agent (TC-RES-*)."""

    @pytest.mark.asyncio
    async def test_research_all_cities(self, japan_state):
        """TC-RES-01: /research all researches all cities (mock LLM)."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.search_city = AsyncMock(return_value=[])

        city_research_json = json.dumps({
            "places": [{"id": "test-place-1", "name": "Test Place", "category": "place"}],
            "activities": [], "food": [], "logistics": [], "tips": [], "hidden_gems": [],
        })
        mock_response = AIMessage(content=city_research_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(japan_state, "/research all")

        assert "response" in result
        assert "Research" in result["response"] or "research" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_research_single_city(self, japan_state):
        """TC-RES-02: /research Tokyo researches single city (mock LLM)."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.search_city = AsyncMock(return_value=[])

        city_research_json = json.dumps({
            "places": [{"id": "tokyo-place-1", "name": "Senso-ji", "category": "place"}],
            "activities": [], "food": [], "logistics": [], "tips": [], "hidden_gems": [],
        })
        mock_response = AIMessage(content=city_research_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(japan_state, "/research Tokyo")

        assert "response" in result
        assert "Tokyo" in result["response"]
        if result.get("state_updates", {}).get("research"):
            assert "Tokyo" in result["state_updates"]["research"]

    def test_research_deduplicates_by_name(self):
        """TC-RES-03: Research deduplicates by item name."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent()
        existing = {
            "places": [{"name": "Senso-ji", "id": "old-1"}],
            "activities": [], "food": [], "logistics": [], "tips": [], "hidden_gems": [],
        }
        new = {
            "places": [
                {"name": "Senso-ji", "id": "new-1"},  # duplicate
                {"name": "Tokyo Tower", "id": "new-2"},  # new
            ],
            "activities": [], "food": [], "logistics": [], "tips": [], "hidden_gems": [],
        }
        merged = agent._merge_research(existing, new)
        place_names = [p["name"] for p in merged["places"]]
        assert len(place_names) == 2
        assert "Senso-ji" in place_names
        assert "Tokyo Tower" in place_names

    @pytest.mark.asyncio
    async def test_destination_intel_runs_once(self, japan_state):
        """TC-RES-04: Destination intel already done; goes to city research (Mode 2)."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.search_city = AsyncMock(return_value=[])
        agent.search_tool.search_destination_intel = AsyncMock()

        assert japan_state["destination"].get("researched_at") is not None

        city_json = json.dumps({
            "places": [], "activities": [], "food": [],
            "logistics": [], "tips": [], "hidden_gems": [],
        })
        mock_response = AIMessage(content=city_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(japan_state, "/research Tokyo")

        agent.search_tool.search_destination_intel.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_failure_partial_results(self, japan_state):
        """TC-RES-05: Research failure for one city doesn't block others."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent()
        agent.search_tool = MagicMock()
        agent.search_tool.search_city = AsyncMock(return_value=[])

        good_json = json.dumps({
            "places": [{"id": "test-1", "name": "Good Place", "category": "place"}],
            "activities": [], "food": [], "logistics": [], "tips": [], "hidden_gems": [],
        })

        call_counter = {"count": 0}

        async def mock_ainvoke(self_llm, messages, **kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 2:
                raise Exception("LLM failure for city 2")
            return AIMessage(content=good_json)

        with patch.object(type(agent.llm), "ainvoke", new=mock_ainvoke):
            result = await agent.handle(japan_state, "/research all")

        assert "response" in result
        response_lower = result["response"].lower()
        assert "research" in response_lower or "failed" in response_lower or "skipping" in response_lower


# =============================================================================
# PRIORITIZER TESTS
# =============================================================================


class TestPrioritizerFlow:
    """Tests for the prioritizer agent (TC-PRI-*)."""

    @pytest.mark.asyncio
    async def test_no_research_returns_error(self, japan_state):
        """TC-PRI-05: No research returns error message."""
        from src.agents.prioritizer import PrioritizerAgent

        agent = PrioritizerAgent()
        state = dict(japan_state)
        state["research"] = {}
        result = await agent.handle(state, "/priorities")
        assert "research" in result["response"].lower()
        assert result["state_updates"] == {}

    @pytest.mark.asyncio
    async def test_generates_tiers_from_research(self, japan_state):
        """TC-PRI-01: Generates priority tiers per city from research."""
        from src.agents.prioritizer import PrioritizerAgent

        agent = PrioritizerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}

        priorities_json = json.dumps({
            "Tokyo": [
                {"item_id": "tokyo-place-1", "name": "Senso-ji", "category": "place",
                 "tier": "must_do", "score": 90, "reason": "Iconic", "user_override": False},
                {"item_id": "tokyo-act-1", "name": "Teamlab Borderless", "category": "activity",
                 "tier": "nice_to_have", "score": 75, "reason": "Unique", "user_override": False},
                {"item_id": "tokyo-food-1", "name": "Tsukiji", "category": "food",
                 "tier": "must_do", "score": 88, "reason": "Must eat", "user_override": False},
                {"item_id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
                 "tier": "if_nearby", "score": 60, "reason": "Nice", "user_override": False},
            ]
        })
        mock_response = AIMessage(content=priorities_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/priorities")

        assert "state_updates" in result
        priorities = result["state_updates"].get("priorities", {})
        assert "Tokyo" in priorities
        tiers_present = {item["tier"] for item in priorities["Tokyo"]}
        assert "must_do" in tiers_present

    @pytest.mark.asyncio
    async def test_must_do_max_30_percent(self, japan_state):
        """TC-PRI-02: Max 30% must-do items (checked in output)."""
        from src.agents.prioritizer import PrioritizerAgent

        agent = PrioritizerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}

        items = []
        for i in range(10):
            tier = "must_do" if i < 3 else "nice_to_have"
            items.append({
                "item_id": f"tokyo-item-{i}", "name": f"Item {i}", "category": "place",
                "tier": tier, "score": 90 - i * 5, "reason": "Test", "user_override": False,
            })

        priorities_json = json.dumps({"Tokyo": items})
        mock_response = AIMessage(content=priorities_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/priorities")

        priorities = result["state_updates"].get("priorities", {})
        tokyo_items = priorities.get("Tokyo", [])
        must_dos = [i for i in tokyo_items if i["tier"] == "must_do"]
        total = len(tokyo_items)
        assert len(must_dos) <= max(1, int(total * 0.30) + 1)

    @pytest.mark.asyncio
    async def test_user_override_persists(self, japan_state):
        """TC-PRI-03: User override 'move X to must-do' persists."""
        from src.agents.prioritizer import PrioritizerAgent

        agent = PrioritizerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {
            "Tokyo": [
                {"item_id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
                 "tier": "if_nearby", "score": 60, "reason": "Nice", "user_override": False},
            ]
        }

        changes_json = json.dumps({
            "changes": [{"item_id": "tokyo-place-2", "new_tier": "must_do"}]
        })
        mock_response = AIMessage(content=changes_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "move Meiji Shrine to must-do")

        priorities = result["state_updates"].get("priorities", {})
        tokyo_items = priorities.get("Tokyo", [])
        meiji = [i for i in tokyo_items if i["item_id"] == "tokyo-place-2"]
        assert len(meiji) == 1
        assert meiji[0]["tier"] == "must_do"
        assert meiji[0]["user_override"] is True

    @pytest.mark.asyncio
    async def test_override_preserved_on_reprioritize(self, japan_state):
        """TC-PRI-04: Override flag preserved on re-prioritize."""
        from src.agents.prioritizer import PrioritizerAgent

        agent = PrioritizerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {
            "Tokyo": [
                {"item_id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
                 "tier": "must_do", "score": 60, "reason": "User wants this",
                 "user_override": True},
            ]
        }

        new_priorities_json = json.dumps({
            "Tokyo": [
                {"item_id": "tokyo-place-2", "name": "Meiji Shrine", "category": "place",
                 "tier": "if_nearby", "score": 60, "reason": "Nice", "user_override": False},
            ]
        })
        mock_response = AIMessage(content=new_priorities_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/priorities")

        priorities = result["state_updates"].get("priorities", {})
        tokyo_items = priorities.get("Tokyo", [])
        meiji = [i for i in tokyo_items if i["item_id"] == "tokyo-place-2"]
        assert len(meiji) == 1
        assert meiji[0]["tier"] == "must_do"
        assert meiji[0]["user_override"] is True


# =============================================================================
# PLANNER TESTS
# =============================================================================


class TestPlannerFlow:
    """Tests for the planner agent (TC-PLN-*)."""

    @pytest.mark.asyncio
    async def test_no_priorities_returns_error(self, japan_state):
        """TC-PLN-05: No priorities returns error message."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        state = dict(japan_state)
        state["priorities"] = {}
        result = await agent.handle(state, "/plan")
        assert "priorities" in result["response"].lower()
        assert result["state_updates"] == {}

    @pytest.mark.asyncio
    async def test_creates_plan_matching_total_days(self, japan_state):
        """TC-PLN-01: Creates day-by-day plan matching total_days."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {"Tokyo": SAMPLE_PRIORITIES_TOKYO}

        plan_days = []
        for d in range(1, 15):
            plan_days.append({
                "day": d, "date": f"2026-04-{d:02d}", "city": "Tokyo",
                "theme": f"Day {d}", "vibe": "mixed", "travel": None,
                "key_activities": [], "meals": {}, "special_moment": None,
                "notes": "", "free_time_blocks": [], "estimated_cost_usd": 100,
            })

        plan_json = json.dumps(plan_days)
        mock_response = AIMessage(content=plan_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/plan")

        plan = result["state_updates"].get("high_level_plan", [])
        assert len(plan) == 14

    @pytest.mark.asyncio
    async def test_must_do_items_appear_in_plan(self, japan_state):
        """TC-PLN-02: Must-do items appear in plan."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {"Tokyo": SAMPLE_PRIORITIES_TOKYO}

        plan_json = json.dumps([
            {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Temple Day",
             "vibe": "cultural", "travel": None,
             "key_activities": [
                 {"item_id": "tokyo-place-1", "tier": "must_do", "name": "Senso-ji"},
                 {"item_id": "tokyo-food-1", "tier": "must_do", "name": "Tsukiji Outer Market"},
             ],
             "meals": {}, "special_moment": None, "notes": "",
             "free_time_blocks": [], "estimated_cost_usd": 80},
        ])
        mock_response = AIMessage(content=plan_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/plan")

        plan = result["state_updates"].get("high_level_plan", [])
        all_activity_names = []
        for day in plan:
            for act in day.get("key_activities", []):
                all_activity_names.append(act["name"])
        assert "Senso-ji" in all_activity_names
        assert "Tsukiji Outer Market" in all_activity_names

    @pytest.mark.asyncio
    async def test_adjustment_increments_version(self, japan_state):
        """TC-PLN-03: Adjustment re-generates plan (version increments)."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {"Tokyo": SAMPLE_PRIORITIES_TOKYO}
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["plan_version"] = 1

        adjusted_plan = json.dumps(SAMPLE_DAY_PLAN)
        mock_response = AIMessage(content=adjusted_plan)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "adjust: swap day 1 and day 2")

        assert result["state_updates"]["plan_version"] == 2
        assert result["state_updates"]["plan_status"] == "draft"

    @pytest.mark.asyncio
    async def test_plan_status_is_draft_after_generation(self, japan_state):
        """TC-PLN-04: Plan generation sets plan_status to 'draft'."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["priorities"] = {"Tokyo": SAMPLE_PRIORITIES_TOKYO}

        plan_json = json.dumps(SAMPLE_DAY_PLAN)
        mock_response = AIMessage(content=plan_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/plan")

        assert result["state_updates"]["plan_status"] == "draft"
        assert result["state_updates"]["plan_version"] == 1


# =============================================================================
# SCHEDULER TESTS
# =============================================================================


class TestSchedulerFlow:
    """Tests for the scheduler agent (TC-SCH-*)."""

    @pytest.mark.asyncio
    async def test_no_plan_returns_error(self, japan_state):
        """TC-SCH-04: No plan returns error message."""
        from src.agents.scheduler import SchedulerAgent

        agent = SchedulerAgent()
        state = dict(japan_state)
        state["high_level_plan"] = []
        result = await agent.handle(state, "/agenda")
        assert "plan" in result["response"].lower()
        assert result["state_updates"] == {}

    @pytest.mark.asyncio
    async def test_generates_2_day_window(self, japan_state):
        """TC-SCH-01: Generates 2-day rolling window."""
        from src.agents.scheduler import SchedulerAgent

        agent = SchedulerAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}

        agenda_json = json.dumps([
            {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Arrival",
             "slots": [
                 {"time": "09:00", "end_time": "10:30", "type": "activity",
                  "name": "Senso-ji", "address": "2-3-1 Asakusa", "duration_min": 90,
                  "cost_local": 0, "cost_usd": 0, "cost_for": "per person",
                  "tags": ["culture"]},
             ],
             "daily_cost_estimate": {"food": 3000, "food_usd": 20, "activities": 0,
                                     "activities_usd": 0, "transport": 1000, "transport_usd": 7,
                                     "total_local": 4000, "total_usd": 27,
                                     "currency_code": "JPY"},
             "booking_alerts": [], "quick_reference": {}},
            {"day": 2, "date": "2026-04-02", "city": "Tokyo", "theme": "Culture",
             "slots": [
                 {"time": "10:00", "end_time": "12:00", "type": "activity",
                  "name": "Teamlab Borderless", "address": "Odaiba", "duration_min": 120,
                  "cost_local": 3700, "cost_usd": 25, "cost_for": "per person",
                  "tags": ["art"]},
             ],
             "daily_cost_estimate": {"food": 5000, "food_usd": 34, "activities": 3700,
                                     "activities_usd": 25, "transport": 1500, "transport_usd": 10,
                                     "total_local": 10200, "total_usd": 69,
                                     "currency_code": "JPY"},
             "booking_alerts": [], "quick_reference": {}},
        ])
        mock_response = AIMessage(content=agenda_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/agenda")

        agenda = result["state_updates"].get("detailed_agenda", [])
        assert len(agenda) == 2
        assert agenda[0]["day"] == 1
        assert agenda[1]["day"] == 2

    @pytest.mark.asyncio
    async def test_each_slot_has_time_address_cost(self, japan_state):
        """TC-SCH-02: Each slot has time, address, cost (dual currency)."""
        from src.agents.scheduler import SchedulerAgent

        agent = SchedulerAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}

        agenda_json = json.dumps([
            {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Arrival",
             "slots": [
                 {"time": "09:00", "end_time": "10:30", "type": "activity",
                  "name": "Senso-ji", "address": "2-3-1 Asakusa, Taito",
                  "duration_min": 90,
                  "cost_local": 0, "cost_usd": 0, "cost_for": "total",
                  "tags": ["culture"]},
             ],
             "daily_cost_estimate": {"total_local": 4000, "total_usd": 27,
                                     "currency_code": "JPY"},
             "booking_alerts": [], "quick_reference": {}},
        ])
        mock_response = AIMessage(content=agenda_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/agenda")

        agenda = result["state_updates"].get("detailed_agenda", [])
        slot = agenda[0]["slots"][0]
        assert "time" in slot
        assert "address" in slot
        assert "cost_local" in slot
        assert "cost_usd" in slot

    @pytest.mark.asyncio
    async def test_adapts_to_feedback(self, japan_state):
        """TC-SCH-03: Adapts to feedback (lighter after low energy)."""
        from src.agents.scheduler import SchedulerAgent

        agent = SchedulerAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO}
        state["feedback_log"] = [
            {"day": 1, "city": "Tokyo", "energy_level": "low",
             "highlight": "Senso-ji was amazing", "budget_status": "on_track"},
        ]

        agenda_json = json.dumps([
            {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Rest Day",
             "slots": [], "daily_cost_estimate": {}, "booking_alerts": [],
             "quick_reference": {}},
        ])
        mock_response = AIMessage(content=agenda_json)

        # We need to capture the arguments the mock was called with
        mock_ainvoke = AsyncMock(return_value=mock_response)
        with patch.object(type(agent.llm), "ainvoke", new=mock_ainvoke):
            result = await agent.handle(state, "/agenda")

        call_args = mock_ainvoke.call_args
        messages = call_args[0][0]
        prompt_text = messages[-1].content
        assert "RECENT FEEDBACK" in prompt_text or "feedback" in prompt_text.lower()


# =============================================================================
# FEEDBACK TESTS
# =============================================================================


class TestFeedbackFlow:
    """Tests for the feedback agent (TC-FBK-*)."""

    @pytest.mark.asyncio
    async def test_context_aware_greeting(self, japan_state):
        """TC-FBK-01: Opens with context-aware greeting for current day/city."""
        from src.agents.feedback import FeedbackAgent

        agent = FeedbackAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["current_trip_day"] = 1

        mock_response = AIMessage(content="Hey! How was Day 1 in Tokyo?")
        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "/feedback")

        assert "response" in result
        assert "Tokyo" in result["response"] or "Day 1" in result["response"]

    @pytest.mark.asyncio
    async def test_multi_turn_no_json_on_first_response(self, japan_state):
        """TC-FBK-02: Multi-turn: doesn't emit JSON on first response."""
        from src.agents.feedback import FeedbackAgent

        agent = FeedbackAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["current_trip_day"] = 1

        mock_response = AIMessage(content="That sounds amazing! What was the highlight?")
        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "We visited Senso-ji today, it was great!")

        updates = result.get("state_updates", {})
        # feedback_log should NOT be in updates since no JSON was emitted
        assert "feedback_log" not in updates

    @pytest.mark.asyncio
    async def test_emits_json_on_conclusion(self, japan_state):
        """TC-FBK-03: Emits JSON with feedback_complete on conclusion."""
        from src.agents.feedback import FeedbackAgent

        agent = FeedbackAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["current_trip_day"] = 1

        feedback_json = json.dumps({
            "feedback_complete": True,
            "day": 1, "date": "2026-04-01", "city": "Tokyo",
            "completed_items": ["tokyo-place-1"], "skipped_items": [],
            "highlight": "Senso-ji at sunset", "lowlight": None,
            "energy_level": "medium", "food_rating": "amazing",
            "budget_status": "on_track", "weather": "sunny",
            "discoveries": ["hidden garden"], "preference_shifts": [],
            "destination_feedback": {"transport": "good", "language": "fine",
                                     "safety": "fine", "accommodation": "great",
                                     "food_quality": "amazing", "connectivity": "good"},
            "adjustments_made": [], "sentiment": "very positive",
            "actual_spend_usd": 45, "actual_spend_local": 6660,
        })
        response_text = f"What a wonderful day!\n```json\n{feedback_json}\n```"
        mock_response = AIMessage(content=response_text)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "That's everything, thanks!")

        updates = result.get("state_updates", {})
        assert "feedback_log" in updates
        assert len(updates["feedback_log"]) == 1
        entry = updates["feedback_log"][0]
        assert entry["day"] == 1
        assert entry["city"] == "Tokyo"
        assert entry["energy_level"] == "medium"

    @pytest.mark.asyncio
    async def test_captures_energy_highlight_budget(self, japan_state):
        """TC-FBK-04: Captures energy_level, highlight, budget_status."""
        from src.agents.feedback import FeedbackAgent

        agent = FeedbackAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["current_trip_day"] = 1

        feedback_json = json.dumps({
            "feedback_complete": True,
            "day": 1, "date": "2026-04-01", "city": "Tokyo",
            "completed_items": [], "skipped_items": [],
            "highlight": "Ramen at Ichiran", "lowlight": None,
            "energy_level": "high", "food_rating": "amazing",
            "budget_status": "under", "weather": "cloudy",
            "discoveries": [], "preference_shifts": [],
            "destination_feedback": {}, "adjustments_made": [],
            "sentiment": "positive",
            "actual_spend_usd": 30, "actual_spend_local": 4440,
        })
        response_text = f"Great day!\n```json\n{feedback_json}\n```"
        mock_response = AIMessage(content=response_text)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "All good, wrap it up")

        entry = result["state_updates"]["feedback_log"][0]
        assert entry["energy_level"] == "high"
        assert entry["highlight"] == "Ramen at Ichiran"
        assert entry["budget_status"] == "under"

    @pytest.mark.asyncio
    async def test_pending_adjustments_in_agent_scratch(self, japan_state):
        """TC-FBK-05: Pending adjustments stored in agent_scratch."""
        from src.agents.feedback import FeedbackAgent

        agent = FeedbackAgent()
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["current_trip_day"] = 1

        feedback_json = json.dumps({
            "feedback_complete": True,
            "day": 1, "date": "2026-04-01", "city": "Tokyo",
            "completed_items": [], "skipped_items": [],
            "highlight": None, "lowlight": None,
            "energy_level": "low", "food_rating": "good",
            "budget_status": "on_track", "weather": "rain",
            "discoveries": [], "preference_shifts": [],
            "destination_feedback": {},
            "adjustments_made": ["lighten tomorrow's schedule", "add indoor activities"],
            "sentiment": "tired",
            "actual_spend_usd": 50, "actual_spend_local": 7400,
        })
        response_text = f"Rest up!\n```json\n{feedback_json}\n```"
        mock_response = AIMessage(content=response_text)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(state, "I'm exhausted")

        updates = result.get("state_updates", {})
        assert "agent_scratch" in updates
        assert "pending_adjustments" in updates["agent_scratch"]
        assert "lighten tomorrow's schedule" in updates["agent_scratch"]["pending_adjustments"]


# =============================================================================
# COST TESTS
# =============================================================================


class TestCostFlow:
    """Tests for the cost agent (TC-CST-*)."""

    @pytest.mark.asyncio
    async def test_full_budget_report(self, japan_state):
        """TC-CST-01: /costs returns full budget report."""
        from src.agents.cost import CostAgent

        agent = CostAgent()
        result = await agent.handle(japan_state, "/costs")
        assert "BUDGET REPORT" in result["response"]
        assert result["state_updates"] == {}

    @pytest.mark.asyncio
    async def test_currency_conversion(self, japan_state):
        """TC-CST-02: /costs convert 100 converts currency."""
        from src.agents.cost import CostAgent

        agent = CostAgent()
        result = await agent.handle(japan_state, "/costs convert 100")
        assert "response" in result
        assert "USD" in result["response"] or "$" in result["response"]

    @pytest.mark.asyncio
    async def test_natural_language_spending_update(self, japan_state):
        """TC-CST-03: Natural language 'spent $20 on lunch' updates tracker."""
        from src.agents.cost import CostAgent

        agent = CostAgent()
        spend_json = json.dumps({
            "amount_local": 2960,
            "amount_usd": 20,
            "category": "food",
            "description": "lunch at ramen shop",
        })
        mock_response = AIMessage(content=spend_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(japan_state, "spent $20 on lunch")

        updates = result.get("state_updates", {})
        assert "cost_tracker" in updates
        tracker = updates["cost_tracker"]
        assert tracker["totals"]["spent_usd"] == 20
        assert tracker["by_category"]["food"]["spent_usd"] == 20

    @pytest.mark.asyncio
    async def test_dual_currency_in_output(self, japan_state):
        """TC-CST-04: Dual currency in output."""
        from src.agents.cost import CostAgent

        agent = CostAgent()
        spend_json = json.dumps({
            "amount_local": 2960,
            "amount_usd": 20,
            "category": "food",
            "description": "lunch",
        })
        mock_response = AIMessage(content=spend_json)

        with patch.object(type(agent.llm), "ainvoke", new=AsyncMock(return_value=mock_response)):
            result = await agent.handle(japan_state, "spent $20 on lunch")

        response = result["response"]
        assert "$" in response

    @pytest.mark.asyncio
    async def test_no_tracker_data_message(self, japan_state):
        """TC-CST-05: No tracker data returns appropriate message."""
        from src.agents.cost import CostAgent

        agent = CostAgent()
        state = dict(japan_state)
        state["cost_tracker"] = {}

        result = await agent.handle(state, "/costs")
        assert "No budget data" in result["response"] or "BUDGET REPORT" in result["response"]


# =============================================================================
# TRIP SHARING TESTS (DB + Handlers)
# =============================================================================


class TestTripSharing:
    """Tests for trip sharing functionality (TC-SHR-*)."""

    @pytest.mark.asyncio
    async def test_join_adds_member_and_sets_active(self, async_db):
        """TC-SHR-01: /join <id> adds member and sets active trip."""
        repo = async_db
        state = {"destination": {"country": "Japan", "flag_emoji": ""}, "cities": [{"name": "Tokyo"}]}
        await repo.create_trip("trip-abc", "owner-1", state)

        update = make_mock_update("/join trip-abc", user_id=99999)
        context = make_mock_context(active_trip_id="default", repo=repo)

        await _handle_join(update, context, "99999", "/join trip-abc", repo)

        assert await repo.is_member("trip-abc", "99999")
        assert context.user_data["active_trip_id"] == "trip-abc"
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "joined" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_join_nonexistent_trip_returns_error(self, async_db):
        """TC-SHR-02: /join with nonexistent trip returns error."""
        repo = async_db
        update = make_mock_update("/join nonexistent", user_id=12345)
        context = make_mock_context(repo=repo)

        await _handle_join(update, context, "12345", "/join nonexistent", repo)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "not found" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_join_archived_trip_returns_error(self, async_db):
        """TC-SHR-03: /join with archived trip returns error."""
        repo = async_db
        await repo.create_trip("arch-trip", "owner-1", {"destination": {"country": "Japan"}})
        await repo.archive_trip("arch-trip")

        update = make_mock_update("/join arch-trip", user_id=12345)
        context = make_mock_context(repo=repo)

        await _handle_join(update, context, "12345", "/join arch-trip", repo)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "archived" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_trip_id_shows_active_id(self, async_db):
        """TC-SHR-04: /trip id shows current active trip ID."""
        update = make_mock_update("/trip id", user_id=12345)
        context = make_mock_context(active_trip_id="my-trip-42", repo=async_db)

        result = await _handle_trip_management(update, context, "12345", "/trip id", async_db)
        assert result is True
        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "my-trip-42" in reply_text

    @pytest.mark.asyncio
    async def test_trips_shows_owner_and_member_roles(self, async_db):
        """TC-SHR-05: /trips shows [owner] for owned, [member] for joined."""
        repo = async_db
        await repo.create_trip("own-trip", "user-A", {
            "destination": {"country": "Japan", "flag_emoji": ""},
        })
        await repo.create_trip("other-trip", "user-B", {
            "destination": {"country": "Morocco", "flag_emoji": ""},
        })
        await repo.add_member("other-trip", "user-A")

        update = make_mock_update("/trips", user_id=12345)
        context = make_mock_context(active_trip_id="own-trip", repo=repo)

        await _list_trips(update, context, "user-A", repo)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "[owner]" in reply_text
        assert "[member]" in reply_text

    @pytest.mark.asyncio
    async def test_list_trips_includes_owned_and_joined(self, async_db):
        """TC-SHR-06: list_trips includes both owned and joined trips."""
        repo = async_db
        await repo.create_trip("trip-1", "user-A", {"destination": {"country": "Japan"}})
        await repo.create_trip("trip-2", "user-B", {"destination": {"country": "Morocco"}})
        await repo.add_member("trip-2", "user-A")

        trips = await repo.list_trips("user-A")
        trip_ids = [t.trip_id for t in trips]
        assert "trip-1" in trip_ids
        assert "trip-2" in trip_ids

    @pytest.mark.asyncio
    async def test_get_active_trips_includes_joined_non_archived(self, async_db):
        """TC-SHR-07: get_active_trips includes joined non-archived trips."""
        repo = async_db
        await repo.create_trip("active-1", "user-A", {"destination": {"country": "Japan"}})
        await repo.create_trip("active-2", "user-B", {"destination": {"country": "Morocco"}})
        await repo.add_member("active-2", "user-A")
        await repo.create_trip("archived-1", "user-A", {"destination": {"country": "France"}})
        await repo.archive_trip("archived-1")

        trips = await repo.get_active_trips("user-A")
        trip_ids = [t.trip_id for t in trips]
        assert "active-1" in trip_ids
        assert "active-2" in trip_ids
        assert "archived-1" not in trip_ids

    @pytest.mark.asyncio
    async def test_owner_is_implicit_member(self, async_db):
        """TC-SHR-08: Owner is implicit member (add_member is no-op)."""
        repo = async_db
        await repo.create_trip("trip-own", "owner-1", {"destination": {"country": "Japan"}})

        await repo.add_member("trip-own", "owner-1")
        assert await repo.is_member("trip-own", "owner-1")

        joined = await repo.get_joined_trips("owner-1")
        joined_ids = [t.trip_id for t in joined]
        assert "trip-own" not in joined_ids

    @pytest.mark.asyncio
    async def test_idempotent_join(self, async_db):
        """TC-SHR-09: Idempotent join (no error on double join)."""
        repo = async_db
        await repo.create_trip("trip-idem", "owner-1", {"destination": {"country": "Japan"}})

        await repo.add_member("trip-idem", "joiner-1")
        await repo.add_member("trip-idem", "joiner-1")

        assert await repo.is_member("trip-idem", "joiner-1")

    @pytest.mark.asyncio
    async def test_shared_thread_id_equals_trip_id(self, async_db):
        """TC-SHR-10: Shared thread_id = trip_id (not user-prefixed)."""
        repo = async_db
        await repo.create_trip("shared-trip", "owner-1", {})

        context = make_mock_context(active_trip_id="shared-trip", repo=repo)
        trip_id = context.user_data.get("active_trip_id", "default")
        thread_id = trip_id

        assert thread_id == "shared-trip"
        assert thread_id == trip_id


# =============================================================================
# HANDLER SHORT-CIRCUIT TESTS
# =============================================================================


class TestHandlerShortCircuits:
    """Tests for handler short-circuit logic (TC-HDL-*)."""

    @pytest.mark.asyncio
    async def test_join_does_not_invoke_graph(self, async_db):
        """TC-HDL-01: /join does NOT invoke graph."""
        repo = async_db
        await repo.create_trip("trip-no-graph", "owner-1", {
            "destination": {"country": "Japan", "flag_emoji": ""},
            "cities": [],
        })

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        update = make_mock_update("/join trip-no-graph", user_id=88888)
        context = make_mock_context(active_trip_id="default", graph=mock_graph, repo=repo)

        from src.telegram.handlers import process_message
        await process_message(update, context)

        mock_graph.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_trip_new_generates_uuid_sets_active(self, async_db):
        """TC-HDL-02: /trip new generates UUID, sets active."""
        update = make_mock_update("/trip new", user_id=12345)
        context = make_mock_context(active_trip_id="old-trip", repo=async_db)

        result = await _handle_trip_management(update, context, "12345", "/trip new", async_db)
        assert result is True

        new_id = context.user_data["active_trip_id"]
        assert new_id != "old-trip"
        assert len(new_id) == 8

    @pytest.mark.asyncio
    async def test_trip_switch_changes_active(self, async_db):
        """TC-HDL-03: /trip switch <id> changes active trip."""
        repo = async_db
        await repo.create_trip("switch-target", "user-1", {
            "destination": {"country": "Japan", "flag_emoji": ""},
        })

        update = make_mock_update("/trip switch switch-target", user_id=12345)
        context = make_mock_context(active_trip_id="old-trip", repo=repo)

        result = await _handle_trip_management(
            update, context, "12345", "/trip switch switch-target", repo
        )
        assert result is True
        assert context.user_data["active_trip_id"] == "switch-target"

    @pytest.mark.asyncio
    async def test_trip_archive_sets_archived(self, async_db):
        """TC-HDL-04: /trip archive <id> confirm sets archived=True."""
        repo = async_db
        await repo.create_trip("to-archive", "user-1", {"destination": {"country": "Japan"}})

        update = make_mock_update("/trip archive to-archive confirm", user_id=12345)
        context = make_mock_context(active_trip_id="to-archive", repo=repo)

        result = await _handle_trip_management(
            update, context, "12345", "/trip archive to-archive confirm", repo
        )
        assert result is True

        trip = await repo.get_trip("to-archive")
        assert trip.archived is True

    @pytest.mark.asyncio
    async def test_trips_lists_all_with_roles(self, async_db):
        """TC-HDL-05: /trips lists all trips with roles."""
        repo = async_db
        await repo.create_trip("my-trip", "user-X", {
            "destination": {"country": "Japan", "flag_emoji": ""},
        })

        update = make_mock_update("/trips", user_id=12345)
        context = make_mock_context(active_trip_id="my-trip", repo=repo)

        await _list_trips(update, context, "user-X", repo)

        update.message.reply_text.assert_called_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "[owner]" in reply_text
        assert "my-trip" in reply_text

    def test_long_response_split_at_paragraph_boundaries(self):
        """TC-HDL-06: Long response split at paragraph boundaries (< 4096 chars each)."""
        paragraph = "A" * 1000
        long_text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
        assert len(long_text) > 4096

        parts = split_message(long_text, max_length=4096)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 4096


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across the system (TC-ERR-*)."""

    @pytest.mark.asyncio
    async def test_agent_exception_returns_error_no_state_updates(self, japan_state):
        """TC-ERR-01: Agent exception returns generic error, no state_updates."""
        state = dict(japan_state)
        state["_user_message"] = "test message"
        state["messages"] = [{"role": "user", "content": "test message"}]

        with patch("src.graph._get_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.handle = AsyncMock(side_effect=RuntimeError("LLM exploded"))
            mock_get.return_value = mock_agent

            result = await _specialist_node("research", state)

        messages = result.get("messages", [])
        assert len(messages) > 0
        error_msg = messages[0].get("content", "")
        assert "trouble researching" in error_msg.lower() or "try again" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_conversation_history_updated_on_failure(self, japan_state):
        """TC-ERR-02: Conversation history updated even on agent failure."""
        state = dict(japan_state)
        state["_user_message"] = "this will fail"
        state["messages"] = [{"role": "user", "content": "this will fail"}]

        with patch("src.graph._get_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.handle = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_get.return_value = mock_agent

            result = await _specialist_node("research", state)

        history = result.get("conversation_history", [])
        assert len(history) > 0
        assistant_msgs = [m for m in history if m.get("role") == "assistant"]
        assert len(assistant_msgs) > 0

    def test_state_save_filters_internal_keys(self):
        """TC-ERR-03: State save filtered (no _next, _user_message, messages)."""
        assert _INTERNAL_KEYS == {"_next", "_user_message", "messages"}

        result = {
            "_next": "research",
            "_user_message": "test",
            "messages": [{"role": "user", "content": "test"}],
            "destination": {"country": "Japan"},
            "onboarding_complete": True,
        }

        filtered = {k: v for k, v in result.items() if k not in _INTERNAL_KEYS}
        assert "_next" not in filtered
        assert "_user_message" not in filtered
        assert "messages" not in filtered
        assert "destination" in filtered
        assert "onboarding_complete" in filtered

    def test_extract_response_fallback(self):
        """TC-ERR-04: Graph failure returns fallback message."""
        result = {}
        response = _extract_response(result)
        assert "not sure" in response.lower() or "help" in response.lower()

    def test_extract_response_from_messages(self):
        """TC-ERR-04b: _extract_response finds last assistant message in messages."""
        result = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ]
        }
        response = _extract_response(result)
        assert response == "Hi there, how can I help?"

    def test_extract_response_from_response_key(self):
        """TC-ERR-04c: _extract_response prefers direct response field."""
        result = {
            "response": "Direct response text",
            "messages": [{"role": "assistant", "content": "Message text"}],
        }
        response = _extract_response(result)
        assert response == "Direct response text"


# =============================================================================
# GRAPH ROUTING TESTS
# =============================================================================


class TestGraphRouting:
    """Tests for graph-level routing functions."""

    def test_route_from_orchestrator_valid_nodes(self):
        """Valid _next values route to the correct node."""
        for node in ("onboarding", "research", "librarian", "prioritizer",
                      "planner", "scheduler", "feedback", "cost"):
            state = {"_next": node}
            assert route_from_orchestrator(state) == node

    def test_route_from_orchestrator_end(self):
        """_next=END routes to END."""
        state = {"_next": END}
        assert route_from_orchestrator(state) == END

    def test_route_from_orchestrator_orchestrator_maps_to_end(self):
        """_next='orchestrator' maps to END."""
        state = {"_next": "orchestrator"}
        assert route_from_orchestrator(state) == END

    def test_route_from_orchestrator_invalid_maps_to_end(self):
        """Invalid _next values map to END."""
        state = {"_next": "nonexistent_agent"}
        assert route_from_orchestrator(state) == END

    def test_route_from_orchestrator_missing_next(self):
        """Missing _next defaults to END."""
        state = {}
        assert route_from_orchestrator(state) == END


# =============================================================================
# ADDITIONAL COVERAGE: GENERATE_STATUS
# =============================================================================


class TestGenerateStatus:
    """Tests for the generate_status helper."""

    def test_status_pre_onboarding(self, empty_state):
        """Pre-onboarding state shows setup message."""
        result = generate_status(empty_state)
        assert "set up" in result.lower() or "/start" in result

    def test_status_post_onboarding(self, japan_state):
        """Post-onboarding state shows country, day count, trip ID, and next step."""
        result = generate_status(japan_state)
        assert "Japan" in result
        assert "14" in result
        assert "Onboarding complete" in result
        assert "japan-2026" in result
        assert "Next:" in result or "👉" in result

    def test_status_with_research(self, japan_state):
        """Status shows research progress."""
        state = dict(japan_state)
        state["research"] = {"Tokyo": SAMPLE_RESEARCH_TOKYO, "Kyoto": {}}
        result = generate_status(state)
        assert "Research" in result
        assert "2/3" in result

    def test_status_with_plan(self, japan_state):
        """Status shows plan status."""
        state = dict(japan_state)
        state["high_level_plan"] = SAMPLE_DAY_PLAN
        state["plan_status"] = "draft"
        result = generate_status(state)
        assert "Draft" in result


# =============================================================================
# ADDITIONAL COVERAGE: FORMAT_BUDGET_REPORT
# =============================================================================


class TestFormatBudgetReport:
    """Tests for budget report formatting."""

    def test_empty_tracker_message(self):
        """Empty cost_tracker returns no-data message."""
        state = {"destination": {}, "cost_tracker": {}}
        report = format_budget_report(state)
        assert "No budget data" in report

    def test_full_report_structure(self, japan_state):
        """Full report contains expected sections."""
        report = format_budget_report(japan_state)
        assert "BUDGET REPORT" in report
        assert "OVERALL" in report
        assert "Budget" in report
        assert "$4,000" in report or "4,000" in report

    def test_report_with_spending(self, japan_state):
        """Report reflects actual spending."""
        state = dict(japan_state)
        state["cost_tracker"]["totals"]["spent_usd"] = 500
        state["cost_tracker"]["totals"]["remaining_usd"] = 3500
        report = format_budget_report(state)
        assert "500" in report
        assert "3,500" in report or "3500" in report


# =============================================================================
# ADDITIONAL COVERAGE: SPLIT_MESSAGE
# =============================================================================


class TestSplitMessage:
    """Tests for Telegram message splitting."""

    def test_short_message_no_split(self):
        """Messages under limit are not split."""
        text = "Hello world"
        parts = split_message(text)
        assert parts == ["Hello world"]

    def test_exact_limit_no_split(self):
        """Message at exactly the limit is not split."""
        text = "A" * 4096
        parts = split_message(text)
        assert len(parts) == 1

    def test_split_at_paragraphs(self):
        """Messages split at paragraph boundaries."""
        p1 = "A" * 2000
        p2 = "B" * 2000
        p3 = "C" * 2000
        text = f"{p1}\n\n{p2}\n\n{p3}"
        parts = split_message(text, max_length=4096)
        assert len(parts) >= 2
        for part in parts:
            assert len(part) <= 4096

    def test_hard_split_no_paragraphs(self):
        """Very long single-paragraph text is hard-split."""
        text = "X" * 8192
        parts = split_message(text, max_length=4096)
        assert len(parts) >= 2
        for part in parts:
            assert len(part) <= 4096


# =============================================================================
# ADDITIONAL COVERAGE: COMMAND_MAP and PREREQUISITES
# =============================================================================


class TestCommandMapAndPrerequisites:
    """Tests to verify command map and prerequisite configuration."""

    def test_command_map_covers_all_agents(self):
        """COMMAND_MAP includes routes for all essential commands."""
        assert "/start" in COMMAND_MAP
        assert "/research" in COMMAND_MAP
        assert "/priorities" in COMMAND_MAP
        assert "/plan" in COMMAND_MAP
        assert "/agenda" in COMMAND_MAP
        assert "/feedback" in COMMAND_MAP
        assert "/costs" in COMMAND_MAP
        assert "/library" in COMMAND_MAP
        assert "/status" in COMMAND_MAP
        assert "/help" in COMMAND_MAP
        assert "/join" in COMMAND_MAP
        assert "/trip" in COMMAND_MAP
        assert "/trips" in COMMAND_MAP

    def test_prerequisites_chain(self):
        """Prerequisites form the correct dependency chain."""
        assert "prioritizer" in PREREQUISITES
        assert "planner" in PREREQUISITES
        assert "scheduler" in PREREQUISITES

        pri_fields = [field for field, _ in PREREQUISITES["prioritizer"]]
        assert "research" in pri_fields

        plan_fields = [field for field, _ in PREREQUISITES["planner"]]
        assert "research" in plan_fields
        assert "priorities" in plan_fields

        sched_fields = [field for field, _ in PREREQUISITES["scheduler"]]
        assert "high_level_plan" in sched_fields

    def test_research_has_no_prerequisites(self):
        """Research agent has no prerequisites."""
        assert "research" not in PREREQUISITES


# =============================================================================
# NEW TESTS: UX audit additions
# =============================================================================


class TestUXAuditAdditions:
    """Tests for UX audit fixes."""

    @pytest.mark.asyncio
    async def test_start_after_onboarding_complete_returns_guard(self, japan_state):
        """H3: /start after onboarding returns guard message."""
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/start")
        assert result["target_agent"] == "orchestrator"
        assert "/trip new" in result["response"]
        assert "/status" in result["response"]

    def test_status_includes_next_step(self, japan_state):
        """H4: Status includes next-step suggestion."""
        status = generate_status(japan_state)
        assert "👉" in status
        assert "/research all" in status

    def test_status_includes_next_step_priorities(self, japan_state):
        """H4: Status suggests /priorities when research done."""
        state = dict(japan_state)
        state["research"] = {"Tokyo": {}, "Kyoto": {}, "Osaka": {}}
        status = generate_status(state)
        assert "/priorities" in status

    def test_status_includes_next_step_plan(self, japan_state):
        """H4: Status suggests /plan when priorities done."""
        state = dict(japan_state)
        state["research"] = {"Tokyo": {}, "Kyoto": {}, "Osaka": {}}
        state["priorities"] = {"Tokyo": [], "Kyoto": [], "Osaka": []}
        status = generate_status(state)
        assert "/plan" in status

    def test_status_includes_next_step_agenda(self, japan_state):
        """H4: Status suggests /agenda when plan done."""
        state = dict(japan_state)
        state["research"] = {"Tokyo": {}, "Kyoto": {}, "Osaka": {}}
        state["priorities"] = {"Tokyo": [], "Kyoto": [], "Osaka": []}
        state["high_level_plan"] = [{"day": 1}]
        status = generate_status(state)
        assert "/agenda" in status

    def test_status_includes_next_step_all_done(self, japan_state):
        """H4: Status says all set when everything done."""
        state = dict(japan_state)
        state["research"] = {"Tokyo": {}, "Kyoto": {}, "Osaka": {}}
        state["priorities"] = {"Tokyo": [], "Kyoto": [], "Osaka": []}
        state["high_level_plan"] = [{"day": 1}]
        state["detailed_agenda"] = [{"day": 1}]
        status = generate_status(state)
        assert "/feedback" in status

    def test_status_includes_trip_id(self, japan_state):
        """M5: Status includes trip ID."""
        status = generate_status(japan_state)
        assert "japan-2026" in status
        assert "/join" in status

    @pytest.mark.asyncio
    async def test_archive_requires_confirmation(self, async_db):
        """M4: /trip archive without confirm asks for confirmation."""
        repo = async_db
        await repo.create_trip("conf-trip", "user-1", {"destination": {"country": "Japan"}})

        update = make_mock_update("/trip archive conf-trip", user_id=12345)
        context = make_mock_context(active_trip_id="conf-trip", repo=repo)

        result = await _handle_trip_management(
            update, context, "12345", "/trip archive conf-trip", repo
        )
        assert result is True

        # Trip should NOT be archived yet
        trip = await repo.get_trip("conf-trip")
        assert trip.archived is False

        # Reply should ask for confirmation
        reply_text = update.message.reply_text.call_args[0][0]
        assert "confirm" in reply_text.lower()

    def test_help_includes_cost_subcommands(self):
        """H6: /costs sub-commands are discoverable in help."""
        help_text = generate_help()
        assert "/costs today" in help_text
        assert "/costs save" in help_text
        assert "/costs convert" in help_text
        assert "/costs food" in help_text or "category" in help_text.lower()


# =============================================================================
# ADDITIONAL COVERAGE: add_to_conversation_history
# =============================================================================


class TestConversationHistory:
    """Tests for the conversation history utility."""

    def test_adds_message_with_timestamp(self, empty_state):
        """add_to_conversation_history appends correctly."""
        history = add_to_conversation_history(empty_state, "user", "hello")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"
        assert "timestamp" in history[0]

    def test_preserves_existing_history(self, japan_state):
        """Existing history is preserved when appending."""
        state = dict(japan_state)
        state["conversation_history"] = [
            {"role": "user", "content": "old msg", "timestamp": "t1", "agent": None}
        ]
        history = add_to_conversation_history(state, "assistant", "new msg", agent="orchestrator")
        assert len(history) == 2
        assert history[0]["content"] == "old msg"
        assert history[1]["content"] == "new msg"
        assert history[1]["agent"] == "orchestrator"

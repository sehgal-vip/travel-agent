"""Tests for v2 loopback routing: awaiting_input, delegation, chaining, depth enforcement, error_handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator import OrchestratorAgent
from src.state import TripState


@pytest.fixture
def loopback_state() -> dict:
    """State with onboarding complete and all v2 graph control keys."""
    return {
        "trip_id": "test-loop",
        "onboarding_complete": True,
        "onboarding_step": 11,
        "destination": {"country": "Japan", "country_code": "JP", "flag_emoji": "\U0001f1ef\U0001f1f5"},
        "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
        "dates": {"start": "2026-04-01", "end": "2026-04-05", "total_days": 4},
        "travelers": {"count": 2, "type": "couple"},
        "budget": {"style": "midrange"},
        "interests": ["food"],
        "must_dos": [],
        "deal_breakers": [],
        "accommodation_pref": "hotel",
        "transport_pref": ["train"],
        "research": {"Tokyo": {"places": [], "food": []}},
        "priorities": {},
        "high_level_plan": [],
        "plan_status": "not_started",
        "plan_version": 0,
        "detailed_agenda": [],
        "feedback_log": [],
        "cost_tracker": {},
        "library": {},
        "current_agent": "orchestrator",
        "conversation_history": [],
        "current_trip_day": 1,
        "agent_scratch": {},
        "messages": [],
        "_next": "",
        "_user_message": "",
        "_awaiting_input": None,
        "_callback": None,
        "_delegate_to": None,
        "_chain": [],
        "_routing_echo": "",
        "_error_agent": None,
        "_error_context": None,
        "_loopback_depth": 0,
    }


class TestAwaitingInput:
    """Tests for _awaiting_input routing."""

    @pytest.mark.asyncio
    async def test_routes_to_waiting_agent(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_awaiting_input"] = "research"
        result = await orch.route(loopback_state, "yes, more details about Tokyo food")
        assert result["target_agent"] == "research"

    @pytest.mark.asyncio
    async def test_overrides_command_routing(self, loopback_state):
        """Awaiting input takes priority over command-based routing."""
        orch = OrchestratorAgent()
        loopback_state["_awaiting_input"] = "planner"
        result = await orch.route(loopback_state, "/costs")
        assert result["target_agent"] == "planner"

    @pytest.mark.asyncio
    async def test_invalid_agent_falls_through(self, loopback_state):
        """Invalid _awaiting_input agent falls through to normal routing."""
        orch = OrchestratorAgent()
        loopback_state["_awaiting_input"] = "nonexistent"
        result = await orch.route(loopback_state, "/research")
        # Should fall through to normal command routing
        assert result["target_agent"] == "research"


class TestDelegation:
    """Tests for _delegate_to routing."""

    @pytest.mark.asyncio
    async def test_routes_to_delegate(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_delegate_to"] = "cost"
        result = await orch.route(loopback_state, "checking budget")
        assert result["target_agent"] == "cost"

    @pytest.mark.asyncio
    async def test_callback_returns_to_caller(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_callback"] = "planner"
        result = await orch.route(loopback_state, "budget info ready")
        assert result["target_agent"] == "planner"


class TestChaining:
    """Tests for _chain sequential routing."""

    @pytest.mark.asyncio
    async def test_routes_to_first_in_chain(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_chain"] = ["research", "planner"]
        result = await orch.route(loopback_state, "continue")
        assert result["target_agent"] == "research"


class TestLoopbackDepth:
    """Tests for loopback depth enforcement."""

    @pytest.mark.asyncio
    async def test_enforces_depth_limit(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_loopback_depth"] = 6
        result = await orch.route(loopback_state, "anything")
        assert result["target_agent"] == "orchestrator"
        assert result.get("response") is not None

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, loopback_state):
        orch = OrchestratorAgent()
        loopback_state["_loopback_depth"] = 3
        loopback_state["_awaiting_input"] = "research"
        result = await orch.route(loopback_state, "yes")
        assert result["target_agent"] == "research"


class TestErrorHandler:
    """Tests for error_handler node."""

    @pytest.mark.asyncio
    async def test_error_handler_returns_message(self):
        from src.graph import error_handler_node
        state = {
            "_error_agent": "research",
            "_error_context": "LLM timeout",
            "messages": [],
        }
        result = await error_handler_node(state)
        msg = result["messages"][0]["content"]
        assert "research" in msg.lower()
        assert result["_error_agent"] is None
        assert result["_error_context"] is None

    @pytest.mark.asyncio
    async def test_error_handler_clears_error_state(self):
        from src.graph import error_handler_node
        state = {
            "_error_agent": "planner",
            "_error_context": "Parse error",
            "messages": [],
        }
        result = await error_handler_node(state)
        assert result["_error_agent"] is None
        assert result["_error_context"] is None

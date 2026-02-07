"""Integration tests — compile real graph, mock LLM agents, verify full routing flow."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END

from src.graph import build_graph, compile_graph, _specialist_node
from src.state import TripState
from src.telegram.handlers import _extract_response


# ─── Helpers ────────────────────────────────────────


def _make_state(messages: list[dict] | None = None, **overrides) -> dict:
    """Minimal input state for graph invocation."""
    state = {
        "messages": messages or [],
        "conversation_history": [],
        "onboarding_complete": False,
        "onboarding_step": 0,
        "current_agent": "orchestrator",
        "agent_scratch": {},
        "destination": {},
        "cities": [],
        "travelers": {},
        "budget": {},
        "dates": {},
        "interests": [],
        "must_dos": [],
        "deal_breakers": [],
        "accommodation_pref": "",
        "transport_pref": [],
        "research": {},
        "priorities": {},
        "high_level_plan": [],
        "plan_status": "not_started",
        "plan_version": 0,
        "detailed_agenda": [],
        "feedback_log": [],
        "cost_tracker": {},
        "library": {},
        "current_trip_day": None,
        "trip_id": "test",
        "created_at": "",
        "updated_at": "",
        "_next": "",
        "_user_message": "",
    }
    state.update(overrides)
    return state


def _onboarded_state(**overrides) -> dict:
    """State that has completed onboarding (for specialist routing tests)."""
    return _make_state(
        onboarding_complete=True,
        onboarding_step=11,
        current_trip_day=1,
        destination={"country": "Japan", "country_code": "JP", "flag_emoji": "🇯🇵"},
        cities=[{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
        dates={"start": "2026-04-01", "end": "2026-04-05", "total_days": 4},
        travelers={"count": 2, "type": "couple"},
        budget={"style": "midrange"},
        research={"Tokyo": {"places": [], "food": []}},
        priorities={"Tokyo": [{"item_id": "t1", "name": "Tsukiji", "tier": "must_do", "score": 5}]},
        high_level_plan=[{"day": 1, "city": "Tokyo", "theme": "Arrival"}],
        **overrides,
    )


def _mock_orchestrator_route(target: str, response: str | None = None):
    """Return an AsyncMock for _orchestrator.route that routes to *target*."""
    result = {"target_agent": target}
    if response:
        result["response"] = response
    mock = AsyncMock(return_value=result)
    return mock


def _mock_agent_handle(response_text: str, state_updates: dict | None = None):
    """Return an AsyncMock for agent.handle that returns a canned response."""
    return AsyncMock(return_value={
        "response": response_text,
        "state_updates": state_updates or {},
    })


# ─── Test: /start → onboarding ─────────────────────


class TestStartRoutesToOnboarding:
    """Verify /start reaches the onboarding node and produces a response."""

    @pytest.mark.asyncio
    async def test_start_routes_to_onboarding(self):
        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._onboarding") as mock_onb,
        ):
            mock_orch.route = _mock_orchestrator_route("onboarding")
            mock_onb.handle = _mock_agent_handle("Welcome! Where are you going?")

            graph = compile_graph()
            input_state = _make_state(messages=[{"role": "user", "content": "/start"}])
            result = await graph.ainvoke(input_state)

            # Onboarding was called
            mock_onb.handle.assert_called_once()
            # Response is in messages
            response = _extract_response(result)
            assert "Welcome" in response


# ─── Test: natural text pre-onboarding → onboarding ─


class TestNaturalTextRouting:

    @pytest.mark.asyncio
    async def test_free_text_pre_onboarding_routes_to_onboarding(self):
        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._onboarding") as mock_onb,
        ):
            mock_orch.route = _mock_orchestrator_route("onboarding")
            mock_onb.handle = _mock_agent_handle("Great, tell me about your trip!")

            graph = compile_graph()
            input_state = _make_state(
                messages=[{"role": "user", "content": "I want to go to Vietnam"}],
            )
            result = await graph.ainvoke(input_state)

            mock_onb.handle.assert_called_once()
            response = _extract_response(result)
            assert response != "I'm not sure how to respond to that. Try /help for available commands."
            assert "trip" in response.lower()


# ─── Test: /help handled directly (no specialist) ───


class TestHelpHandledDirectly:

    @pytest.mark.asyncio
    async def test_help_returns_help_text(self):
        with patch("src.graph._orchestrator") as mock_orch:
            mock_orch.route = _mock_orchestrator_route(
                "orchestrator",
                response="Available commands:\n\n/start — Begin planning",
            )

            graph = compile_graph()
            input_state = _make_state(
                messages=[{"role": "user", "content": "/help"}],
            )
            result = await graph.ainvoke(input_state)

            response = _extract_response(result)
            assert "/start" in response
            assert "Available commands" in response


# ─── Test: /status handled directly ─────────────────


class TestStatusHandledDirectly:

    @pytest.mark.asyncio
    async def test_status_returns_dashboard(self):
        with patch("src.graph._orchestrator") as mock_orch:
            mock_orch.route = _mock_orchestrator_route(
                "orchestrator",
                response="🇯🇵 Japan Trip — 14 Days — Planning Status\n\n✅ Onboarding complete",
            )

            graph = compile_graph()
            input_state = _onboarded_state(
                messages=[{"role": "user", "content": "/status"}],
            )
            result = await graph.ainvoke(input_state)

            response = _extract_response(result)
            assert "Japan" in response or "Onboarding" in response


# ─── Test: prerequisite guard blocks ────────────────


class TestPrerequisiteGuard:

    @pytest.mark.asyncio
    async def test_research_before_onboarding_is_guarded(self):
        with patch("src.graph._orchestrator") as mock_orch:
            mock_orch.route = _mock_orchestrator_route(
                "orchestrator",
                response="Let's set up your trip first!",
            )

            graph = compile_graph()
            input_state = _make_state(
                messages=[{"role": "user", "content": "/research"}],
            )
            result = await graph.ainvoke(input_state)

            response = _extract_response(result)
            assert "set up" in response.lower() or "trip first" in response.lower()


# ─── Test: all specialist routes ────────────────────


class TestAllSpecialistRoutes:
    """Parametrized: each command routes to the correct specialist node."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command,agent_name", [
        ("/start", "onboarding"),
        ("/research", "research"),
        ("/library", "librarian"),
        ("/priorities", "prioritizer"),
        ("/plan", "planner"),
        ("/agenda", "scheduler"),
        ("/feedback", "feedback"),
        ("/costs", "cost"),
    ])
    async def test_command_routes_to_specialist(self, command, agent_name):
        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._onboarding") as mock_onb,
            patch("src.graph._get_agent") as mock_get,
        ):
            mock_orch.route = _mock_orchestrator_route(agent_name)

            # Set up handler mocks
            agent_mock = AsyncMock()
            agent_mock.handle = _mock_agent_handle(f"{agent_name} response here")
            mock_onb.handle = _mock_agent_handle(f"{agent_name} response here")
            mock_get.return_value = agent_mock

            graph = compile_graph()
            input_state = _onboarded_state(
                messages=[{"role": "user", "content": command}],
            )
            result = await graph.ainvoke(input_state)

            response = _extract_response(result)
            assert response != "I'm not sure how to respond to that. Try /help for available commands."
            assert "response here" in response


# ─── Test: _user_message reaches specialist ─────────


class TestUserMessageReachesSpecialist:

    @pytest.mark.asyncio
    async def test_user_message_passed_to_onboarding(self):
        captured_args = {}

        async def capturing_handle(state, user_msg):
            captured_args["state"] = state
            captured_args["user_msg"] = user_msg
            return {"response": "Got it!", "state_updates": {}}

        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._onboarding") as mock_onb,
        ):
            mock_orch.route = _mock_orchestrator_route("onboarding")
            mock_onb.handle = AsyncMock(side_effect=capturing_handle)

            graph = compile_graph()
            input_state = _make_state(
                messages=[{"role": "user", "content": "Hello, planning a trip"}],
            )
            result = await graph.ainvoke(input_state)

            # The specialist should have received the user message
            assert captured_args.get("user_msg") == "Hello, planning a trip"

    @pytest.mark.asyncio
    async def test_user_message_in_state_for_specialist(self):
        """Verify _user_message is present in state when a specialist node runs."""
        captured_state = {}

        async def capturing_handle(state, user_msg):
            captured_state.update(state)
            return {"response": "Researching!", "state_updates": {}}

        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._get_agent") as mock_get,
        ):
            mock_orch.route = _mock_orchestrator_route("research")
            agent_mock = AsyncMock()
            agent_mock.handle = AsyncMock(side_effect=capturing_handle)
            mock_get.return_value = agent_mock

            graph = compile_graph()
            input_state = _onboarded_state(
                messages=[{"role": "user", "content": "/research Tokyo"}],
            )
            result = await graph.ainvoke(input_state)

            # _user_message should be set in the state passed to the specialist
            assert captured_state.get("_user_message") == "/research Tokyo"


# ─── Test: _extract_response finds assistant message ─


class TestExtractResponse:

    def test_finds_dict_assistant_message(self):
        result = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello there!"},
            ],
        }
        assert _extract_response(result) == "Hello there!"

    def test_returns_fallback_when_empty(self):
        result = {"messages": []}
        response = _extract_response(result)
        assert "not sure" in response.lower() or "help" in response.lower()

    def test_prefers_last_assistant_message(self):
        result = {
            "messages": [
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
        }
        assert _extract_response(result) == "second"


# ─── Test: onboarding error handling ────────────────


class TestOnboardingErrorHandling:

    @pytest.mark.asyncio
    async def test_onboarding_error_returns_fallback(self):
        with (
            patch("src.graph._orchestrator") as mock_orch,
            patch("src.graph._onboarding") as mock_onb,
        ):
            mock_orch.route = _mock_orchestrator_route("onboarding")
            mock_onb.handle = AsyncMock(side_effect=RuntimeError("LLM call failed"))

            graph = compile_graph()
            input_state = _make_state(
                messages=[{"role": "user", "content": "/start"}],
            )
            result = await graph.ainvoke(input_state)

            response = _extract_response(result)
            assert "issue" in response.lower() or "try again" in response.lower()


# ─── Test: research error isolation ────────────────


class TestResearchErrorIsolation:
    """Verify that per-city errors don't lose accumulated results."""

    @pytest.mark.asyncio
    async def test_research_all_continues_after_city_failure(self):
        """If one city research fails, others still succeed and state is preserved."""
        from src.agents.research import ResearchAgent

        agent = ResearchAgent.__new__(ResearchAgent)
        agent.search_tool = MagicMock()
        agent.llm = MagicMock()

        state = {
            "destination": {"country": "Japan", "researched_at": "2026-01-01T00:00:00Z"},
            "cities": [
                {"name": "Tokyo", "country": "Japan", "days": 3},
                {"name": "Kyoto", "country": "Japan", "days": 2},
            ],
            "research": {},
            "interests": ["food"],
            "travelers": {"type": "couple"},
            "dates": {"start": "2026-04-01", "end": "2026-04-06"},
        }

        call_count = 0

        async def _mock_research_city(s, city):
            nonlocal call_count
            call_count += 1
            city_name = city.get("name", "")
            if city_name == "Tokyo":
                raise RuntimeError("Simulated LLM timeout")
            return {
                "response": f"Research for {city_name} complete!",
                "state_updates": {"research": {city_name: {"places": [{"name": "Temple"}]}}},
            }

        agent._research_city = _mock_research_city

        result = await agent._research_all_cities(state)

        # Both cities were attempted
        assert call_count == 2
        # Kyoto research was preserved despite Tokyo failure
        assert "Kyoto" in result.get("state_updates", {}).get("research", {})
        # Response mentions failure for Tokyo
        response = result.get("response", "")
        assert "failed" in response.lower() or "⚠️" in response
        # Response also has Kyoto success
        assert "Kyoto" in response

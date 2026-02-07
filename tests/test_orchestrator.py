"""Tests for the orchestrator — command routing, prerequisite guards, status."""

from __future__ import annotations

import pytest

from src.agents.orchestrator import (
    COMMAND_MAP,
    OrchestratorAgent,
    generate_help,
    generate_status,
)


class TestCommandMap:
    """Verify all expected commands are mapped."""

    def test_all_commands_present(self):
        expected = {"/start", "/research", "/library", "/priorities", "/plan", "/agenda", "/feedback", "/costs", "/adjust", "/status", "/help", "/trips", "/trip", "/join"}
        assert expected == set(COMMAND_MAP.keys())

    def test_start_routes_to_onboarding(self):
        assert COMMAND_MAP["/start"] == "onboarding"

    def test_costs_routes_to_cost(self):
        assert COMMAND_MAP["/costs"] == "cost"

    def test_status_and_help_route_to_orchestrator(self):
        assert COMMAND_MAP["/status"] == "orchestrator"
        assert COMMAND_MAP["/help"] == "orchestrator"


class TestPrerequisiteGuards:
    """Verify prerequisite guard logic."""

    def test_blocks_planner_without_research(self, japan_state):
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "planner")
        assert result is not None
        assert "research" in result.lower()

    def test_blocks_planner_without_priorities(self, japan_state):
        japan_state["research"] = {"Tokyo": {"places": [], "food": []}}
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "planner")
        assert result is not None
        assert "priorities" in result.lower() or "priorit" in result.lower()

    def test_allows_research_after_onboarding(self, japan_state):
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "research")
        assert result is None

    def test_blocks_scheduler_without_plan(self, japan_state):
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "scheduler")
        assert result is not None
        assert "plan" in result.lower()

    def test_blocks_everything_before_onboarding(self, empty_state):
        orch = OrchestratorAgent()
        for agent in ("research", "prioritizer", "planner", "scheduler"):
            result = orch._check_prerequisites(empty_state, agent)
            assert result is not None


class TestStatusGeneration:
    """Verify dynamic status dashboard."""

    def test_status_before_onboarding(self, empty_state):
        status = generate_status(empty_state)
        assert "set up" in status.lower() or "start" in status.lower()

    def test_status_after_onboarding(self, japan_state):
        status = generate_status(japan_state)
        assert "Japan" in status
        assert "14 Days" in status or "14" in status
        assert "Onboarding complete" in status
        assert "0/3 cities" in status  # No research yet

    def test_status_uses_flag_from_state(self, japan_state):
        status = generate_status(japan_state)
        assert "\U0001f1ef\U0001f1f5" in status  # JP flag

    def test_status_morocco_different(self, morocco_state):
        status = generate_status(morocco_state)
        assert "Morocco" in status
        assert "5 Days" in status or "5" in status
        assert "\U0001f1f2\U0001f1e6" in status  # MA flag


class TestHelpGeneration:
    def test_help_lists_commands(self):
        help_text = generate_help()
        for cmd in ("/start", "/research", "/plan", "/costs", "/feedback"):
            assert cmd in help_text

    def test_help_includes_cost_subcommands(self):
        help_text = generate_help()
        assert "/costs today" in help_text
        assert "/costs save" in help_text
        assert "/costs convert" in help_text

    def test_help_includes_adjust(self):
        help_text = generate_help()
        assert "/adjust" in help_text

    def test_help_includes_groups(self):
        help_text = generate_help()
        assert "Setup" in help_text
        assert "Research & Planning" in help_text
        assert "On-Trip" in help_text
        assert "Meta" in help_text

    def test_help_includes_archive(self):
        help_text = generate_help()
        assert "/trip archive" in help_text


class TestCommandRouting:
    """Test synchronous command-based routing (no LLM calls)."""

    @pytest.mark.asyncio
    async def test_start_routes_to_onboarding(self, empty_state):
        orch = OrchestratorAgent()
        result = await orch.route(empty_state, "/start")
        assert result["target_agent"] == "onboarding"
        assert "welcome" in result.get("response", "").lower() or "get started" in result.get("response", "").lower()

    @pytest.mark.asyncio
    async def test_start_after_onboarding_complete_returns_guard(self, japan_state):
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/start")
        assert result["target_agent"] == "orchestrator"
        assert "/trip new" in result.get("response", "")

    @pytest.mark.asyncio
    async def test_status_returns_direct_response(self, japan_state):
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/status")
        assert result["target_agent"] == "orchestrator"
        assert result.get("response")
        assert "Japan" in result["response"]

    @pytest.mark.asyncio
    async def test_help_returns_commands(self, japan_state):
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/help")
        assert "/start" in result.get("response", "")

    @pytest.mark.asyncio
    async def test_unknown_command(self, japan_state):
        orch = OrchestratorAgent()
        result = await orch.route(japan_state, "/unknown")
        assert "unknown" in result.get("response", "").lower() or result["target_agent"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_onboarding_guard_for_free_text(self, empty_state):
        orch = OrchestratorAgent()
        result = await orch.route(empty_state, "I want to go to Japan")
        assert result["target_agent"] == "onboarding"

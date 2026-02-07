"""Tests for LangGraph compilation and routing."""

from __future__ import annotations

import pytest

from src.graph import build_graph, route_from_orchestrator
from src.state import TripState


class TestGraphCompilation:
    """Verify the graph builds correctly with all nodes."""

    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        graph = build_graph()
        expected_nodes = {
            "orchestrator", "onboarding", "research", "librarian",
            "prioritizer", "planner", "scheduler", "feedback", "cost",
        }
        # LangGraph includes __start__ and __end__ nodes
        node_names = set(graph.nodes.keys())
        assert expected_nodes.issubset(node_names)

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_entry_is_orchestrator(self):
        graph = build_graph()
        # The entry point should be orchestrator
        assert "orchestrator" in graph.nodes


class TestRouting:
    """Test the conditional routing function."""

    def test_route_to_onboarding(self):
        state = {"_next": "onboarding"}
        assert route_from_orchestrator(state) == "onboarding"

    def test_route_to_research(self):
        state = {"_next": "research"}
        assert route_from_orchestrator(state) == "research"

    def test_route_to_end(self):
        from langgraph.graph import END
        state = {"_next": END}
        assert route_from_orchestrator(state) == END

    def test_route_orchestrator_goes_to_end(self):
        from langgraph.graph import END
        state = {"_next": "orchestrator"}
        assert route_from_orchestrator(state) == END

    def test_route_invalid_goes_to_end(self):
        from langgraph.graph import END
        state = {"_next": "nonexistent_agent"}
        assert route_from_orchestrator(state) == END

    def test_all_valid_routes(self):
        valid = ["onboarding", "research", "librarian", "prioritizer", "planner", "scheduler", "feedback", "cost"]
        for agent in valid:
            state = {"_next": agent}
            assert route_from_orchestrator(state) == agent


class TestStateSchema:
    """Verify state schema types."""

    def test_trip_state_is_typeddict(self):
        # TripState should be usable as a dict
        state = TripState(
            trip_id="test",
            onboarding_complete=False,
            messages=[],
        )
        assert isinstance(state, dict)
        assert state["trip_id"] == "test"
        assert state["onboarding_complete"] is False

    def test_state_accepts_nested_types(self):
        state = TripState(
            trip_id="test",
            destination={"country": "Japan", "flag_emoji": "🇯🇵"},
            cities=[{"name": "Tokyo", "days": 4}],
            messages=[],
        )
        assert state["destination"]["country"] == "Japan"
        assert state["cities"][0]["name"] == "Tokyo"

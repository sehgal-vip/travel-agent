"""LangGraph graph definition — orchestrator + 8 specialist agent nodes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.base import add_to_conversation_history
from src.agents.onboarding import OnboardingAgent
from src.agents.orchestrator import OrchestratorAgent
from src.state import TripState

logger = logging.getLogger(__name__)

# Lazy singletons — instantiated once per process
_orchestrator = OrchestratorAgent()
_onboarding = OnboardingAgent()

# Specialist agents imported lazily to avoid circular imports at module level
_agents: dict[str, Any] = {}


def _get_agent(name: str):
    """Lazy-load specialist agents."""
    if name not in _agents:
        if name == "research":
            from src.agents.research import ResearchAgent
            _agents[name] = ResearchAgent()
        elif name == "librarian":
            from src.agents.librarian import LibrarianAgent
            _agents[name] = LibrarianAgent()
        elif name == "prioritizer":
            from src.agents.prioritizer import PrioritizerAgent
            _agents[name] = PrioritizerAgent()
        elif name == "planner":
            from src.agents.planner import PlannerAgent
            _agents[name] = PlannerAgent()
        elif name == "scheduler":
            from src.agents.scheduler import SchedulerAgent
            _agents[name] = SchedulerAgent()
        elif name == "feedback":
            from src.agents.feedback import FeedbackAgent
            _agents[name] = FeedbackAgent()
        elif name == "cost":
            from src.agents.cost import CostAgent
            _agents[name] = CostAgent()
    return _agents.get(name)


# ─── Node functions ──────────────────────────────


async def orchestrator_node(state: TripState) -> dict:
    """Entry node — routes every message to the correct agent."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [{"role": "assistant", "content": "Send /start to begin planning!"}]}

    last_msg = messages[-1]
    user_text = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")

    routing = await _orchestrator.route(state, user_text)
    target = routing.get("target_agent", "orchestrator")
    direct_response = routing.get("response")

    if direct_response and target == "orchestrator":
        history = add_to_conversation_history(state, "user", user_text)
        history.append({
            "role": "assistant",
            "content": direct_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "orchestrator",
        })
        return {
            "messages": [{"role": "assistant", "content": direct_response}],
            "conversation_history": history,
            "current_agent": "orchestrator",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "_next": END,
        }

    return {
        "current_agent": target,
        "_user_message": user_text,
        "_next": target,
    }


async def onboarding_node(state: TripState) -> dict:
    """Onboarding agent node."""
    user_msg = state.get("_user_message", "")
    if not user_msg:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            user_msg = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")

    try:
        result = await _onboarding.handle(state, user_msg)
        response = result.get("response", "")
        updates = result.get("state_updates", {})
    except Exception:
        logger.exception("Error in onboarding agent")
        response = "I encountered an issue. Please try again or send /help."
        updates = {}

    output = {
        "messages": [{"role": "assistant", "content": response}],
        **updates,
        "_next": END,
    }
    return output


async def _specialist_node(agent_name: str, state: TripState) -> dict:
    """Generic specialist agent node."""
    agent = _get_agent(agent_name)
    if not agent:
        return {
            "messages": [{"role": "assistant", "content": f"The {agent_name} agent is not available yet."}],
            "_next": END,
        }

    user_msg = state.get("_user_message", "")
    if not user_msg:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            user_msg = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")

    try:
        result = await agent.handle(state, user_msg)
        response = result.get("response", "")
        updates = result.get("state_updates", {})
    except Exception:
        logger.exception("Error in %s agent", agent_name)
        response = f"I encountered an issue. Please try again or send /help."
        updates = {}

    history = add_to_conversation_history(state, "user", user_msg)
    history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
    })

    return {
        "messages": [{"role": "assistant", "content": response}],
        "conversation_history": history,
        "current_agent": agent_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **updates,
        "_next": END,
    }


# Node factory functions for each specialist
async def research_node(state: TripState) -> dict:
    return await _specialist_node("research", state)

async def librarian_node(state: TripState) -> dict:
    return await _specialist_node("librarian", state)

async def prioritizer_node(state: TripState) -> dict:
    return await _specialist_node("prioritizer", state)

async def planner_node(state: TripState) -> dict:
    return await _specialist_node("planner", state)

async def scheduler_node(state: TripState) -> dict:
    return await _specialist_node("scheduler", state)

async def feedback_node(state: TripState) -> dict:
    return await _specialist_node("feedback", state)

async def cost_node(state: TripState) -> dict:
    return await _specialist_node("cost", state)


# ─── Routing ─────────────────────────────────────


def route_from_orchestrator(state: TripState) -> str:
    """Conditional edge — read _next from state to decide the next node."""
    next_node = state.get("_next", END)
    if next_node == END or next_node == "orchestrator":
        return END
    valid = {"onboarding", "research", "librarian", "prioritizer", "planner", "scheduler", "feedback", "cost"}
    return next_node if next_node in valid else END


# ─── Graph construction ──────────────────────────


def build_graph(checkpointer=None) -> StateGraph:
    """Build the LangGraph StateGraph with all agent nodes."""
    graph = StateGraph(TripState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("research", research_node)
    graph.add_node("librarian", librarian_node)
    graph.add_node("prioritizer", prioritizer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("cost", cost_node)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Conditional edges from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "onboarding": "onboarding",
            "research": "research",
            "librarian": "librarian",
            "prioritizer": "prioritizer",
            "planner": "planner",
            "scheduler": "scheduler",
            "feedback": "feedback",
            "cost": "cost",
            END: END,
        },
    )

    # All specialist nodes end after processing
    for node_name in ("onboarding", "research", "librarian", "prioritizer", "planner", "scheduler", "feedback", "cost"):
        graph.add_edge(node_name, END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with an optional checkpointer."""
    graph = build_graph(checkpointer)
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled with %d nodes.", len(graph.nodes))
    return compiled

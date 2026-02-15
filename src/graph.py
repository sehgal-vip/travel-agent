"""LangGraph graph definition — orchestrator + 7 specialist agent nodes (v2)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from src.agents.base import add_to_conversation_history
from src.agents.onboarding import OnboardingAgent
from src.agents.orchestrator import OrchestratorAgent
from src.state import TripState

logger = logging.getLogger(__name__)

# ─── Agent sets ─────────────────────────────────

SPECIALIST_AGENTS = {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost"}
LOOPBACK_AGENTS = {"research", "planner", "feedback"}  # agents that can loop back to user

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
    # v2: Check graph control keys before normal routing
    if state.get("_awaiting_input"):
        # Resuming from a loopback — route back to the agent that requested input
        target = state["_awaiting_input"]
        return {
            "current_agent": target,
            "_awaiting_input": None,
            "_next": target,
        }

    if state.get("_callback"):
        target = state["_callback"]
        return {
            "current_agent": target,
            "_callback": None,
            "_next": target,
        }

    if state.get("_delegate_to"):
        target = state["_delegate_to"]
        return {
            "current_agent": target,
            "_delegate_to": None,
            "_next": target,
        }

    chain = state.get("_chain", [])
    if chain:
        target = chain[0]
        remaining = chain[1:]
        return {
            "current_agent": target,
            "_chain": remaining,
            "_next": target,
        }

    # Normal routing
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

    result = {
        "current_agent": target,
        "_user_message": user_text,
        "_next": target,
    }

    # Pass through routing echo (e.g. "Let me check with the research team...")
    if direct_response and target != "orchestrator":
        result["_routing_echo"] = direct_response

    return result


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

    # Prepend routing echo if present (e.g. welcome message on first /start)
    routing_echo = state.get("_routing_echo", "")
    if routing_echo:
        response = f"{routing_echo}\n\n{response}"

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
        _agent_error_messages = {
            "research": "I had trouble researching that. Try again or try a specific city.",
            "planner": "I had trouble generating the plan. Try /plan again.",
            "scheduler": "I had trouble building the agenda. Try /agenda again.",
            "feedback": "I had trouble processing your feedback. Try /feedback again.",
            "cost": "I had trouble with the cost calculation. Try /costs again.",
        }
        response = _agent_error_messages.get(
            agent_name, "Something went wrong. Try your last command again."
        )
        updates = {}

    # Prepend routing echo if present (e.g. "Let me check with the research team...")
    routing_echo = state.get("_routing_echo", "")
    if routing_echo:
        response = f"{routing_echo}\n\n{response}"

    history = add_to_conversation_history(state, "user", user_msg)
    history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
    })

    # Increment loopback depth
    loopback_depth = state.get("_loopback_depth", 0) + 1

    output = {
        "messages": [{"role": "assistant", "content": response}],
        "conversation_history": history,
        "current_agent": agent_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "_loopback_depth": loopback_depth,
        **updates,
        "_next": END,
    }

    # v2: Check for loopback signals in state_updates
    if agent_name in LOOPBACK_AGENTS:
        if updates.get("_awaiting_input"):
            output["_awaiting_input"] = updates["_awaiting_input"]
            return Command(update=output, goto="orchestrator")

        if updates.get("_delegate_to"):
            delegate_target = updates["_delegate_to"]
            output["_delegate_to"] = delegate_target
            return Command(update=output, goto=delegate_target)

        if updates.get("_callback"):
            output["_callback"] = updates["_callback"]

    return output


async def error_handler_node(state: TripState) -> dict:
    """Handle errors from specialist agents."""
    error_agent = state.get("_error_agent", "unknown")
    error_context = state.get("_error_context", "An error occurred")
    return {
        "messages": [{"role": "assistant", "content": f"I ran into an issue with {error_agent}. {error_context} Please try again."}],
        "_error_agent": None,
        "_error_context": None,
        "_next": END,
    }


# ─── Node factory ───────────────────────────────


def _make_specialist_node(name: str):
    async def node_fn(state: TripState) -> dict:
        return await _specialist_node(name, state)
    node_fn.__name__ = f"{name}_node"
    return node_fn


# ─── Routing ─────────────────────────────────────


def route_from_orchestrator(state: TripState) -> str:
    """Conditional edge — read _next from state to decide the next node."""
    next_node = state.get("_next", END)
    if next_node == END or next_node == "orchestrator":
        return END
    valid = {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost", "error_handler"}
    return next_node if next_node in valid else END


def route_from_specialist(state: TripState) -> str:
    """Conditional edge for loopback agents — check for delegation / awaiting input."""
    if state.get("_awaiting_input"):
        return END  # Wait for next user message
    if state.get("_delegate_to"):
        target = state["_delegate_to"]
        return target if target in SPECIALIST_AGENTS else END
    if state.get("_error_agent"):
        return "error_handler"
    return END


# ─── Graph construction ──────────────────────────


def build_graph(checkpointer=None) -> StateGraph:
    """Build the LangGraph StateGraph with all agent nodes."""
    graph = StateGraph(TripState)

    # Add orchestrator and onboarding nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("onboarding", onboarding_node)

    # Add specialist nodes via factory loop
    for agent_name in SPECIALIST_AGENTS:
        if agent_name == "onboarding":
            continue  # already added above
        graph.add_node(agent_name, _make_specialist_node(agent_name))

    # Add error handler node
    graph.add_node("error_handler", error_handler_node)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Conditional edges from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "onboarding": "onboarding",
            "research": "research",
            "prioritizer": "prioritizer",
            "planner": "planner",
            "scheduler": "scheduler",
            "feedback": "feedback",
            "cost": "cost",
            "error_handler": "error_handler",
            END: END,
        },
    )

    # Onboarding always ends after processing
    graph.add_edge("onboarding", END)

    # Loopback agents get conditional edges
    for agent_name in LOOPBACK_AGENTS:
        graph.add_conditional_edges(
            agent_name,
            route_from_specialist,
            {
                "error_handler": "error_handler",
                END: END,
                **{a: a for a in SPECIALIST_AGENTS if a != "onboarding"},
            },
        )

    # Non-loopback specialist agents get direct edge to END
    non_loopback = SPECIALIST_AGENTS - LOOPBACK_AGENTS - {"onboarding"}
    for agent_name in non_loopback:
        graph.add_edge(agent_name, END)

    # Error handler always ends
    graph.add_edge("error_handler", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile the graph with an optional checkpointer."""
    graph = build_graph(checkpointer)
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled with %d nodes.", len(graph.nodes))
    return compiled

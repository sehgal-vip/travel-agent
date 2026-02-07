"""Orchestrator agent — routes every user message to the correct specialist."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent, add_to_conversation_history, format_money
from src.state import TripState

logger = logging.getLogger(__name__)

COMMAND_MAP: dict[str, str] = {
    "/start": "onboarding",
    "/research": "research",
    "/library": "librarian",
    "/priorities": "prioritizer",
    "/plan": "planner",
    "/agenda": "scheduler",
    "/feedback": "feedback",
    "/costs": "cost",
    "/adjust": "feedback",
    "/status": "orchestrator",
    "/help": "orchestrator",
    "/trips": "orchestrator",
    "/trip": "orchestrator",
    "/join": "orchestrator",
}

# Agents that require prior steps to have been completed
PREREQUISITES: dict[str, list[tuple[str, str]]] = {
    "prioritizer": [("research", "You need to research cities first. Try /research all")],
    "planner": [
        ("research", "You need to research cities first. Try /research all"),
        ("priorities", "You need to set priorities first. Try /priorities"),
    ],
    "scheduler": [
        ("high_level_plan", "You need a plan first. Try /plan"),
    ],
}


class OrchestratorAgent(BaseAgent):
    agent_name = "orchestrator"

    def get_system_prompt(self) -> str:
        return (
            "You are the Orchestrator of a multi-agent travel planning assistant. "
            "You work for ANY destination worldwide.\n\n"
            "Your job is to:\n"
            "1. Interpret the user's message and determine intent\n"
            "2. Route to the correct specialist agent\n"
            "3. Handle meta-conversations (greetings, status checks, unclear requests)\n"
            "4. Maintain conversation continuity across agent handoffs\n\n"
            "ROUTING RULES:\n"
            "- If no trip exists or onboarding is not complete → route to ONBOARDING\n"
            "- Research requests → RESEARCH\n"
            "- Priority/ranking questions → PRIORITIZER\n"
            "- Library/knowledge base requests → LIBRARIAN\n"
            "- Plan/itinerary requests → PLANNER\n"
            "- Detailed agenda / 'what today' → SCHEDULER\n"
            "- Feedback about their day → FEEDBACK\n"
            "- Cost/budget/spending questions → COST\n\n"
            "When intent is unclear, ask ONE clarifying question.\n"
            "When just chatting, be friendly and suggest what they can do next.\n\n"
            "NEVER expose internal agent names or system architecture.\n"
            "NEVER assume any destination — always read from state.\n\n"
            "Respond with ONLY a JSON object: {\"route\": \"<agent_name>\", \"message\": \"<any message to show user>\"}\n"
            "Valid routes: onboarding, research, librarian, prioritizer, planner, scheduler, feedback, cost, orchestrator"
        )

    async def route(self, state: TripState, message: str) -> dict[str, Any]:
        """Determine which agent should handle the message.

        Returns dict with 'target_agent' and optional 'response' for meta-commands.
        """
        # 1. Command-based routing
        if message.startswith("/"):
            cmd = message.split()[0].lower()
            target = COMMAND_MAP.get(cmd)

            if target:
                # Handle meta-commands directly
                if cmd == "/status":
                    return {"target_agent": "orchestrator", "response": generate_status(state)}
                if cmd == "/help":
                    return {"target_agent": "orchestrator", "response": generate_help()}
                if cmd == "/trips":
                    return {"target_agent": "orchestrator", "response": "__list_trips__"}
                if cmd == "/trip":
                    return await self._handle_trip_command(state, message)

                # Check prerequisites before routing
                guard = self._check_prerequisites(state, target)
                if guard:
                    return {"target_agent": "orchestrator", "response": guard}

                return {"target_agent": target}

            return {"target_agent": "orchestrator", "response": f"Unknown command: {cmd}. Try /help for available commands."}

        # 2. Onboarding guard — if not complete, always go to onboarding
        if not state.get("onboarding_complete"):
            return {"target_agent": "onboarding"}

        # 3. LLM-based intent classification
        return await self._classify_intent(state, message)

    def _check_prerequisites(self, state: TripState, target: str) -> str | None:
        """Return an error message if prerequisites are not met, else None."""
        if not state.get("onboarding_complete") and target not in ("onboarding", "orchestrator"):
            return "Let's set up your trip first! Send /start to begin."

        for field, msg in PREREQUISITES.get(target, []):
            if field == "research" and not state.get("research"):
                return msg
            if field == "priorities" and not state.get("priorities"):
                return msg
            if field == "high_level_plan" and not state.get("high_level_plan"):
                return msg
        return None

    async def _classify_intent(self, state: TripState, message: str) -> dict[str, Any]:
        """Use LLM to classify natural-language intent into an agent route."""
        classification_prompt = (
            "Classify the user's intent into one of these agents:\n"
            "- onboarding (trip setup, changes to trip config)\n"
            "- research (learn about places, cities, things to do)\n"
            "- librarian (library, knowledge base, markdown files)\n"
            "- prioritizer (rank, prioritize, tier, must-do)\n"
            "- planner (itinerary, plan, schedule days)\n"
            "- scheduler (detailed agenda, what to do today/tomorrow)\n"
            "- feedback (how was today, rate experience, adjustments)\n"
            "- cost (budget, spending, money, prices, costs)\n"
            "- orchestrator (general chat, unclear, greeting)\n\n"
            "Respond with ONLY the agent name, nothing else."
        )

        messages = [
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"User message: {message}"),
        ]

        response = await self.llm.ainvoke(messages)
        agent_name = response.content.strip().lower()

        valid_agents = {
            "onboarding", "research", "librarian", "prioritizer",
            "planner", "scheduler", "feedback", "cost", "orchestrator",
        }

        if agent_name not in valid_agents:
            agent_name = "orchestrator"

        # Check prerequisites
        guard = self._check_prerequisites(state, agent_name)
        if guard:
            return {"target_agent": "orchestrator", "response": guard}

        if agent_name == "orchestrator":
            # Handle conversationally
            response_text = await self.invoke(state, message)
            return {"target_agent": "orchestrator", "response": response_text}

        return {"target_agent": agent_name}

    async def _handle_trip_command(self, state: TripState, message: str) -> dict[str, Any]:
        """Handle /trip new, /trip switch <id>, /trip archive <id>."""
        parts = message.split()
        if len(parts) < 2:
            return {"target_agent": "orchestrator", "response": "Usage: /trip new | /trip switch <id> | /trip archive <id>"}

        sub = parts[1].lower()
        if sub == "new":
            return {"target_agent": "onboarding", "response": "__new_trip__"}
        if sub == "switch" and len(parts) >= 3:
            return {"target_agent": "orchestrator", "response": f"__switch_trip__{parts[2]}"}
        if sub == "archive" and len(parts) >= 3:
            return {"target_agent": "orchestrator", "response": f"__archive_trip__{parts[2]}"}

        return {"target_agent": "orchestrator", "response": "Usage: /trip new | /trip switch <id> | /trip archive <id>"}


def generate_status(state: TripState) -> str:
    """Build a dynamic status dashboard from current state."""
    if not state.get("onboarding_complete"):
        return "Your trip hasn't been set up yet. Send /start to begin planning!"

    dest = state.get("destination", {})
    cities = state.get("cities", [])
    dates = state.get("dates", {})
    flag = dest.get("flag_emoji", "")
    country = dest.get("country", "Unknown")
    total_days = dates.get("total_days", "?")

    researched = [c for c in cities if c.get("name") in (state.get("research") or {})]
    prioritized = [c for c in cities if c.get("name") in (state.get("priorities") or {})]

    plan_status = state.get("plan_status", "not_started")
    has_plan = bool(state.get("high_level_plan"))
    has_agenda = bool(state.get("detailed_agenda"))

    cost_tracker = state.get("cost_tracker", {})
    totals = cost_tracker.get("totals", {})
    spent = totals.get("spent_usd", 0)
    currency = dest.get("currency_code", "USD")

    lines = [
        f"{flag} {country} Trip — {total_days} Days — Planning Status",
        "",
        f"{'✅' if state.get('onboarding_complete') else '❌'} Onboarding complete",
        f"{'✅' if researched else '❌'} Research: {len(researched)}/{len(cities)} cities",
        f"{'✅' if prioritized else '❌'} Priorities: {len(prioritized)}/{len(cities)} cities",
        f"{'✅' if has_plan else '❌'} Itinerary: {'Approved' if plan_status == 'approved' else 'Draft' if has_plan else 'Not started'}",
        f"{'✅' if has_agenda else '❌'} Detailed agenda: {'Generated' if has_agenda else 'Not started'}",
        f"💰 Budget: {currency} / ${spent:,.0f} spent",
    ]

    return "\n".join(lines)


def generate_help() -> str:
    """Return the help text listing all available commands."""
    return (
        "Available commands:\n\n"
        "/start — Begin planning a new trip\n"
        "/research <city> — Research a specific city\n"
        "/research all — Research all cities\n"
        "/library — Sync your markdown knowledge base\n"
        "/priorities — View or adjust priority tiers\n"
        "/plan — Generate or view your itinerary\n"
        "/agenda — Get detailed agenda for the next 2 days\n"
        "/feedback — End-of-day check-in\n"
        "/costs — View budget breakdown\n"
        "/status — Trip planning progress\n"
        "/trips — List all your trips\n"
        "/trip new — Start a new trip\n"
        "/trip switch <id> — Switch to a different trip\n"
        "/trip id — Show your current trip ID (shareable)\n"
        "/join <id> — Join a trip shared by a travel companion\n"
        "/help — Show this help message\n\n"
        "Or just chat naturally — I'll understand what you need!"
    )

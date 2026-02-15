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
    "/priorities": "prioritizer",
    "/plan": "planner",
    "/agenda": "scheduler",
    "/feedback": "feedback",
    "/costs": "cost",
    "/adjust": "feedback",
    "/status": "orchestrator",
    "/help": "orchestrator",
    "/trips": "orchestrator",
    "/mytrips": "orchestrator",
    "/trip": "orchestrator",
    "/join": "orchestrator",
}

# Agents that require prior steps — soft guards that offer to help
PREREQUISITES: dict[str, list[tuple[str, str]]] = {
    "prioritizer": [(
        "research",
        "I'd need research findings to help prioritize. "
        "Want me to research your cities first? Just say 'yes' or /research all"
    )],
    "planner": [
        (
            "research",
            "I'll need research on your cities before building an itinerary. "
            "Want me to kick that off? /research all"
        ),
    ],
    "scheduler": [(
        "high_level_plan",
        "I need a day-by-day plan before making a detailed agenda. "
        "Want me to draft one? /plan"
    )],
}


class OrchestratorAgent(BaseAgent):
    agent_name = "orchestrator"

    def get_system_prompt(self, state: TripState | None = None) -> str:
        """Phase-aware system prompt — shifts from strategist to EA on-trip."""
        identity = (
            "You are the lead coordinator of a travel planning team. "
            "You know everything about the user's trip — it's in your Trip Memory context. "
            "You work for ANY destination worldwide.\n\n"
        )

        # Determine mode from state
        on_trip = False
        if state:
            has_agenda = bool(state.get("detailed_agenda"))
            has_feedback = bool(state.get("feedback_log"))
            on_trip = has_agenda or has_feedback

        if on_trip:
            mode = (
                "MODE: EXECUTIVE ASSISTANT (on-trip)\n"
                "You are running the user's day. Be structured and proactive.\n"
                "- Give clear schedules: times, locations, routes, duration\n"
                "- Anticipate logistics: weather, transport, opening hours, crowds\n"
                "- Track budget without being asked — flag if overspending\n"
                "- Handle problems creatively: 'Rain tomorrow — I've moved the garden to Thursday'\n"
                "- End-of-day: prompt for a quick check-in, suggest adjustments\n"
                "- Keep responses organized: use bullet points, time blocks, clear headers\n"
            )
        else:
            mode = (
                "MODE: TRAVEL STRATEGIST (pre-trip)\n"
                "You are a brainstorming partner helping shape the trip.\n"
                "- Read conversational cues — don't wait for commands\n"
                "- 'I'm thinking about food' → talk about food spots, offer to research\n"
                "- 'What should I definitely not miss?' → discuss priorities\n"
                "- Be specific to their destination and interests\n"
                "- When a prerequisite is missing, offer to handle it\n"
                "- Bounce ideas, suggest what's next, connect dots\n"
            )

        rules = (
            "\nTONE: Warm, direct, knowledgeable — like a sharp, well-traveled friend. "
            "Never cheesy. Never robotic. Never vague.\n\n"
            "RULES:\n"
            "- NEVER expose internal agent names or system architecture\n"
            "- NEVER assume any destination — always read from Trip Memory / state\n"
            "- When genuinely unsure, ask ONE clarifying question\n"
            "- Suggest relevant commands as a convenience, not as the only way to interact"
        )

        return identity + mode + rules

    async def route(self, state: TripState, message: str) -> dict[str, Any]:
        """Determine which agent should handle the message.

        Returns dict with 'target_agent' and optional 'response' for meta-commands.
        """
        # v2 loopback handling — priority routing
        # Priority 1: Agent waiting for user reply
        if state.get("_awaiting_input"):
            target = state["_awaiting_input"]
            if target in {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost"}:
                return {"target_agent": target}

        # Priority 2: Callback from delegation
        if state.get("_callback"):
            target = state["_callback"]
            if target in {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost"}:
                return {"target_agent": target}

        # Priority 3: Delegate to another agent
        if state.get("_delegate_to"):
            target = state["_delegate_to"]
            if target in {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost"}:
                return {"target_agent": target}

        # Priority 4: Pop from chain
        chain = state.get("_chain", [])
        if chain:
            target = chain[0]  # Will be popped by graph
            if target in {"onboarding", "research", "prioritizer", "planner", "scheduler", "feedback", "cost"}:
                return {"target_agent": target}

        # Loopback depth enforcement
        if state.get("_loopback_depth", 0) > 5:
            return {"target_agent": "orchestrator", "response": "I've been going back and forth too much. Let me reset. How can I help?"}

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
                if cmd in ("/trips", "/mytrips"):
                    return {"target_agent": "orchestrator", "response": "__list_trips__"}
                if cmd == "/trip":
                    return await self._handle_trip_command(state, message)

                # Welcome back when onboarding is already complete
                if cmd == "/start" and state.get("onboarding_complete"):
                    return {
                        "target_agent": "orchestrator",
                        "response": await self._welcome_back_with_ideas(state),
                    }

                # Welcome message on first /start
                if cmd == "/start" and not state.get("onboarding_complete"):
                    return {
                        "target_agent": "onboarding",
                        "response": (
                            "Welcome to your travel planning assistant! "
                            "I'll help you plan an amazing trip step by step. Let's get started!"
                        ),
                    }

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

    _MAX_ROUTING_CONTEXT_LINES = 12  # Hard cap to prevent context creep

    def _get_state_summary(self, state: TripState) -> str:
        """Enriched state summary for routing context, capped at ~200 tokens."""
        if not state.get("onboarding_complete"):
            return "Onboarding: in progress"

        parts: list[str] = []
        dest = state.get("destination", {})
        parts.append(f"Destination: {dest.get('country', '?')}")

        # Research status (1 line)
        research = state.get("research", {})
        cities = state.get("cities", [])
        if research:
            researched = sum(1 for c in cities if c.get("name") in research)
            items = sum(
                sum(len(d.get(k, [])) for k in ("places", "activities", "food"))
                for d in research.values()
            )
            parts.append(f"Research: {researched}/{len(cities)} cities, {items} items")
        else:
            parts.append("Research: not started")

        # Priorities (1 line)
        priorities = state.get("priorities", {})
        parts.append(f"Priorities: {len(priorities)} cities prioritized" if priorities else "Priorities: not done")

        # Plan (1 line)
        plan = state.get("high_level_plan", [])
        parts.append(f"Plan: {len(plan)} days" if plan else "Plan: not done")

        # Agenda (1 line)
        agenda = state.get("detailed_agenda", [])
        parts.append(f"Agenda: {len(agenda)} days detailed" if agenda else "Agenda: not done")

        # Feedback (1 line)
        feedback = state.get("feedback_log", [])
        if feedback:
            parts.append(
                f"Feedback: {len(feedback)} entries, "
                f"last energy={feedback[-1].get('energy_level', '?')}"
            )

        # Budget (1 line)
        totals = state.get("cost_tracker", {}).get("totals", {})
        if totals.get("spent_usd"):
            parts.append(f"Budget: ${totals['spent_usd']:,.0f} spent")

        return "\n".join(parts[:self._MAX_ROUTING_CONTEXT_LINES])

    async def _classify_intent(self, state: TripState, message: str) -> dict[str, Any]:
        """Use LLM to classify natural-language intent into an agent route."""
        state_summary = self._get_state_summary(state)

        # On-trip mode: bias toward action-oriented routing
        on_trip = bool(state.get("detailed_agenda") or state.get("feedback_log"))

        classification_prompt = (
            "You route user messages to the right specialist. "
            "Read the user's INTENT, not just their words.\n\n"
            "Agents:\n"
            "- onboarding: trip setup, change dates/cities/budget/travelers\n"
            "- research: curious about places, food, activities, 'what's there to do', 'tell me about X'\n"
            "- prioritizer: rank, prioritize, 'what's most important', 'must-do'\n"
            "- planner: itinerary, schedule, 'plan my days', 'what order'\n"
            "- scheduler: detailed agenda, 'what's today', 'what's tomorrow', 'next 2 days'\n"
            "- feedback: 'how was today', rate, adjust, 'that was great/bad', 'change plans', energy level\n"
            "- cost: budget, spending, prices, 'how much', money, convert currency\n"
            "- orchestrator: general chat, greeting, 'what's next', meta questions\n\n"
            "CUE EXAMPLES:\n"
            "- 'I'm thinking about food in Tokyo' → research\n"
            "- 'That was a long day' → feedback\n"
            "- 'We spent a lot today' → cost\n"
            "- 'How much have I spent?' → cost\n"
            "- 'What should I definitely not miss?' → prioritizer\n"
            "- 'Can we move the Kyoto day?' → planner\n"
            "- 'What's the plan for tomorrow?' → scheduler\n"
            "- 'It's raining, what now?' → scheduler\n"
            "- 'We're tired, anything chill nearby?' → scheduler\n\n"
        )

        if on_trip:
            classification_prompt += (
                "CONTEXT: User is CURRENTLY ON THEIR TRIP. "
                "Bias toward scheduler (logistics), feedback (experiences), and cost (spending). "
                "Any mention of today/tomorrow/time → scheduler. "
                "Any mention of feelings/energy/experience → feedback.\n\n"
            )

        classification_prompt += (
            f"TRIP STATE:\n{state_summary}\n\n"
            "Respond with ONLY the agent name."
        )

        messages = [
            SystemMessage(content=classification_prompt),
            HumanMessage(content=f"User message: {message}"),
        ]

        response = await self.llm.ainvoke(messages)
        agent_name = response.content.strip().lower()

        valid_agents = {
            "onboarding", "research", "prioritizer",
            "planner", "scheduler", "feedback", "cost", "orchestrator",
        }

        if agent_name not in valid_agents:
            agent_name = "orchestrator"

        # Clear stale _awaiting_input if LLM routes to a different agent
        awaiting = state.get("_awaiting_input")
        if awaiting and agent_name != awaiting:
            # Will be cleared when state is saved
            pass  # The routing will override, clearing happens in handler

        # Check prerequisites — soft guards that offer to help
        guard = self._check_prerequisites(state, agent_name)
        if guard:
            return {"target_agent": "orchestrator", "response": guard}

        if agent_name == "orchestrator":
            response_text = await self.invoke(state, message)
            return {"target_agent": "orchestrator", "response": response_text}

        return {"target_agent": agent_name}

    async def _welcome_back_with_ideas(self, state: TripState) -> str:
        """Show trip summary + contextual next-step ideas.

        Reads phase info from state directly (no memory files needed).
        """
        dest = state.get("destination", {})
        flag = dest.get("flag_emoji", "")
        title = state.get("trip_title") or f"{dest.get('country', 'Your')} Trip"

        # Build context from state directly (no file reads)
        context = generate_status(state)

        phase_prompt = self._get_phase_prompt(state)
        response = await self.llm.ainvoke([
            SystemMessage(content=(
                "The user is returning to their trip. You have the full trip context below.\n\n"
                "Write a SHORT welcome-back message (3-6 lines) that:\n"
                "1. Acknowledges where they are in planning (be specific — mention city names, dates)\n"
                "2. Gives 2-3 concrete, destination-specific ideas for what to do next\n"
                "3. Ends with one clear action they can take right now\n\n"
                "Tone: warm, direct, knowledgeable. Like a friend who remembers everything "
                "about your trip. Not cheesy, not corporate.\n\n"
                f"TRIP CONTEXT:\n{context}"
            )),
            HumanMessage(content=phase_prompt),
        ])

        header = f"{flag} **{title}**\n"
        return f"{header}\n{response.content}"

    def _get_phase_prompt(self, state: TripState) -> str:
        """Generate a phase-appropriate prompt for the welcome-back message."""
        dest = state.get("destination", {})
        country = dest.get("country", "their destination")
        cities = [c.get("name", "?") for c in state.get("cities", [])]
        interests = ", ".join(state.get("interests", []))

        if not state.get("research"):
            return (
                f"Trip to {country} is anchored — cities: {', '.join(cities)}. "
                f"Interests: {interests}. "
                "They haven't researched yet. Suggest specific things worth looking into "
                f"for {country} (seasonal, cultural, food). Mention /research all."
            )
        if not state.get("high_level_plan"):
            researched = list((state.get("research") or {}).keys())
            if not state.get("priorities"):
                return (
                    f"Research done for: {', '.join(researched)}. "
                    "They're ready to build their itinerary. "
                    "Mention /plan as the next step, and /priorities as optional."
                )
            return (
                "Research and priorities done. Help them think about day structure — "
                f"city order, pacing for {len(cities)} cities. Mention /plan."
            )
        if not state.get("detailed_agenda"):
            return (
                "Itinerary exists. Suggest getting a detailed agenda for the first days. "
                "Mention /agenda."
            )
        return (
            "Trip is fully planned. Suggest pre-trip prep specific to "
            f"{country} — what to download, book, pack. Keep it practical."
        )

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
        f"💰 Budget: ${spent:,.0f} USD spent" + (f" ({currency})" if currency != "USD" else ""),
    ]

    trip_id = state.get("trip_id")
    if trip_id:
        trip_title = state.get("trip_title") or f"{dest.get('country', 'Your')} Trip"
        lines.append(f"\n✈️ *{trip_title}*")
        lines.append(f"🆔 Trip ID: `{trip_id}` — share with /join {trip_id}")

    # Next-step suggestion
    if not researched:
        lines.append("\n👉 Next: /research all")
    elif not has_plan:
        if not prioritized:
            lines.append("\n👉 Next: /plan (or /priorities to fine-tune first)")
        else:
            lines.append("\n👉 Next: /plan")
    elif not has_agenda:
        lines.append("\n👉 Next: /agenda")
    else:
        lines.append("\n👉 You're all set! Use /feedback during your trip.")

    return "\n".join(lines)


def generate_help() -> str:
    """Return the help text listing all available commands."""
    return (
        "Here's everything I can do:\n\n"
        "--- Setup ---\n"
        "/start — Begin planning a new trip\n"
        "/join <id> — Join a trip shared by a travel companion\n"
        "/trip new — Start a new trip\n"
        "/trip switch <id> — Switch to a different trip\n"
        "/trip id — Show your current trip ID (shareable)\n"
        "/trip archive <id> — Archive a completed trip\n\n"
        "--- Research & Planning ---\n"
        "/research <city> — Research a specific city\n"
        "/research all — Research all cities\n"
        "/library — Sync your markdown knowledge base\n"
        "/priorities — View or adjust priority tiers\n"
        "/plan — Generate or view your itinerary\n"
        "/adjust — Adjust your plan based on feedback\n"
        "/agenda — Get detailed agenda for the next 2 days\n\n"
        "--- On-Trip ---\n"
        "/feedback — End-of-day check-in\n"
        "/costs — View full budget breakdown\n"
        "/costs today — Today's spending\n"
        "/costs <city> — Spending for a city\n"
        "/costs food — Spending by category\n"
        "/costs save — Destination-specific savings tips\n"
        "/costs convert <amount> — Quick currency conversion\n\n"
        "--- Meta ---\n"
        "/status — Trip planning progress\n"
        "/mytrips — Browse and switch between your trips\n"
        "/help — Show this help message\n\n"
        "Or just chat naturally — I'll understand what you need!"
    )

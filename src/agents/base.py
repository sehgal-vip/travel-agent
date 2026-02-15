"""Base agent class shared by all specialist agents."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.constants import AGENT_MAX_TOKENS, MEMORY_AGENTS
from src.config.settings import get_settings
from src.state import TripState

logger = logging.getLogger(__name__)


def get_destination_context(state: TripState) -> str:
    """Build a destination-context block injected into every agent's system prompt.

    Returns an empty string (falsy) when no destination country is set, allowing
    callers to use a simple ``if dest_context:`` guard.
    """
    dest = state.get("destination")
    if not dest or not dest.get("country"):
        return ""

    parts = [
        f"\n--- DESTINATION CONTEXT ---",
        f"Country: {dest.get('flag_emoji', '')} {dest.get('country', 'Unknown')}",
        f"Region: {dest.get('region', 'Unknown')}",
        f"Language: {dest.get('language', 'Unknown')}",
        f"Currency: {dest.get('currency_code', '???')} ({dest.get('currency_symbol', '?')})",
        f"Exchange rate: 1 USD = {dest.get('exchange_rate_to_usd', '?')} {dest.get('currency_code', '')}",
        f"Climate: {dest.get('climate_type', 'unknown')}",
        f"Payment norms: {dest.get('payment_norms', 'unknown')}",
        f"Tipping: {dest.get('tipping_culture', 'unknown')}",
    ]

    phrases = dest.get("useful_phrases")
    if phrases:
        parts.append("Key phrases: " + ", ".join(f'"{k}" = "{v}"' for k, v in list(phrases.items())[:5]))

    cities = state.get("cities", [])
    if cities:
        route = " → ".join(f"{c.get('name', '?')} ({c.get('days', '?')}d)" for c in cities)
        parts.append(f"Route: {route}")

    dates = state.get("dates")
    if dates and dates.get("start"):
        parts.append(f"Dates: {dates['start']} to {dates.get('end', '?')} ({dates.get('total_days', '?')} days)")

    travelers = state.get("travelers")
    if travelers:
        parts.append(f"Travelers: {travelers.get('count', '?')} ({travelers.get('type', '?')})")

    budget = state.get("budget")
    if budget:
        parts.append(f"Budget style: {budget.get('style', '?')}")

    interests = state.get("interests", [])
    if interests:
        parts.append(f"Interests: {', '.join(interests)}")

    parts.append("--- END DESTINATION CONTEXT ---\n")
    return "\n".join(parts)


def format_money(amount: float | None, symbol: str = "$", code: str = "USD") -> str:
    """Format a money amount, handling zero-decimal currencies (JPY, KRW, VND, etc.)."""
    if amount is None:
        return "N/A"

    zero_decimal = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "PYG", "RWF"}
    if code.upper() in zero_decimal:
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def add_to_conversation_history(state: TripState, role: str, content: str, agent: str | None = None) -> list:
    """Append a message to conversation history and return updated list."""
    history = list(state.get("conversation_history", []))
    history.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
        }
    )
    return history


# ─── Token Budgeting ────────────────────────────────────

_DEFAULT_INPUT_BUDGET = 160_000  # conservative for 200K context models


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate. Uses //3 to be safe with non-ASCII (CJK, emoji)."""
    return len(text) // 3


def _truncate_memory(memory: str, available_tokens: int) -> str:
    """Truncate memory to fit within token budget, preserving notes section."""
    if _estimate_tokens(memory) <= available_tokens:
        return memory
    logger.warning(
        "Memory exceeds token budget (%d > %d). Truncating.",
        _estimate_tokens(memory), available_tokens,
    )
    char_limit = available_tokens * 3  # reverse estimate
    from src.tools.trip_memory import NOTES_MARKER
    if NOTES_MARKER in memory:
        marker_pos = memory.index(NOTES_MARKER)
        notes = memory[marker_pos:]
        body_limit = char_limit - len(notes)
        if body_limit > 200:
            return memory[:body_limit] + "\n...[truncated]...\n\n" + notes
        return notes[:char_limit]
    return memory[:char_limit] + "\n...[truncated]..."


class BaseAgent:
    """Wraps ChatAnthropic with model selection and destination-context injection."""

    agent_name: str = "base"

    def __init__(self) -> None:
        settings = get_settings()
        model_id = settings.AGENT_MODELS.get(self.agent_name, settings.DEFAULT_MODEL)
        max_tokens = AGENT_MAX_TOKENS.get(self.agent_name, 4096)
        self.llm = ChatAnthropic(
            model=model_id,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=max_tokens,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    _TONE_PREAMBLE = (
        "You are a friendly, knowledgeable travel assistant. "
        "Be warm but concise. Use plain language.\n\n"
    )

    def _ensure_memory_fresh(self, state: TripState) -> str:
        """Build memory content for this agent's prompt. Does NOT write to disk.

        Only runs for agents in MEMORY_AGENTS. Others get thin destination context.
        The handler (handlers.py) is the sole writer — it calls build_and_write_agent_memory
        after the agent responds, ensuring the persisted file reflects post-response state.
        """
        if self.agent_name not in MEMORY_AGENTS:
            return ""

        from src.tools.trip_memory import build_agent_memory_content

        trip_id = state.get("trip_id")
        if not trip_id:
            return ""

        return build_agent_memory_content(trip_id, self.agent_name, state) or ""

    def build_system_prompt(self, state: TripState) -> str:
        """Combine tone preamble, agent-specific prompt, and per-agent memory (or destination context)."""
        base_prompt = self._TONE_PREAMBLE + self.get_system_prompt(state=state)

        # Specialist agents: refresh + inject their memory file
        memory = self._ensure_memory_fresh(state)
        if memory:
            # Apply token budget to prevent exceeding context window
            base_tokens = _estimate_tokens(base_prompt)
            output_budget = AGENT_MAX_TOKENS.get(self.agent_name, 4096)
            available = _DEFAULT_INPUT_BUDGET - base_tokens - output_budget
            memory = _truncate_memory(memory, available)
            return (
                f"{base_prompt}\n"
                f"--- {self.agent_name.upper()} MEMORY ---\n"
                f"{memory}\n"
                f"--- END {self.agent_name.upper()} MEMORY ---"
            )

        # Orchestrator/onboarding/librarian + pre-onboarding fallback: thin destination context
        dest_context = get_destination_context(state)
        result = f"{base_prompt}\n{dest_context}" if dest_context else base_prompt

        # Inject user profile if available
        user_profile = state.get("_user_profile") if state else None
        if user_profile and any(user_profile.get(k) for k in ("dietary", "pace", "budget_tendency", "interests_history", "visited_countries")):
            profile_parts = ["\n--- USER PROFILE (cross-trip) ---"]
            if user_profile.get("dietary"):
                profile_parts.append(f"Dietary: {', '.join(user_profile['dietary'])}")
            if user_profile.get("pace"):
                profile_parts.append(f"Pace preference: {user_profile['pace']}")
            if user_profile.get("budget_tendency"):
                profile_parts.append(f"Budget tendency: {user_profile['budget_tendency']}")
            if user_profile.get("interests_history"):
                profile_parts.append(f"Past interests: {', '.join(user_profile['interests_history'][:8])}")
            if user_profile.get("visited_countries"):
                profile_parts.append(f"Visited: {', '.join(user_profile['visited_countries'])}")
            if user_profile.get("mobility_notes"):
                profile_parts.append(f"Mobility: {', '.join(user_profile['mobility_notes'])}")
            profile_parts.append("--- END USER PROFILE ---")
            result += "\n".join(profile_parts)

        return result

    def get_system_prompt(self, state: TripState | None = None) -> str:
        """Override in subclasses to provide the agent-specific system prompt."""
        return "You are a helpful travel planning assistant."

    def bind_tools(self, tools: list) -> Any:
        """Bind LangChain tools to the LLM."""
        return self.llm.bind_tools(tools)

    def _compress_history(self, history: list, max_recent: int = 15) -> tuple[str, list]:
        """Compress older conversation history, keeping recent messages verbatim.

        Returns:
            (summary, recent_messages) — summary is empty string if no compression needed.
        """
        if len(history) <= max_recent:
            return "", history

        older = history[:-max_recent]
        recent = history[-max_recent:]

        # Build a simple extractive summary (no LLM call — keep it fast)
        summary_parts = ["## Conversation Summary (older messages)"]

        # Group by agent
        agent_topics: dict[str, list[str]] = {}
        for msg in older:
            agent = msg.get("agent") or msg.get("role", "user")
            content = msg.get("content", "")
            # Extract key info — first sentence or first 100 chars
            snippet = content.split(".")[0][:100] if content else ""
            if snippet:
                agent_topics.setdefault(agent, []).append(snippet)

        for agent, snippets in agent_topics.items():
            if agent == "user":
                summary_parts.append(f"- User discussed: {'; '.join(snippets[:5])}")
            else:
                summary_parts.append(f"- {agent} covered: {'; '.join(snippets[:3])}")

        summary = "\n".join(summary_parts)
        return summary, recent

    async def invoke(self, state: TripState, user_message: str) -> str:
        """Run a single LLM call with system prompt + conversation context + user message."""
        system_prompt = self.build_system_prompt(state)
        messages = [SystemMessage(content=system_prompt)]

        # Compressed conversation history
        history = state.get("conversation_history", [])
        summary, recent = self._compress_history(history)
        if summary:
            messages.append(HumanMessage(content=f"[Prior conversation summary]\n{summary}"))
        for msg in recent:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_message))

        response = await self.llm.ainvoke(messages)
        return response.content

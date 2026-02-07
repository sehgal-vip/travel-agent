"""Base agent class shared by all specialist agents."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config.settings import get_settings
from src.state import TripState

logger = logging.getLogger(__name__)


def get_destination_context(state: TripState) -> str:
    """Build a destination-context block injected into every agent's system prompt."""
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


class BaseAgent:
    """Wraps ChatAnthropic with model selection and destination-context injection."""

    agent_name: str = "base"

    def __init__(self) -> None:
        settings = get_settings()
        model_id = settings.AGENT_MODELS.get(self.agent_name, settings.DEFAULT_MODEL)
        self.llm = ChatAnthropic(
            model=model_id,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=4096,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    _TONE_PREAMBLE = (
        "You are a friendly, knowledgeable travel assistant. "
        "Be warm but concise. Use plain language.\n\n"
    )

    def build_system_prompt(self, state: TripState) -> str:
        """Combine tone preamble, agent-specific prompt, and destination context."""
        base_prompt = self._TONE_PREAMBLE + self.get_system_prompt()
        dest_context = get_destination_context(state)
        return f"{base_prompt}\n{dest_context}" if dest_context else base_prompt

    def get_system_prompt(self) -> str:
        """Override in subclasses to provide the agent-specific system prompt."""
        return "You are a helpful travel planning assistant."

    def bind_tools(self, tools: list) -> Any:
        """Bind LangChain tools to the LLM."""
        return self.llm.bind_tools(tools)

    async def invoke(self, state: TripState, user_message: str) -> str:
        """Run a single LLM call with system prompt + conversation context + user message."""
        system_prompt = self.build_system_prompt(state)
        messages = [SystemMessage(content=system_prompt)]

        # Add recent conversation history for context (last 20 messages)
        history = state.get("conversation_history", [])
        for msg in history[-20:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_message))

        response = await self.llm.ainvoke(messages)
        return response.content

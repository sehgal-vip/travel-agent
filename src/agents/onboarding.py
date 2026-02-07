"""Onboarding agent — collects trip parameters through conversational flow."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent, add_to_conversation_history
from src.state import TripState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Onboarding Agent for a travel planning assistant that works for ANY destination worldwide.

Your job is to collect trip details through natural conversation. You are warm, enthusiastic about travel, and knowledgeable about destinations globally.

INFORMATION TO COLLECT (in rough order):

1. **Destination** — Which country or region? If they're unsure, help them narrow it down.
   - If they name a country, confirm it
   - If they name a region ("Southeast Asia"), help them pick countries
   - If they name specific cities, infer the country

2. **Cities to visit** — Either they have a list, or you help build one.
   - If they have cities: validate they're in the same country/region, check logistics
   - If they don't: suggest popular routes based on trip duration and interests
   - For each city, note if it's confirmed or tentative

3. **Travel dates** — Start and end date, or at least month/season

4. **Days per city** — Help them allocate based on city size and interests
   - Large cities (capitals, major metros): suggest 3-4 days
   - Medium cities: suggest 2-3 days
   - Small towns / nature areas: suggest 1-2 days
   - Factor in travel days between cities

5. **Travel style & budget**
   - Budget style: backpacker / mid-range / luxury / mixed
   - Total budget (if they have one)
   - What they splurge on vs save on

6. **Traveler profile**
   - Solo / couple / friends / family
   - Number of travelers
   - Ages (if family)

7. **Interests & priorities**
   - Categories: food, culture, history, nature, adventure, nightlife, relaxation,
     photography, architecture, shopping, art, spiritual/religious, sports, wildlife
   - Ask them to pick top 3-5

8. **Dietary preferences or restrictions**

9. **Mobility/accessibility needs**

10. **Must-do items** — Anything already on their bucket list for this destination

11. **Deal-breakers** — Things they want to avoid

12. **Accommodation preferences** — hostels, boutique hotels, homestays, resorts, Airbnb

13. **Transport preferences** — public transit, rental car, private transfers, flights, trains

CONVERSATION STYLE:
- Ask 1-2 questions at a time, never more
- After they answer, acknowledge and add a small destination insight
- Use their answers to make the next question contextual
- Offer smart defaults based on the destination
- Be flexible: if they want to skip, move on

CURRENT STATE:
You will receive the current onboarding step number and any data already collected.
Track progress through step numbers:
  0 = Just started, ask about destination
  1 = Have destination, ask about cities
  2 = Have cities, ask about dates
  3 = Have dates, ask about day allocation
  4 = Have day allocation, ask about travelers
  5 = Have travelers, ask about budget
  6 = Have budget, ask about interests
  7 = Have interests, ask about dietary/accessibility
  8 = Have dietary, ask about must-dos and deal-breakers
  9 = Have must-dos, ask about accommodation/transport
  10 = Have accommodation, present summary for confirmation
  11 = User confirmed — set onboarding complete

When you have enough information, summarize the trip configuration and ask for confirmation.

AFTER CONFIRMATION:
Respond with a JSON block wrapped in ```json ... ``` containing the complete trip config.
The JSON must have these keys:
{
  "confirmed": true,
  "destination": {"country": "...", "country_code": "...", "region": "...", "flag_emoji": "...", "language": "...", "currency_code": "...", "currency_symbol": "..."},
  "cities": [{"name": "...", "country": "...", "days": N, "order": N}],
  "dates": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "total_days": N},
  "travelers": {"count": N, "type": "...", "dietary": [], "accessibility": []},
  "budget": {"style": "...", "total_estimate_usd": null, "splurge_on": [], "save_on": []},
  "interests": [...],
  "must_dos": [...],
  "deal_breakers": [...],
  "accommodation_pref": "...",
  "transport_pref": [...]
}

If the user has NOT yet confirmed, do NOT output JSON. Just continue the conversation naturally.
"""


class OnboardingAgent(BaseAgent):
    agent_name = "onboarding"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Process a user message during onboarding.

        Returns a dict with:
          - response: str (message to send back)
          - state_updates: dict (partial TripState updates, if any)
        """
        step = state.get("onboarding_step", 0)

        # Build context about what we already know
        context_parts = [f"Current onboarding step: {step}"]
        if state.get("destination", {}).get("country"):
            context_parts.append(f"Destination: {state['destination']['country']}")
        if state.get("cities"):
            context_parts.append(f"Cities: {', '.join(c.get('name', '') for c in state['cities'])}")
        if state.get("dates", {}).get("start"):
            context_parts.append(f"Dates: {state['dates']['start']} to {state['dates'].get('end', '?')}")
        if state.get("travelers", {}).get("type"):
            context_parts.append(f"Travelers: {state['travelers']}")
        if state.get("budget", {}).get("style"):
            context_parts.append(f"Budget: {state['budget']['style']}")
        if state.get("interests"):
            context_parts.append(f"Interests: {', '.join(state['interests'])}")

        context_str = "\n".join(context_parts)

        # Build messages for LLM
        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system)]

        # Add conversation history
        history = state.get("conversation_history", [])
        for msg in history[-20:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=f"[Context: {context_str}]\n\nUser message: {user_message}"))

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # Check if the response contains a confirmed JSON config
        state_updates: dict = {}
        if "```json" in response_text and '"confirmed": true' in response_text:
            config = self._extract_config(response_text)
            if config:
                state_updates = self._build_state_from_config(config, state)
                # Clean the response — remove JSON block for user display
                clean_response = response_text.split("```json")[0].strip()
                if not clean_response:
                    clean_response = "Your trip is all set! I'll kick off destination research now."
                response_text = clean_response

        # Advance step heuristically based on what we know
        if not state_updates:
            new_step = self._estimate_step(state, user_message, step)
            if new_step != step:
                state_updates["onboarding_step"] = new_step

        # Update conversation history
        updated_history = add_to_conversation_history(state, "user", user_message)
        updated_history = list(updated_history)
        updated_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "onboarding",
        })
        state_updates["conversation_history"] = updated_history
        state_updates["current_agent"] = "onboarding"

        return {"response": response_text, "state_updates": state_updates}

    def _extract_config(self, text: str) -> dict | None:
        """Extract JSON config from markdown code block."""
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to extract onboarding config JSON")
            return None

    def _build_state_from_config(self, config: dict, state: TripState) -> dict:
        """Build state updates from confirmed onboarding config."""
        trip_id = state.get("trip_id") or str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        updates: dict = {
            "trip_id": trip_id,
            "onboarding_complete": True,
            "onboarding_step": 11,
            "updated_at": now,
        }

        if "destination" in config:
            updates["destination"] = config["destination"]
        if "cities" in config:
            updates["cities"] = config["cities"]
        if "dates" in config:
            updates["dates"] = config["dates"]
        if "travelers" in config:
            updates["travelers"] = config["travelers"]
        if "budget" in config:
            updates["budget"] = config["budget"]
        if "interests" in config:
            updates["interests"] = config["interests"]
        if "must_dos" in config:
            updates["must_dos"] = config["must_dos"]
        if "deal_breakers" in config:
            updates["deal_breakers"] = config["deal_breakers"]
        if "accommodation_pref" in config:
            updates["accommodation_pref"] = config["accommodation_pref"]
        if "transport_pref" in config:
            updates["transport_pref"] = config["transport_pref"]

        # Initialise empty containers
        updates.setdefault("research", {})
        updates.setdefault("priorities", {})
        updates.setdefault("high_level_plan", [])
        updates.setdefault("plan_status", "not_started")
        updates.setdefault("plan_version", 0)
        updates.setdefault("detailed_agenda", [])
        updates.setdefault("feedback_log", [])
        updates.setdefault("cost_tracker", {
            "budget_total_usd": (config.get("budget") or {}).get("total_estimate_usd") or 0,
            "budget_daily_target_usd": 0,
            "local_currency": (config.get("destination") or {}).get("currency_code", "USD"),
            "currency_symbol": (config.get("destination") or {}).get("currency_symbol", "$"),
            "exchange_rate": 1.0,
            "pricing_benchmarks": {},
            "daily_log": [],
            "totals": {"spent_usd": 0, "remaining_usd": 0, "daily_avg_usd": 0, "projected_total_usd": 0, "status": "on_track"},
            "by_category": {},
            "by_city": {},
            "savings_tips": [],
        })
        updates.setdefault("library", {
            "workspace_path": None,
            "guide_written": False,
            "synced_cities": {},
            "feedback_days_written": [],
            "last_synced": None,
        })
        updates.setdefault("agent_scratch", {})

        return updates

    def _estimate_step(self, state: TripState, message: str, current_step: int) -> int:
        """Heuristic step advancement based on what state fields are populated."""
        if state.get("destination", {}).get("country") and current_step < 1:
            return 1
        if state.get("cities") and current_step < 2:
            return 2
        if state.get("dates", {}).get("start") and current_step < 3:
            return 3
        # For later steps, increment by 1 per message if we're still progressing
        if current_step < 10:
            return current_step + 1
        return current_step

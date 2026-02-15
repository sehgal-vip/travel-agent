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

SMART FLOW:
- If context shows fields as ALREADY COLLECTED, acknowledge them briefly and move on
- Focus your questions on the STILL NEEDED fields
- You may ask about 2 related missing fields at once to speed things up
- If only 1-2 fields remain, ask about them and then present the summary
- NEVER assume or fill in information the user hasn't provided
- NEVER skip asking about a field just because you can guess — always confirm with the user

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


STEP_FIELD_MAP: list[tuple[int, str, str]] = [
    # (step, state_key, human_label)
    (0, "destination", "Destination"),
    (1, "cities", "Cities"),
    (2, "dates", "Travel dates"),
    (3, "cities", "Days per city"),       # reuses cities — checks for 'days' key
    (4, "travelers", "Traveler profile"),
    (5, "budget", "Budget"),
    (6, "interests", "Interests"),
    (7, "travelers", "Dietary/accessibility"),  # checks dietary/accessibility keys
    (8, "must_dos", "Must-dos & deal-breakers"),
    (9, "accommodation_pref", "Accommodation"),
    (10, "transport_pref", "Transport"),
]


# Minimum viable state — only these 3 are required for onboarding to complete
MINIMUM_VIABLE_FIELDS = {"destination", "dates", "travelers"}


def _fallback_title(dest: dict, cities: list) -> str:
    """Deterministic fallback title when LLM title generation fails."""
    country = dest.get("country", "Adventure")
    if cities:
        city_names = " & ".join(c.get("name", "") for c in cities[:2])
        if len(cities) > 2:
            city_names += f" +{len(cities) - 2}"
        return f"{country}: {city_names}"
    return f"{country} Trip"


class OnboardingAgent(BaseAgent):
    agent_name = "onboarding"

    def get_system_prompt(self, state=None) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Process a user message during onboarding.

        Returns a dict with:
          - response: str (message to send back)
          - state_updates: dict (partial TripState updates, if any)
        """
        step = state.get("onboarding_step", 0)

        # Build context about what we already know and what's missing
        filled, missing = self._compute_filled_and_missing(state)
        context_parts = [f"Current onboarding step: {step} (Step {step + 1} of 11) — mention progress naturally"]

        if filled:
            context_parts.append(f"ALREADY COLLECTED: {', '.join(filled)}")
        # Add detail for collected fields
        if state.get("destination", {}).get("country"):
            context_parts.append(f"  Destination: {state['destination']['country']}")
        if state.get("cities"):
            context_parts.append(f"  Cities: {', '.join(c.get('name', '') for c in state['cities'])}")
        if state.get("dates", {}).get("start"):
            context_parts.append(f"  Dates: {state['dates']['start']} to {state['dates'].get('end', '?')}")
        if state.get("travelers", {}).get("type"):
            context_parts.append(f"  Travelers: {state['travelers']}")
        if state.get("budget", {}).get("style"):
            context_parts.append(f"  Budget: {state['budget']['style']}")
        if state.get("interests"):
            context_parts.append(f"  Interests: {', '.join(state['interests'])}")

        if missing:
            context_parts.append(f"STILL NEEDED: {', '.join(missing)}")

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
                title = await self._generate_trip_title(state_updates)
                state_updates["trip_title"] = title
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

        # Progressive profiling: check minimum viable state
        if not state.get("onboarding_complete") and not state_updates.get("onboarding_complete"):
            # Merge any partial state updates into a temporary state for checking
            check_state = {**state, **state_updates}
            if self._check_minimum_viable(check_state):
                depth = self._compute_onboarding_depth(check_state)
                if depth == "minimal":
                    # Don't force completion on minimal — let the LLM continue asking
                    # But if step >= 5 (past travelers), offer early exit
                    current_step = state_updates.get("onboarding_step", step)
                    if current_step >= 5:
                        state_updates["onboarding_complete"] = True
                        state_updates["onboarding_depth"] = depth
                        state_updates["onboarding_step"] = current_step
                elif depth in ("standard", "complete"):
                    state_updates["onboarding_complete"] = True
                    state_updates["onboarding_depth"] = depth

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
            "onboarding_depth": "complete",  # Full onboarding completed
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

    def _field_is_filled(self, state: TripState, step: int) -> bool:
        """Check whether the data for a given step is already present."""
        if step == 0:
            return bool(state.get("destination", {}).get("country"))
        if step == 1:
            return bool(state.get("cities"))
        if step == 2:
            return bool(state.get("dates", {}).get("start"))
        if step == 3:
            cities = state.get("cities", [])
            return bool(cities and all(c.get("days") for c in cities))
        if step == 4:
            return bool(state.get("travelers", {}).get("type"))
        if step == 5:
            return bool(state.get("budget", {}).get("style"))
        if step == 6:
            return bool(state.get("interests"))
        if step == 7:
            t = state.get("travelers", {})
            return "dietary" in t and "accessibility" in t
        if step == 8:
            return "must_dos" in state and state["must_dos"] is not None
        if step == 9:
            return bool(state.get("accommodation_pref"))
        if step == 10:
            return bool(state.get("transport_pref"))
        return False

    def _compute_filled_and_missing(self, state: TripState) -> tuple[list[str], list[str]]:
        """Return (filled_labels, missing_labels) based on STEP_FIELD_MAP."""
        filled, missing = [], []
        for step_num, _key, label in STEP_FIELD_MAP:
            if self._field_is_filled(state, step_num):
                filled.append(label)
            else:
                missing.append(label)
        return filled, missing

    def _check_minimum_viable(self, state: TripState) -> bool:
        """Check if minimum viable state is reached (destination + dates + travelers)."""
        has_dest = bool(state.get("destination", {}).get("country"))
        has_dates = bool(state.get("dates", {}).get("start"))
        has_travelers = bool(state.get("travelers", {}).get("type"))
        return has_dest and has_dates and has_travelers

    def _compute_onboarding_depth(self, state: TripState) -> str:
        """Compute onboarding depth based on how many fields are filled."""
        filled_count = sum(1 for step, _, _ in STEP_FIELD_MAP if self._field_is_filled(state, step))
        if filled_count >= 10:
            return "complete"
        if filled_count >= 6:
            return "standard"
        return "minimal"

    def _estimate_step(self, state: TripState, message: str, current_step: int) -> int:
        """Jump to the first unfilled step, rather than incrementing by 1."""
        for step_num, _key, _label in STEP_FIELD_MAP:
            if not self._field_is_filled(state, step_num):
                return max(step_num, current_step)
        return 10  # all filled → go to summary/confirmation

    async def _generate_trip_title(self, state_updates: dict) -> str:
        """Generate a short, witty title for the trip using LLM."""
        dest = state_updates.get("destination", {})
        cities = state_updates.get("cities", [])
        interests = state_updates.get("interests", [])
        travelers = state_updates.get("travelers", {})
        budget = state_updates.get("budget", {})
        city_names = ", ".join(c.get("name", "") for c in cities)

        prompt = (
            "Generate a single witty, fun trip title (max 6 words). "
            "Be creative and playful. No quotes. No emoji. Just the title.\n\n"
            f"Country: {dest.get('country', '?')}\n"
            f"Cities: {city_names}\n"
            f"Travelers: {travelers.get('count', '?')} ({travelers.get('type', '?')})\n"
            f"Style: {budget.get('style', '?')}\n"
            f"Interests: {', '.join(interests)}\n\n"
            "Examples: Ramen & Temples Run, Souks Spices & Sunsets, "
            "Two Foodies Loose in Tuscany, Cherry Blossoms on a Budget"
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            title = response.content.strip().strip('"').strip("'")
            return title[:60] if len(title) > 60 else title
        except Exception:
            logger.warning("Failed to generate trip title, using fallback")
            return _fallback_title(dest, cities)

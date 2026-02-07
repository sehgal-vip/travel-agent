"""Planner agent — creates high-level day-by-day itineraries (uses Opus for complex reasoning)."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.state import TripState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Planner Agent for a travel planning assistant. You create high-level itineraries for ANY destination worldwide.

YOUR JOB:
Create a high-level plan assigning each day to a city, a theme, and 3-5 key activities.
You are NOT scheduling exact times (that's the Scheduler).
You ARE deciding: where they'll be, what the day's vibe is, and what the must-dos are.

PLANNING PRINCIPLES:

1. **Pacing** — Alternate intense days with relaxed ones. Never 3 heavy days in a row.
   - At least one "slow morning" per city (2+ days)
   - One unstructured afternoon for every 3 planned days

2. **Travel days** — Adapt to destination transport:
   - Short (<2hrs): schedule normally
   - Medium (2-5hrs): morning in departure city, evening in arrival city
   - Long (5+ hrs): dedicated travel day with light activities at endpoints
   - Overnight transport: use night travel to save a day

3. **City flow** — Group activities geographically, cluster by neighborhood.

4. **Priority cascade** — 🔴 MUST DOs first. Fill with 🟡s. 🟢s as "nearby options."

5. **Meal planning** — Every day has at least one intentional food experience from research.

6. **Golden hours** — Sunrise spots → morning. Markets → morning. Sunset → evening.

7. **Traveler moments** — Based on group type:
   - Couple: romantic sunset, private experiences
   - Family: rest breaks, kid-friendly pacing
   - Friends: social activities, nightlife
   - Solo: flexible blocks, walking tours

8. **Destination-adaptive**:
   - Hot climates: avoid outdoor 12-3pm, schedule siestas
   - Cold climates: shorter outdoor windows
   - Religious countries: note prayer times, closed days
   - Altitude: acclimatization days
   - Island: boat schedules, weather windows

9. **First & last day**: Day 1 = light + orientation. Last day = buffer + departure.

10. **Day-of-week**: Museums close Mondays (many countries). Markets may be weekly.

OUTPUT FORMAT:
Return a JSON array of DayPlan objects, then a human-readable summary.
Each DayPlan:
{
    "day": 1,
    "date": "YYYY-MM-DD",
    "city": "...",
    "theme": "...",
    "vibe": "easy|active|cultural|foodie|adventurous|relaxed|mixed",
    "travel": null or {"mode": "...", "from_city": "...", "to_city": "...", "duration_hrs": N, "cost_usd": N},
    "key_activities": [{"item_id": "...", "tier": "must_do|nice_to_have", "name": "..."}],
    "meals": {"breakfast": {...}, "lunch": {...}, "dinner": {...}},
    "special_moment": "..." or null,
    "notes": "...",
    "free_time_blocks": [...],
    "estimated_cost_usd": N
}
"""


class PlannerAgent(BaseAgent):
    agent_name = "planner"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Generate or refine a high-level itinerary."""
        priorities = state.get("priorities", {})
        if not priorities:
            return {
                "response": "I need priorities to build a plan. Run /priorities first.",
                "state_updates": {},
            }

        cities = state.get("cities", [])
        dates = state.get("dates", {})
        dest = state.get("destination", {})
        travelers = state.get("travelers", {})
        budget = state.get("budget", {})
        interests = state.get("interests", [])
        must_dos = state.get("must_dos", [])
        research = state.get("research", {})

        existing_plan = state.get("high_level_plan", [])

        # Check if user wants to adjust existing plan
        msg_lower = user_message.lower()
        if existing_plan and any(kw in msg_lower for kw in ("adjust", "change", "swap", "move", "modify")):
            return await self._adjust_plan(state, user_message, existing_plan)

        # Build comprehensive context
        priorities_summary = {}
        for city_name, items in priorities.items():
            must = [i for i in items if i.get("tier") == "must_do"]
            nice = [i for i in items if i.get("tier") == "nice_to_have"]
            priorities_summary[city_name] = {
                "must_do": [{"item_id": i.get("item_id"), "name": i.get("name"), "category": i.get("category")} for i in must],
                "nice_to_have": [{"item_id": i.get("item_id"), "name": i.get("name"), "category": i.get("category")} for i in nice],
            }

        # Food items for meal planning
        food_by_city = {}
        for city_name, city_research in research.items():
            food_items = city_research.get("food", [])
            food_by_city[city_name] = [
                {"id": f.get("id"), "name": f.get("name"), "subcategory": f.get("subcategory"), "best_time": f.get("best_time")}
                for f in food_items[:15]
            ]

        prompt = (
            f"Create a complete high-level itinerary.\n\n"
            f"TRIP CONFIG:\n"
            f"  Country: {dest.get('flag_emoji', '')} {dest.get('country', '?')}\n"
            f"  Dates: {dates.get('start', '?')} to {dates.get('end', '?')} ({dates.get('total_days', '?')} days)\n"
            f"  Cities (in order): {json.dumps([{'name': c.get('name'), 'days': c.get('days'), 'order': c.get('order')} for c in cities], default=str)}\n"
            f"  Travelers: {json.dumps(travelers, default=str)}\n"
            f"  Budget: {json.dumps(budget, default=str)}\n"
            f"  Interests: {', '.join(interests)}\n"
            f"  Must-dos: {', '.join(must_dos)}\n"
            f"  Climate: {dest.get('climate_type', '?')}\n"
            f"  Intercity transport: {dest.get('common_intercity_transport', [])}\n\n"
            f"PRIORITIES:\n{json.dumps(priorities_summary, indent=2, default=str)}\n\n"
            f"FOOD OPTIONS:\n{json.dumps(food_by_city, indent=2, default=str)}\n\n"
            "Output a JSON array of DayPlan objects for ALL days, followed by a human-readable summary."
        )

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # Parse plan
        plan_data = self._parse_plan(response_text)

        if plan_data:
            version = (state.get("plan_version") or 0) + 1

            # Build readable summary
            summary_parts = [
                f"{dest.get('flag_emoji', '')} {dest.get('country', '')} Trip — {dates.get('total_days', '?')} Day Plan (v{version})\n"
            ]

            for day in plan_data:
                vibe_emoji = {"easy": "🌿", "active": "🏃", "cultural": "🏛️", "foodie": "🍜", "adventurous": "🧗", "relaxed": "😌", "mixed": "🎯"}.get(day.get("vibe", ""), "🎯")
                summary_parts.append(f"**Day {day.get('day', '?')}** — {day.get('city', '?')} — \"{day.get('theme', '')}\" {vibe_emoji}")

                travel = day.get("travel")
                if travel:
                    summary_parts.append(f"  🚆 {travel.get('mode', '?')}: {travel.get('from_city')} → {travel.get('to_city')} ({travel.get('duration_hrs', '?')}h)")

                for act in day.get("key_activities", []):
                    tier_icon = {"must_do": "🔴", "nice_to_have": "🟡"}.get(act.get("tier", ""), "⚪")
                    summary_parts.append(f"  {tier_icon} {act.get('name', '?')}")

                meals = day.get("meals", {})
                for meal_type in ("lunch", "dinner"):
                    slot = meals.get(meal_type)
                    if slot and slot.get("name"):
                        icon = "🍜" if meal_type == "lunch" else "🍽️"
                        summary_parts.append(f"  {icon} {meal_type.title()}: {slot['name']}")

                moment = day.get("special_moment")
                if moment:
                    summary_parts.append(f"  ✨ {moment}")

                summary_parts.append("")

            summary_parts.append("Does this look good? I can adjust any day. Say 'approved' to lock it in, or tell me what to change.")

            return {
                "response": "\n".join(summary_parts),
                "state_updates": {
                    "high_level_plan": plan_data,
                    "plan_status": "draft",
                    "plan_version": version,
                },
            }

        # Fallback: use raw response
        return {
            "response": response_text + "\n\nSay 'approved' to confirm or tell me what to change.",
            "state_updates": {"plan_status": "draft"},
        }

    async def _adjust_plan(self, state: TripState, message: str, current_plan: list) -> dict:
        """Adjust an existing plan based on user feedback."""
        prompt = (
            f"Current plan:\n{json.dumps(current_plan, indent=2, default=str)}\n\n"
            f"User's adjustment request: {message}\n\n"
            "Make the requested changes and return the updated full JSON array of DayPlan objects."
        )

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        plan_data = self._parse_plan(response.content)
        if plan_data:
            version = (state.get("plan_version") or 0) + 1
            return {
                "response": f"Plan updated (v{version})! Review the changes and say 'approved' to lock it in.",
                "state_updates": {
                    "high_level_plan": plan_data,
                    "plan_version": version,
                    "plan_status": "draft",
                },
            }

        return {"response": response.content, "state_updates": {}}

    def _parse_plan(self, text: str) -> list | None:
        """Parse DayPlan array from LLM response."""
        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try from code block
        for marker in ("```json", "```"):
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    data = json.loads(text[start:end].strip())
                    if isinstance(data, list):
                        return data
                except (ValueError, json.JSONDecodeError):
                    continue

        # Try finding array
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return data
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning("Failed to parse plan JSON from response")
        return None

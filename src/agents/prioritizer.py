"""Prioritizer agent — ranks research items into actionable priority tiers."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.state import TripState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Prioritizer Agent for a travel planning assistant. You rank researched items into priority tiers for ANY destination.

PRIORITY TIERS:
🔴 MUST DO (must_do) — Defining experiences. You'd regret skipping these.
🟡 NICE TO HAVE (nice_to_have) — Great experiences that enrich the trip but aren't essential.
🟢 IF NEARBY (if_nearby) — Worth doing only if you happen to be in the area.
⚪ SKIP (skip) — Tourist traps, overrated, or doesn't match this traveler's profile.

SCORING CRITERIA (weighted):
1. Uniqueness (30%) — Can you ONLY do this here?
2. Match to interests (25%) — How well does it align with their stated interests?
3. Reviews & reputation (15%) — Consistently recommended across sources?
4. Time-value ratio (15%) — Worth the time given limited days?
5. Traveler-type suitability (10%) — Good for their specific group type?
6. Logistics (5%) — Easy to access or requires significant effort?

RULES:
- 🔴 MUST DO: no more than 30% of items per city
- Every city needs at least 2-3 food items in MUST DO tier
- Tourist traps go in ⚪ SKIP with explanation
- Seasonal mismatches may downgrade items
- Food unique to this city/region → lean toward MUST DO
- Legendary specific restaurants/stalls → MUST DO

CROSS-CITY INTELLIGENCE:
- If two cities have similar experiences, prioritize where it's best
- Flag redundancy across cities

OUTPUT FORMAT:
Return a JSON object where keys are city names, and values are arrays of:
{
    "item_id": "...",
    "name": "...",
    "category": "...",
    "tier": "must_do|nice_to_have|if_nearby|skip",
    "score": 1-100,
    "reason": "1 sentence",
    "group": "optional grouping suggestion",
    "user_override": false
}

After the JSON, provide a brief human-readable summary.
"""


class PrioritizerAgent(BaseAgent):
    agent_name = "prioritizer"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Prioritize research items for one or all cities."""
        research = state.get("research", {})
        if not research:
            return {
                "response": "No research data yet. Run /research first to gather information about your cities.",
                "state_updates": {},
            }

        msg_lower = user_message.lower()
        cities = state.get("cities", [])
        existing_priorities = dict(state.get("priorities", {}))

        # Check for user override requests
        if any(keyword in msg_lower for keyword in ("move", "change", "upgrade", "downgrade", "skip")):
            return await self._handle_override(state, user_message, existing_priorities)

        # Determine which cities to prioritize
        target_cities = []
        for city in cities:
            name = city.get("name", "")
            if name.lower() in msg_lower or "all" in msg_lower or msg_lower.startswith("/priorities"):
                if name in research:
                    target_cities.append(name)

        if not target_cities:
            # Prioritize all researched cities
            target_cities = [c.get("name") for c in cities if c.get("name") in research]

        if not target_cities:
            return {
                "response": "No researched cities found to prioritize. Try /research first.",
                "state_updates": {},
            }

        # Build context for LLM
        interests = state.get("interests", [])
        traveler = state.get("travelers", {})
        budget = state.get("budget", {})

        research_summary = {}
        for city_name in target_cities:
            city_research = research.get(city_name, {})
            all_items = []
            for category in ("places", "activities", "food", "logistics", "tips", "hidden_gems"):
                all_items.extend(city_research.get(category, []))
            # Include essential fields for scoring
            research_summary[city_name] = [
                {
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "category": item.get("category", ""),
                    "description": item.get("description", "")[:200],
                    "cost_usd": item.get("cost_usd"),
                    "time_needed_hrs": item.get("time_needed_hrs"),
                    "tags": item.get("tags", []),
                    "traveler_suitability_score": item.get("traveler_suitability_score"),
                }
                for item in all_items
            ]

        # Allocate days per city for context
        city_days = {c.get("name"): c.get("days", 2) for c in cities}

        prompt = (
            f"Prioritize all research items for these cities: {', '.join(target_cities)}\n\n"
            f"Traveler profile: {json.dumps(traveler, default=str)}\n"
            f"Interests: {', '.join(interests)}\n"
            f"Budget style: {budget.get('style', 'midrange')}\n"
            f"Days per city: {json.dumps(city_days)}\n\n"
            f"Research items:\n{json.dumps(research_summary, indent=2, default=str)}\n\n"
            "Return a JSON object where keys are city names and values are arrays of prioritized items. "
            "Then add a brief summary for the user. Output JSON first, then summary."
        )

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # Parse priorities from response
        priorities_data = self._parse_priorities(response_text)

        if priorities_data:
            # Merge with existing priorities (preserving user overrides)
            for city_name, items in priorities_data.items():
                if city_name in existing_priorities:
                    # Preserve user overrides
                    overrides = {
                        p["item_id"]: p
                        for p in existing_priorities[city_name]
                        if p.get("user_override")
                    }
                    for item in items:
                        override = overrides.get(item.get("item_id"))
                        if override:
                            item["tier"] = override["tier"]
                            item["user_override"] = True
                existing_priorities[city_name] = items

            # Build summary
            summary_parts = ["🎯 Priority tiers assigned!\n"]
            for city_name in target_cities:
                items = existing_priorities.get(city_name, [])
                must_do = [i for i in items if i.get("tier") == "must_do"]
                nice = [i for i in items if i.get("tier") == "nice_to_have"]
                nearby = [i for i in items if i.get("tier") == "if_nearby"]
                skip = [i for i in items if i.get("tier") == "skip"]

                summary_parts.append(f"**{city_name}** ({len(items)} items)")
                summary_parts.append(f"  🔴 Must Do: {len(must_do)}")
                summary_parts.append(f"  🟡 Nice to Have: {len(nice)}")
                summary_parts.append(f"  🟢 If Nearby: {len(nearby)}")
                summary_parts.append(f"  ⚪ Skip: {len(skip)}")

                if must_do:
                    summary_parts.append("  Top must-dos:")
                    for item in must_do[:5]:
                        summary_parts.append(f"    🔴 {item.get('name', '?')} — {item.get('reason', '')}")
                summary_parts.append("")

            summary_parts.append("Want to adjust? Say 'move [item] to must-do' or 'skip [item]'.")
            summary_parts.append("Ready to plan? Try /plan")

            return {
                "response": "\n".join(summary_parts),
                "state_updates": {"priorities": existing_priorities},
            }

        # Fallback — extract summary from raw response
        return {
            "response": response_text,
            "state_updates": {},
        }

    async def _handle_override(self, state: TripState, message: str, priorities: dict) -> dict:
        """Handle user requests to move items between tiers."""
        prompt = (
            f"The user wants to adjust priorities: '{message}'\n\n"
            f"Current priorities: {json.dumps(priorities, indent=2, default=str)}\n\n"
            "Identify which item(s) to change and to which tier. "
            "Return a JSON object: {{\"changes\": [{{\"item_id\": \"...\", \"new_tier\": \"must_do|nice_to_have|if_nearby|skip\"}}]}}"
        )

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        try:
            changes_data = json.loads(response.content)
            changes = changes_data.get("changes", [])

            updated = dict(priorities)
            for change in changes:
                item_id = change.get("item_id", "")
                new_tier = change.get("new_tier", "")
                for city_items in updated.values():
                    for item in city_items:
                        if item.get("item_id") == item_id:
                            item["tier"] = new_tier
                            item["user_override"] = True

            return {
                "response": f"Updated {len(changes)} item(s). Use /priorities to see the updated list.",
                "state_updates": {"priorities": updated},
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "response": "I couldn't process that adjustment. Try: 'move [item name] to must-do'",
                "state_updates": {},
            }

    def _parse_priorities(self, text: str) -> dict | None:
        """Parse priority data from LLM response."""
        # Try to find JSON in the response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for marker in ("```json", "```"):
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    return json.loads(text[start:end].strip())
                except (ValueError, json.JSONDecodeError):
                    continue

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse priority JSON")
            return None

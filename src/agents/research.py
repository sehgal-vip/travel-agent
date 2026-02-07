"""Research agent — destination intelligence + city-level deep research."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.state import TripState
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Research Agent for a travel planning assistant that works for ANY destination worldwide.

You have TWO modes:

━━━ MODE 1: DESTINATION INTELLIGENCE ━━━
Run this ONCE when a new destination is set. Build a comprehensive country profile.

Research and compile:
- Official language(s) and 10-15 essential traveler phrases with pronunciation
- Currency, current exchange rate to USD, and typical payment norms (cash vs card)
- Tipping culture and expectations
- Visa requirements (based on traveler's nationality if known)
- Safety overview and common tourist scams
- Popular ride-hailing / transport apps
- Emergency numbers
- Electrical plug type and voltage
- Time zone
- Current season and weather patterns for the travel dates
- Cultural etiquette (dress codes, religious customs, social norms)
- Country pricing tier (budget/moderate/expensive/very_expensive)
- Daily budget benchmarks: backpacker / mid-range / luxury per person per day
- Common intercity transport options and booking platforms
- SIM card / connectivity situation
- Health advisories (vaccines, water safety, altitude, etc.)

━━━ MODE 2: CITY RESEARCH ━━━
Run for each city. Research across these dimensions:
1. Places to Visit — landmarks, temples/churches/mosques, viewpoints, neighborhoods, natural sites, museums, parks
2. Things to Do — activities, tours, experiences, workshops, day trips
3. Food & Drink — must-try local dishes, specific restaurants/stalls/markets, food streets
4. Logistics — internal transport, best areas to stay, neighborhood guide
5. Local Tips — best time of day for key spots, money-saving hacks
6. Hidden Gems — lesser-known spots, local favorites

FOR EACH ITEM, output a JSON object with these fields:
{
    "id": "{city_slug}-{category}-{number}",
    "name": "...",
    "name_local": "..." or null,
    "category": "place|activity|food|logistics|tip",
    "subcategory": "...",
    "description": "2-3 sentences",
    "cost_local": number or null,
    "cost_usd": number or null,
    "time_needed_hrs": number or null,
    "best_time": "morning|afternoon|evening|any" or null,
    "location": "address or area",
    "coordinates": {"lat": ..., "lng": ...} or null,
    "getting_there": "...",
    "traveler_suitability_score": 1-5,
    "advance_booking": true/false,
    "booking_lead_time": "..." or null,
    "tags": [...],
    "seasonal_relevance": "...",
    "sources": [...],
    "notes": "..." or null,
    "must_try_items": [...] or null
}

RESEARCH DEPTH:
- 4+ days in city: 20 places, 15 activities, 25 food items
- 2-3 days: 10 places, 8 activities, 15 food items
- 1 day: 5 places, 5 activities, 8 food items

Output your response as a JSON object. For Mode 1, use the DestinationIntel schema.
For Mode 2, wrap items in: {"places": [...], "activities": [...], "food": [...], "logistics": [...], "tips": [...], "hidden_gems": [...]}
"""


def _get_research_depth(days: int) -> dict[str, int]:
    if days >= 4:
        return {"places": 20, "activities": 15, "food": 25}
    elif days >= 2:
        return {"places": 10, "activities": 8, "food": 15}
    elif days == 1:
        return {"places": 5, "activities": 5, "food": 8}
    return {"places": 3, "activities": 2, "food": 5}


class ResearchAgent(BaseAgent):
    agent_name = "research"

    def __init__(self) -> None:
        super().__init__()
        self.search_tool = WebSearchTool()

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Route to destination intel or city research based on context."""
        dest = state.get("destination", {})
        country = dest.get("country", "")

        # Determine mode
        if not dest.get("researched_at"):
            # Mode 1: Destination Intelligence
            return await self._research_destination(state, country)

        # Mode 2: City Research
        msg_lower = user_message.lower()
        cities = state.get("cities", [])

        if "all" in msg_lower:
            return await self._research_all_cities(state)

        # Try to find a specific city mentioned
        target_city = None
        for city in cities:
            if city.get("name", "").lower() in msg_lower:
                target_city = city
                break

        if not target_city and cities:
            # Research the first unresearched city
            researched = state.get("research", {})
            for city in cities:
                if city.get("name") not in researched:
                    target_city = city
                    break
            if not target_city:
                target_city = cities[0]

        if target_city:
            return await self._research_city(state, target_city)

        return {
            "response": "Which city would you like me to research? You can say '/research all' to research all cities.",
            "state_updates": {},
        }

    async def _research_destination(self, state: TripState, country: str) -> dict:
        """Mode 1: Research destination-level intelligence."""
        dates = state.get("dates", {})
        travel_dates = f"{dates.get('start', '')} to {dates.get('end', '')}"
        travelers = state.get("travelers", {})
        nationalities = travelers.get("nationalities", [])

        # Web search for destination intel
        search_results = await self.search_tool.search_destination_intel(country, travel_dates)
        search_context = "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:500]}"
            for r in search_results[:15]
        )

        prompt = (
            f"Research destination intelligence for {country}.\n"
            f"Travel dates: {travel_dates}\n"
            f"Traveler nationalities: {', '.join(nationalities) if nationalities else 'not specified'}\n\n"
            f"Web search results:\n{search_context}\n\n"
            "Compile a complete DestinationIntel JSON object with ALL fields filled. "
            "Output ONLY valid JSON, no markdown."
        )

        system = self.build_system_prompt(state)
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]

        logger.info("LLM synthesis starting: destination intel for %s", country)
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        logger.info("LLM synthesis complete: destination intel for %s (%d chars)", country, len(response_text))

        # Parse the destination intel
        intel = self._parse_json(response_text)
        if intel:
            intel["researched_at"] = datetime.now(timezone.utc).isoformat()
            # Merge with existing destination data (keep onboarding fields)
            merged = {**state.get("destination", {}), **intel}

            summary = (
                f"🔍 Destination intelligence for {merged.get('flag_emoji', '')} {country} is ready!\n\n"
                f"Language: {merged.get('language', '?')}\n"
                f"Currency: {merged.get('currency_code', '?')} ({merged.get('currency_symbol', '?')})\n"
                f"Climate: {merged.get('climate_type', '?')}\n"
                f"Pricing: {merged.get('pricing_tier', '?')}\n\n"
                f"Would you like me to start researching cities? Say '/research all' or '/research [city name]'."
            )

            return {
                "response": summary,
                "state_updates": {"destination": merged},
            }

        return {
            "response": f"I've gathered information about {country}. Let me start researching your cities next.",
            "state_updates": {},
        }

    async def _research_city(self, state: TripState, city: dict) -> dict:
        """Mode 2: Research a single city."""
        city_name = city.get("name", "")
        country = city.get("country") or state.get("destination", {}).get("country", "")
        days = city.get("days", 2)
        interests = state.get("interests", [])
        traveler_type = state.get("travelers", {}).get("type", "")
        depth = _get_research_depth(days)

        # Web search
        search_results = await self.search_tool.search_city(
            city_name, country, interests, traveler_type
        )
        search_context = "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:400]}"
            for r in search_results[:18]
        )

        dest = state.get("destination", {})
        currency_code = dest.get("currency_code", "USD")
        exchange_rate = dest.get("exchange_rate_to_usd", 1)

        prompt = (
            f"Research {city_name}, {country} for a {traveler_type} trip of {days} days.\n"
            f"Interests: {', '.join(interests)}\n"
            f"Currency: {currency_code} (1 USD = {exchange_rate} {currency_code})\n"
            f"Target depth: {depth}\n\n"
            f"Web search results:\n{search_context}\n\n"
            "Return a JSON object with keys: places, activities, food, logistics, tips, hidden_gems. "
            "Each is an array of research items following the schema. "
            "Output ONLY valid JSON, no markdown."
        )

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

        logger.info("LLM synthesis starting: city research for %s (%d days)", city_name, days)
        response = await self.llm.ainvoke(messages)
        logger.info("LLM synthesis complete: city research for %s (%d chars)", city_name, len(response.content))
        research_data = self._parse_json(response.content)

        if research_data:
            research_data["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Merge with existing research (append, don't overwrite)
            existing = dict(state.get("research", {}))
            if city_name in existing:
                research_data = self._merge_research(existing[city_name], research_data)
            existing[city_name] = research_data

            # Count items
            total = sum(
                len(research_data.get(k, []))
                for k in ("places", "activities", "food", "logistics", "tips", "hidden_gems")
            )

            summary = (
                f"🔍 Research for {city_name} complete!\n\n"
                f"Found {total} items:\n"
                f"  📍 Places: {len(research_data.get('places', []))}\n"
                f"  🎯 Activities: {len(research_data.get('activities', []))}\n"
                f"  🍜 Food: {len(research_data.get('food', []))}\n"
                f"  🚌 Logistics: {len(research_data.get('logistics', []))}\n"
                f"  💡 Tips: {len(research_data.get('tips', []))}\n"
                f"  💎 Hidden gems: {len(research_data.get('hidden_gems', []))}\n\n"
                f"Ready to prioritize? Try /priorities"
            )

            return {"response": summary, "state_updates": {"research": existing}}

        # Fallback: parse failed, nothing was saved
        return {
            "response": f"I wasn't able to parse the research results for {city_name}. Try `/research {city_name}` again.",
            "state_updates": {},
        }

    async def _research_all_cities(self, state: TripState) -> dict:
        """Research all cities in sequence."""
        cities = state.get("cities", [])
        if not cities:
            return {"response": "No cities to research. Complete onboarding first with /start.", "state_updates": {}}

        # First, do destination intel if needed
        dest = state.get("destination", {})
        updates: dict = {}
        messages_parts: list[str] = []

        if not dest.get("researched_at"):
            try:
                result = await self._research_destination(state, dest.get("country", ""))
                messages_parts.append(result["response"])
                if result.get("state_updates", {}).get("destination"):
                    updates["destination"] = result["state_updates"]["destination"]
                    # Update state for subsequent city research calls
                    state = {**state, **updates}
            except Exception:
                logger.exception("Destination intel failed for %s", dest.get("country", ""))
                messages_parts.append("⚠️ Destination intelligence gathering failed — continuing with city research.")

        # Research each city
        for city in cities:
            city_name = city.get("name", "")
            if city_name in (state.get("research") or {}):
                messages_parts.append(f"✅ {city_name} — already researched")
                continue

            messages_parts.append(f"🔍 Researching {city_name}...")
            try:
                result = await self._research_city(state, city)
                if result.get("state_updates", {}).get("research"):
                    updates["research"] = result["state_updates"]["research"]
                    state = {**state, **updates}
                messages_parts.append(result["response"])
            except Exception:
                logger.exception("Research failed for city %s", city_name)
                messages_parts.append(f"⚠️ Research for {city_name} failed — skipping.")

        return {"response": "\n\n".join(messages_parts), "state_updates": updates}

    def _merge_research(self, existing: dict, new: dict) -> dict:
        """Merge new research into existing, deduplicating by name."""
        merged = dict(new)
        for key in ("places", "activities", "food", "logistics", "tips", "hidden_gems"):
            existing_items = existing.get(key, [])
            new_items = new.get(key, [])
            existing_names = {item.get("name", "").lower() for item in existing_items}
            for item in new_items:
                if item.get("name", "").lower() not in existing_names:
                    existing_items.append(item)
            merged[key] = existing_items
        return merged

    def _parse_json(self, text: str) -> dict | None:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        for marker in ("```json", "```"):
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    return json.loads(text[start:end].strip())
                except (ValueError, json.JSONDecodeError):
                    continue

        # Try finding first { to last }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse JSON from research response")
            return None

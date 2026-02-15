"""Research agent — destination intelligence + city-level deep research."""

from __future__ import annotations

import asyncio
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

    def get_system_prompt(self, state=None) -> str:
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

        # Check if resuming from clarification
        scratch = state.get("agent_scratch", {})
        if scratch.get("research_context"):
            # User responded to clarification, resume with enriched context
            target_city = scratch["research_context"].get("target_city")
            if target_city:
                # Find the city config
                for city in cities:
                    if city.get("name") == target_city:
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

        if target_city and self._needs_clarification(state):
            return {
                "response": (
                    f"Before I dig into {target_city.get('name', '')}, it'd help to know what you're most excited about! "
                    "Are you more into food, history, nightlife, nature, hidden local spots, or something else? "
                    "This helps me find the really good stuff, not just the tourist hits."
                ),
                "state_updates": {
                    "_awaiting_input": "research",
                    "agent_scratch": {
                        **(state.get("agent_scratch") or {}),
                        "research_context": {"target_city": target_city.get("name", "")},
                    },
                },
            }

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
        logger.info(
            "LLM synthesis complete: destination intel for %s (chars=%d, est_tokens=%d, max_tokens=%d)",
            country, len(response_text), len(response_text) // 3, self.llm.max_tokens,
        )

        # Parse the destination intel
        intel = self._parse_json(response_text)
        if intel:
            intel["researched_at"] = datetime.now(timezone.utc).isoformat()
            # Merge with existing destination data (keep onboarding fields)
            merged = {**state.get("destination", {}), **intel}

            # Generate a conversational abstract instead of just listing fields
            abstract_prompt = (
                f"You just gathered destination intelligence for {country}.\n\n"
                "Write a 3-4 line conversational summary of what travelers should know. "
                "Highlight the most interesting or surprising practical details. "
                "Be specific — mention currency, payment norms, key cultural notes. "
                "End by offering to research cities next.\n\n"
                "Tone: like a knowledgeable friend sharing what they found. Not a report."
            )
            abstract_messages = [
                SystemMessage(content=abstract_prompt),
                HumanMessage(content=json.dumps({
                    "country": country,
                    "language": merged.get("language"),
                    "currency": merged.get("currency_code"),
                    "exchange_rate": merged.get("exchange_rate_to_usd"),
                    "payment_norms": merged.get("payment_norms"),
                    "tipping": merged.get("tipping_culture"),
                    "climate": merged.get("climate_type"),
                    "season": merged.get("current_season_notes"),
                    "pricing_tier": merged.get("pricing_tier"),
                    "cultural_notes": merged.get("cultural_notes", [])[:3],
                })),
            ]
            abstract_response = await self.llm.ainvoke(abstract_messages)

            summary = (
                f"**{merged.get('flag_emoji', '')} {country}** — destination intel ready\n\n"
                f"{abstract_response.content}\n\n"
                f"_Ready to dive into your cities? Say '/research all' or '/research [city name]'._"
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

        # Check freshness of existing research
        freshness = self._check_freshness(state, city_name)
        if freshness and freshness["status"] == "very_stale":
            logger.info("Research for %s is very stale (%d days old), doing full refresh", city_name, freshness["age_days"])
        elif freshness and freshness["status"] == "stale":
            logger.info("Research for %s is stale (%d days old, trip in %s days), refreshing",
                        city_name, freshness["age_days"], freshness.get("days_until_trip"))

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

        system = self.build_system_prompt(state)

        # Try full-depth research first; on timeout, retry with reduced depth
        research_data = await self._llm_research_with_fallback(
            city_name, country, days, interests, traveler_type,
            depth, currency_code, exchange_rate, search_context, system,
        )

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

            # Generate a conversational abstract (non-critical — don't lose research on failure)
            abstract_text = await self._generate_abstract(city_name, traveler_type, days, interests, research_data, total)

            summary = (
                f"**{city_name}** — {total} items found\n\n"
                f"{abstract_text}\n\n"
                f"_Say 'tell me more about food' or 'what are the hidden gems?' to go deeper._"
            )

            return {"response": summary, "state_updates": {"research": existing}}

        # Fallback: parse failed, nothing was saved
        return {
            "response": f"I wasn't able to parse the research results for {city_name}. Try `/research {city_name}` again.",
            "state_updates": {},
        }

    # Retry schedule: each entry is (depth_divisor, extra_timeout_seconds).
    # Depth is halved each round; timeout grows exponentially via asyncio.sleep backoff.
    _RETRY_SCHEDULE = [
        (1, 0),    # attempt 0: full depth, no extra wait
        (2, 5),    # attempt 1: half depth, 5s backoff
        (4, 15),   # attempt 2: quarter depth, 15s backoff
    ]

    async def _llm_research_with_fallback(
        self, city_name: str, country: str, days: int,
        interests: list[str], traveler_type: str,
        depth: dict[str, int], currency_code: str,
        exchange_rate, search_context: str, system: str,
    ) -> dict | None:
        """Call LLM for city research JSON with exponential backoff and depth reduction.

        Retries up to len(_RETRY_SCHEDULE) times, halving the requested depth and
        adding an exponentially growing backoff between attempts.  Each individual
        LLM call still benefits from langchain-anthropic's own max_retries.
        """
        from anthropic import APITimeoutError

        last_err: Exception | None = None

        for attempt, (divisor, backoff_s) in enumerate(self._RETRY_SCHEDULE):
            current_depth = {k: max(v // divisor, 2) for k, v in depth.items()}
            prompt = self._build_research_prompt(
                city_name, country, days, interests, traveler_type,
                current_depth, currency_code, exchange_rate, search_context,
            )
            messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

            if attempt == 0:
                logger.info("LLM synthesis starting: city research for %s (%d days, depth=%s)", city_name, days, current_depth)
            else:
                logger.warning(
                    "LLM retry %d/%d for %s: depth=%s, backoff=%ds",
                    attempt, len(self._RETRY_SCHEDULE) - 1, city_name, current_depth, backoff_s,
                )
                await asyncio.sleep(backoff_s)

            try:
                response = await self.llm.ainvoke(messages)
                logger.info(
                    "LLM synthesis complete: city research for %s (chars=%d, est_tokens=%d, max_tokens=%d, attempt=%d)",
                    city_name, len(response.content), len(response.content) // 3, self.llm.max_tokens, attempt,
                )
                return self._parse_json(response.content)
            except APITimeoutError as exc:
                last_err = exc
                logger.warning("LLM timed out for %s (attempt %d/%d)", city_name, attempt, len(self._RETRY_SCHEDULE) - 1)

        # All retries exhausted
        raise last_err  # type: ignore[misc]

    @staticmethod
    def _build_research_prompt(
        city_name: str, country: str, days: int,
        interests: list[str], traveler_type: str,
        depth: dict[str, int], currency_code: str,
        exchange_rate, search_context: str,
    ) -> str:
        return (
            f"Research {city_name}, {country} for a {traveler_type} trip of {days} days.\n"
            f"Interests: {', '.join(interests)}\n"
            f"Currency: {currency_code} (1 USD = {exchange_rate} {currency_code})\n"
            f"Target depth: {depth}\n\n"
            f"Web search results:\n{search_context}\n\n"
            "Return a JSON object with keys: places, activities, food, logistics, tips, hidden_gems. "
            "Each is an array of research items following the schema. "
            "For each item, include confidence fields: "
            "source_recency (year string), corroborating_sources (int, how many results mention it), "
            "review_volume ('high'/'medium'/'low'/'unknown'), "
            "confidence_score (0.0-1.0 based on recency, corroboration, and review volume). "
            "Output ONLY valid JSON, no markdown."
        )

    async def _generate_abstract(
        self, city_name: str, traveler_type: str, days: int,
        interests: list[str], research_data: dict, total: int,
    ) -> str:
        """Generate a conversational abstract. Returns fallback text on failure."""
        try:
            abstract_prompt = (
                f"You just researched {city_name} for a {traveler_type} trip ({days} days).\n"
                f"Interests: {', '.join(interests)}\n\n"
                "Write a 3-4 line conversational abstract of what you found. "
                "Highlight what stands out — the best finds, surprises, or things that "
                "match their interests. Be specific (name 2-3 top items). "
                "End by offering to go deeper into any category.\n\n"
                "Tone: like a knowledgeable friend sharing what they found. Not a report."
            )
            abstract_messages = [
                SystemMessage(content=abstract_prompt),
                HumanMessage(content=json.dumps({
                    "total": total,
                    "places": [p.get("name") for p in research_data.get("places", [])[:5]],
                    "food": [f.get("name") for f in research_data.get("food", [])[:5]],
                    "hidden_gems": [g.get("name") for g in research_data.get("hidden_gems", [])[:3]],
                    "activities": [a.get("name") for a in research_data.get("activities", [])[:3]],
                })),
            ]
            abstract_response = await self.llm.ainvoke(abstract_messages)
            return abstract_response.content
        except Exception:
            logger.warning("Abstract generation failed for %s, using fallback summary", city_name)
            top_items = [
                p.get("name") for p in research_data.get("places", [])[:3]
            ] + [
                f.get("name") for f in research_data.get("food", [])[:2]
            ]
            highlights = ", ".join(item for item in top_items if item)
            return f"Found {total} items including {highlights}. Ask me about any category to dive deeper."

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

        messages_parts.append(
            "---\n"
            "Research complete! You can:\n"
            "- **/plan** to generate your day-by-day itinerary\n"
            "- **/priorities** to fine-tune what matters most (optional)\n"
            "- Ask me anything about what I found"
        )

        return {"response": "\n\n".join(messages_parts), "state_updates": updates}

    def _needs_clarification(self, state: TripState) -> bool:
        """Check if interests are too vague for targeted research."""
        interests = state.get("interests", [])
        if not interests:
            return True
        vague = {"sightseeing", "general", "everything", "all", "tourism", "tourist"}
        return all(i.lower() in vague for i in interests)

    def _check_freshness(self, state: TripState, city_name: str) -> dict | None:
        """Check if research for a city is stale and needs refresh.

        Returns a dict with freshness info if stale, None if fresh.
        """
        research = state.get("research", {})
        city_data = research.get(city_name)
        if not city_data:
            return None

        last_updated = city_data.get("last_updated")
        if not last_updated:
            return None

        try:
            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

        now = datetime.now(timezone.utc)
        age_days = (now - updated_dt).days

        # Check trip proximity
        dates = state.get("dates", {})
        trip_start = dates.get("start", "")
        days_until_trip = None
        if trip_start:
            try:
                trip_dt = datetime.fromisoformat(trip_start + "T00:00:00+00:00")
                days_until_trip = (trip_dt - now).days
            except ValueError:
                pass

        if age_days > 14 and (days_until_trip is not None and days_until_trip < 30):
            return {
                "age_days": age_days,
                "days_until_trip": days_until_trip,
                "status": "stale",
                "recommendation": "refresh",
            }

        if age_days > 30:
            return {
                "age_days": age_days,
                "days_until_trip": days_until_trip,
                "status": "very_stale",
                "recommendation": "full_refresh",
            }

        return None

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
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to parse JSON from research response (%d chars). "
                "Error: %s. Start: %.200s... End: ...%.200s",
                len(text), exc, text, text[-200:] if len(text) > 200 else text,
            )
            return None

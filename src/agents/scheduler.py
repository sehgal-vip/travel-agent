"""Scheduler agent — detailed 2-day rolling window agendas."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.state import TripState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Scheduler Agent for a travel planning assistant. You create detailed 2-day agendas for ANY destination.

YOUR JOB:
Convert the high-level plan into precise, hour-by-hour schedules. Every slot should answer:
what, where, when, how to get there, how much, and what to say in the local language.

SCHEDULING PRINCIPLES:

1. **Realistic timing** — Include actual travel time between activities.
   - Use destination_intel.transport_apps to recommend ride options
   - Always add 15-min buffer between activities

2. **Destination-adaptive energy curves** —
   TROPICAL/HOT: Early start (6-8am), main sightseeing (9-11:30), lunch+siesta (12-2:30pm), lighter afternoon, golden hour, evening activities.
   TEMPERATE: Breakfast 8-9, main sightseeing 9:30-12:30, lunch, afternoon activities, evening.
   COLD: Prioritize outdoor during daylight hours, indoor for dark hours.
   HIGH ALTITUDE: Lighter first 1-2 days, hydration reminders.

3. **Cost tracking** — Every slot has cost in BOTH local currency and USD.

4. **Booking alerts**: 🔴 Book NOW, 🟡 Book today, 🟢 Walk-in fine.

5. **Per slot**: Address, transport from previous, payment method, local phrase, rain backup.

6. **Local phrases** from destination_intel.useful_phrases.

7. **Confidence-aware timing**: If an item has low confidence_score (<0.5), add a note in the tips field like "Verify hours/availability before going." For high confidence items (>0.8), express certainty. Include this context in the tips for each slot.

OUTPUT FORMAT:
Return a JSON array of DetailedDay objects:
{
    "day": N,
    "date": "YYYY-MM-DD",
    "city": "...",
    "theme": "...",
    "slots": [
        {
            "time": "HH:MM",
            "end_time": "HH:MM",
            "type": "activity|meal|transport|rest|free|special",
            "name": "...",
            "item_id": "..." or null,
            "address": "...",
            "coordinates": {"lat": N, "lng": N} or null,
            "transport_from_prev": {"mode": "...", "line": "...", "duration_min": N, "cost_local": N, "cost_usd": N, "directions": "..."} or null,
            "duration_min": N,
            "cost_local": N or null,
            "cost_usd": N or null,
            "cost_for": "per person|total",
            "tips": "...",
            "local_phrase": "...",
            "payment": "cash_only|card_ok|mobile_pay",
            "booking_required": false,
            "booking_urgency": "book_now|book_today|walk_in" or null,
            "rain_backup": "..." or null,
            "tags": [...]
        }
    ],
    "daily_cost_estimate": {
        "food": N, "food_usd": N,
        "activities": N, "activities_usd": N,
        "transport": N, "transport_usd": N,
        "accommodation": N, "accommodation_usd": N,
        "other": N, "other_usd": N,
        "total_local": N, "total_usd": N,
        "currency_code": "..."
    },
    "booking_alerts": [...],
    "quick_reference": {
        "hotel": "...",
        "emergency": {"police": "...", "ambulance": "..."},
        "transport_app": "...",
        "exchange_rate": "...",
        "weather": "...",
        "key_phrases": {"thank you": "...", "how much": "...", "help": "..."},
        "reminders": [...]
    }
}

Then provide a formatted human-readable version.
"""


class SchedulerAgent(BaseAgent):
    agent_name = "scheduler"

    def get_system_prompt(self, state=None) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Build detailed 2-day agenda from the high-level plan."""
        plan = state.get("high_level_plan", [])
        if not plan:
            return {
                "response": "I need a high-level plan first. Try /plan to create one.",
                "state_updates": {},
            }

        dest = state.get("destination", {})
        research = state.get("research", {})
        feedback_log = state.get("feedback_log", [])
        current_day = state.get("current_trip_day") or 1

        # Determine which 2 days to schedule
        target_days = [d for d in plan if d.get("day") in (current_day, current_day + 1)]
        if not target_days:
            target_days = plan[:2]

        # Build rich context
        phrases = dest.get("useful_phrases", {})
        emergency = dest.get("emergency_numbers", {})
        climate = dest.get("climate_type", "temperate")
        currency_code = dest.get("currency_code", "USD")
        currency_symbol = dest.get("currency_symbol", "$")
        exchange_rate = dest.get("exchange_rate_to_usd", 1)

        # Get research details for relevant cities
        city_details = {}
        for day in target_days:
            city_name = day.get("city", "")
            if city_name in research and city_name not in city_details:
                city_research = research[city_name]
                all_items = []
                for cat in ("places", "activities", "food", "logistics"):
                    all_items.extend(city_research.get(cat, []))
                city_details[city_name] = [
                    {k: v for k, v in item.items() if k in (
                        "id", "name", "name_local", "category", "description", "cost_local",
                        "cost_usd", "time_needed_hrs", "best_time", "location", "coordinates",
                        "getting_there", "advance_booking", "tags", "notes", "must_try_items"
                    )}
                    for item in all_items
                ]

        # Include recent feedback for adaptive scheduling
        recent_feedback = feedback_log[-2:] if feedback_log else []

        # Drift detection from accumulated feedback
        drift_context = ""
        if feedback_log and len(feedback_log) >= 2:
            from src.tools.drift_detector import detect_drift, format_drift_for_prompt
            drift = detect_drift(feedback_log)
            drift_context = format_drift_for_prompt(drift)

        prompt = (
            f"Create detailed 2-day agendas for days {', '.join(str(d.get('day', '?')) for d in target_days)}.\n\n"
            f"HIGH-LEVEL PLAN:\n{json.dumps(target_days, indent=2, default=str)}\n\n"
            f"DESTINATION:\n"
            f"  Climate: {climate}\n"
            f"  Currency: {currency_code} ({currency_symbol}), rate: {exchange_rate} per USD\n"
            f"  Transport apps: {dest.get('transport_apps', [])}\n"
            f"  Payment: {dest.get('payment_norms', '?')}\n"
            f"  Key phrases: {json.dumps(dict(list(phrases.items())[:6]), default=str)}\n"
            f"  Emergency: {json.dumps(emergency, default=str)}\n\n"
            f"CITY RESEARCH DETAILS:\n{json.dumps(city_details, indent=2, default=str)}\n\n"
        )

        if recent_feedback:
            prompt += f"RECENT FEEDBACK (adapt schedule accordingly):\n{json.dumps(recent_feedback, indent=2, default=str)}\n\n"

        if drift_context:
            prompt += f"{drift_context}\n\n"

        prompt += "Return JSON array of DetailedDay objects, then a formatted human-readable agenda."

        system = self.build_system_prompt(state)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        response_text = response.content

        # Parse agenda
        agenda_data = self._parse_agenda(response_text)

        if agenda_data:
            # Merge into existing detailed_agenda
            existing = list(state.get("detailed_agenda", []))
            existing_days = {d.get("day") for d in existing}
            for new_day in agenda_data:
                day_num = new_day.get("day")
                if day_num in existing_days:
                    existing = [d for d in existing if d.get("day") != day_num]
                existing.append(new_day)
            existing.sort(key=lambda d: d.get("day", 0))

            # Build formatted output
            formatted = self._format_agenda(agenda_data, dest)

            return {
                "response": formatted,
                "state_updates": {"detailed_agenda": existing},
            }

        return {"response": response_text, "state_updates": {}}

    def _format_agenda(self, days: list, dest: dict) -> str:
        """Format agenda days for Telegram display."""
        from src.telegram.formatters import format_agenda_slot

        parts = []
        for day in days:
            parts.append(f"📅 DAY {day.get('day', '?')} — {day.get('date', '?')} — {day.get('city', '?')}")
            parts.append(f"\"{day.get('theme', '')}\"")
            parts.append("")

            for slot in day.get("slots", []):
                parts.append(format_agenda_slot(slot, dest))
                parts.append("")

            # Cost breakdown
            costs = day.get("daily_cost_estimate", {})
            if costs:
                code = costs.get("currency_code", dest.get("currency_code", "USD"))
                symbol = dest.get("currency_symbol", "$")
                parts.append("━━━━━━━━━━━━━━━━━━━━━━")
                parts.append(f"💰 DAY COST BREAKDOWN")
                for cat in ("food", "activities", "transport"):
                    local = costs.get(cat, 0)
                    usd = costs.get(f"{cat}_usd", 0)
                    if local or usd:
                        parts.append(f"   {cat.title()}: {symbol}{local:,.0f} (~${usd:,.0f})")
                total_local = costs.get("total_local", 0)
                total_usd = costs.get("total_usd", 0)
                parts.append(f"   TOTAL: {symbol}{total_local:,.0f} (~${total_usd:,.0f})")

            # Quick reference
            qr = day.get("quick_reference", {})
            if qr:
                parts.append("")
                parts.append("📋 QUICK REFERENCE")
                if qr.get("hotel"):
                    parts.append(f"   🏨 {qr['hotel']}")
                if qr.get("emergency"):
                    for service, num in qr["emergency"].items():
                        parts.append(f"   📞 {service}: {num}")
                if qr.get("transport_app"):
                    parts.append(f"   🚕 {qr['transport_app']}")
                if qr.get("exchange_rate"):
                    parts.append(f"   💱 {qr['exchange_rate']}")
                phrases = qr.get("key_phrases", {})
                for eng, local in list(phrases.items())[:3]:
                    parts.append(f"   💬 {eng} = {local}")
                for reminder in qr.get("reminders", [])[:3]:
                    parts.append(f"   ⚡ {reminder}")

            parts.append("")

        return "\n".join(parts)

    def _parse_agenda(self, text: str) -> list | None:
        """Parse DetailedDay array from LLM response."""
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

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

        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            data = json.loads(text[start:end])
            if isinstance(data, list):
                return data
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning("Failed to parse agenda JSON")
        return None

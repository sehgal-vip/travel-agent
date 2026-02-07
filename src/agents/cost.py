"""Cost agent — dynamic budget tracking for any destination and currency."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.state import TripState
from src.telegram.formatters import format_budget_report
from src.tools.currency import format_dual, format_local, format_usd

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Cost Agent for a travel planning assistant. You handle budgeting for ANY destination worldwide.

DYNAMIC PRICING APPROACH:
1. Read destination_intel.daily_budget_benchmarks for baseline pricing
2. Use research item costs for specific estimates
3. Adapt to local currency and exchange rates dynamically

CURRENCY HANDLING:
- Primary: local currency (from destination)
- Secondary: USD
- Always show both: "¥2,000 (~$13.50)"

COST CATEGORIES:
1. 🏨 Accommodation
2. 🍽️ Food & Drink
3. 🎟️ Activities & Entrance fees
4. 🚗 Transport (intercity + local)
5. 🛍️ Shopping
6. 💆 Wellness
7. 📱 Misc (SIM, tips, emergencies)

TIPPING INTELLIGENCE:
Read from destination_intel.tipping_culture and factor into estimates.

SUB-COMMANDS:
/costs → Full budget report
/costs today → Today's estimated vs actual
/costs {city} → City breakdown
/costs food → Category breakdown
/costs save → Destination-specific savings tips
/costs convert X → Quick currency conversion

When the user logs spending, respond conversationally and update the tracker.
When asked for a report, output structured data.

For savings tips, generate them dynamically based on the destination — never hardcode.
"""


class CostAgent(BaseAgent):
    agent_name = "cost"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Handle cost-related queries and spending logs."""
        msg_lower = user_message.lower().strip()
        dest = state.get("destination", {})
        tracker = dict(state.get("cost_tracker") or {})

        # Route sub-commands
        if msg_lower == "/costs" or msg_lower == "costs":
            return self._full_report(state)

        if msg_lower.startswith("/costs today"):
            return await self._today_report(state)

        if msg_lower.startswith("/costs save"):
            return await self._savings_tips(state)

        if msg_lower.startswith("/costs convert"):
            return await self._convert(state, user_message)

        # Check for a specific city or category
        cities = [c.get("name", "").lower() for c in state.get("cities", [])]
        categories = ["food", "accommodation", "activities", "transport", "shopping", "wellness", "misc"]

        for city in cities:
            if city in msg_lower:
                return self._city_report(state, city)

        for cat in categories:
            if cat in msg_lower:
                return self._category_report(state, cat)

        # Check if user is logging spending
        if any(kw in msg_lower for kw in ("spent", "paid", "cost me", "bought", "logged")):
            return await self._log_spending(state, user_message)

        # Default: show budget report with conversational wrapper
        return await self._conversational_cost_response(state, user_message)

    def _full_report(self, state: dict) -> dict:
        """Generate the full budget report."""
        report = format_budget_report(state)
        return {"response": report, "state_updates": {}}

    async def _today_report(self, state: dict) -> dict:
        """Show today's spending breakdown."""
        current_day = state.get("current_trip_day", 1)
        tracker = state.get("cost_tracker", {})
        daily_log = tracker.get("daily_log", [])

        today_log = None
        for log in daily_log:
            if log.get("day") == current_day:
                today_log = log
                break

        if not today_log:
            dest = state.get("destination", {})
            agenda = state.get("detailed_agenda", [])
            today_agenda = None
            for d in agenda:
                if d.get("day") == current_day:
                    today_agenda = d
                    break

            if today_agenda:
                costs = today_agenda.get("daily_cost_estimate", {})
                return {
                    "response": (
                        f"💰 Day {current_day} — Estimated costs\n\n"
                        f"🍽️ Food: ~${costs.get('food_usd', 0):,.0f}\n"
                        f"🎟️ Activities: ~${costs.get('activities_usd', 0):,.0f}\n"
                        f"🚗 Transport: ~${costs.get('transport_usd', 0):,.0f}\n"
                        f"Total: ~${costs.get('total_usd', 0):,.0f}\n\n"
                        "No actual spending logged yet. Tell me what you've spent!"
                    ),
                    "state_updates": {},
                }

            return {
                "response": f"No cost data for Day {current_day} yet. Log your spending by telling me what you paid!",
                "state_updates": {},
            }

        est = today_log.get("estimated_usd", 0)
        actual = today_log.get("actual_usd", 0)
        diff = actual - est if actual else 0
        diff_str = f"+${diff:,.0f}" if diff > 0 else f"-${abs(diff):,.0f}" if diff < 0 else "on target"

        return {
            "response": (
                f"💰 Day {current_day} — {today_log.get('city', '?')}\n"
                f"  Estimated: {format_usd(est)}\n"
                f"  Actual: {format_usd(actual) if actual else 'not logged'}\n"
                f"  Difference: {diff_str}\n\n"
                f"Notes: {today_log.get('notes', 'none')}"
            ),
            "state_updates": {},
        }

    async def _savings_tips(self, state: dict) -> dict:
        """Generate destination-specific savings tips via LLM."""
        dest = state.get("destination", {})
        budget = state.get("budget", {})
        tracker = state.get("cost_tracker", {})

        # Check if we already have tips
        existing_tips = tracker.get("savings_tips", [])
        if existing_tips:
            formatted = "\n".join(f"💡 {tip}" for tip in existing_tips)
            return {"response": f"💰 Savings Tips for {dest.get('flag_emoji', '')} {dest.get('country', '')}:\n\n{formatted}", "state_updates": {}}

        prompt = (
            f"Generate 5-7 specific, actionable money-saving tips for traveling in {dest.get('country', 'this destination')}.\n"
            f"Budget style: {budget.get('style', 'midrange')}\n"
            f"Currency: {dest.get('currency_code', 'USD')}\n"
            f"Payment norms: {dest.get('payment_norms', 'unknown')}\n"
            f"Transport apps: {dest.get('transport_apps', [])}\n\n"
            "Reference: local transport, where locals eat, accommodation hacks, activity discounts, tourist traps.\n"
            "Return ONLY a JSON array of strings."
        )

        system = self.build_system_prompt(state)
        response = await self.llm.ainvoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        tips = []
        try:
            tips = json.loads(response.content)
            if not isinstance(tips, list):
                tips = [response.content]
        except json.JSONDecodeError:
            tips = [line.strip("- •").strip() for line in response.content.split("\n") if line.strip()]

        # Store tips
        updated_tracker = dict(tracker)
        updated_tracker["savings_tips"] = tips

        formatted = "\n".join(f"💡 {tip}" for tip in tips)
        return {
            "response": f"💰 Savings Tips for {dest.get('flag_emoji', '')} {dest.get('country', '')}:\n\n{formatted}",
            "state_updates": {"cost_tracker": updated_tracker},
        }

    async def _convert(self, state: dict, message: str) -> dict:
        """Handle currency conversion requests."""
        dest = state.get("destination", {})
        rate = dest.get("exchange_rate_to_usd", 1)
        code = dest.get("currency_code", "USD")
        symbol = dest.get("currency_symbol", "$")

        # Try to extract a number from the message
        import re
        numbers = re.findall(r"[\d,]+\.?\d*", message)
        if not numbers:
            return {
                "response": f"💱 Current rate: 1 USD = {rate} {code}\n\nUsage: /costs convert 100",
                "state_updates": {},
            }

        amount = float(numbers[0].replace(",", ""))

        # Determine direction: if amount seems like USD (small) or local (large)
        if amount < rate:
            # Likely USD → local
            local_amount = amount * rate
            return {
                "response": f"💱 ${amount:,.2f} USD = {format_local(local_amount, symbol, code)}",
                "state_updates": {},
            }
        else:
            # Likely local → USD
            usd_amount = amount / rate if rate > 0 else 0
            return {
                "response": f"💱 {format_local(amount, symbol, code)} = ${usd_amount:,.2f} USD",
                "state_updates": {},
            }

    def _city_report(self, state: dict, city: str) -> dict:
        """Show spending for a specific city."""
        tracker = state.get("cost_tracker", {})
        by_city = tracker.get("by_city", {})

        # Case-insensitive lookup
        city_data = None
        for k, v in by_city.items():
            if k.lower() == city.lower():
                city_data = v
                city = k
                break

        if not city_data:
            return {
                "response": f"No spending data for {city} yet.",
                "state_updates": {},
            }

        return {
            "response": f"💰 {city} Spending:\n  Total: {format_usd(city_data.get('spent_usd', 0))}",
            "state_updates": {},
        }

    def _category_report(self, state: dict, category: str) -> dict:
        """Show spending for a specific category."""
        tracker = state.get("cost_tracker", {})
        by_category = tracker.get("by_category", {})
        cat_data = by_category.get(category, {})

        if not cat_data:
            return {
                "response": f"No spending data for {category} yet.",
                "state_updates": {},
            }

        return {
            "response": f"💰 {category.title()} Spending:\n  Total: {format_usd(cat_data.get('spent_usd', 0))}",
            "state_updates": {},
        }

    async def _log_spending(self, state: dict, message: str) -> dict:
        """Use LLM to parse natural-language spending logs."""
        dest = state.get("destination", {})
        current_day = state.get("current_trip_day", 1)

        prompt = (
            f"The user is logging spending: '{message}'\n"
            f"Current day: {current_day}\n"
            f"Currency: {dest.get('currency_code', 'USD')} ({dest.get('currency_symbol', '$')})\n"
            f"Exchange rate: {dest.get('exchange_rate_to_usd', 1)} per USD\n\n"
            "Parse this into JSON: {{\"amount_local\": N, \"amount_usd\": N, \"category\": \"food|activities|transport|shopping|wellness|misc\", \"description\": \"...\"}}\n"
            "Return ONLY the JSON."
        )

        system = self.build_system_prompt(state)
        response = await self.llm.ainvoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        try:
            spend_data = json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "response": "I couldn't parse that spending. Try: 'spent $20 on lunch' or 'paid ¥3000 for temple entrance'",
                "state_updates": {},
            }

        # Update tracker
        tracker = dict(state.get("cost_tracker") or {})
        totals = dict(tracker.get("totals") or {})
        amount_usd = spend_data.get("amount_usd", 0)

        totals["spent_usd"] = totals.get("spent_usd", 0) + amount_usd
        budget = tracker.get("budget_total_usd", 0)
        totals["remaining_usd"] = budget - totals["spent_usd"]

        # Update by_category
        by_cat = dict(tracker.get("by_category") or {})
        cat = spend_data.get("category", "misc")
        cat_data = dict(by_cat.get(cat) or {})
        cat_data["spent_usd"] = cat_data.get("spent_usd", 0) + amount_usd
        by_cat[cat] = cat_data

        tracker["totals"] = totals
        tracker["by_category"] = by_cat

        symbol = dest.get("currency_symbol", "$")
        code = dest.get("currency_code", "USD")
        amount_local = spend_data.get("amount_local", 0)

        return {
            "response": (
                f"✅ Logged: {format_dual(amount_local, symbol, code, dest.get('exchange_rate_to_usd', 1))} "
                f"for {spend_data.get('description', cat)}\n"
                f"💰 Total spent: {format_usd(totals['spent_usd'])} / {format_usd(budget)}"
            ),
            "state_updates": {"cost_tracker": tracker},
        }

    async def _conversational_cost_response(self, state: dict, message: str) -> dict:
        """Handle general cost questions with LLM."""
        system = self.build_system_prompt(state)
        tracker_summary = json.dumps(state.get("cost_tracker", {}), indent=2, default=str)[:3000]

        prompt = (
            f"User question about costs: {message}\n\n"
            f"Current cost tracker:\n{tracker_summary}\n\n"
            "Answer helpfully with specific numbers. Always show both local currency and USD."
        )

        response = await self.llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])

        return {"response": response.content, "state_updates": {}}

"""Markdown file-based knowledge base — replaces Notion with a local folder of .md files."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.tools.currency import format_local

logger = logging.getLogger(__name__)


class MarkdownLibrary:
    """Creates and manages a folder of markdown files for a trip's knowledge base.

    Structure:
        library/
        ├── INDEX.md                     # Master index with links to all pages
        ├── destination-guide.md         # Country-level intel
        ├── itinerary.md                 # High-level plan
        ├── budget.md                    # Budget tracker
        ├── cities/
        │   ├── {city-slug}/
        │   │   ├── overview.md
        │   │   ├── places.md
        │   │   ├── food.md
        │   │   ├── activities.md
        │   │   ├── logistics.md
        │   │   └── hidden-gems.md
        │   └── ...
        └── feedback/
            └── day-{N}.md
    """

    def __init__(self, base_path: str = "./data/library") -> None:
        self.base_path = base_path

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _write(self, filepath: str, content: str) -> None:
        self._ensure_dir(os.path.dirname(filepath))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Wrote %s", filepath)

    def _slug(self, name: str) -> str:
        return name.lower().replace(" ", "-").replace("'", "")

    # ─── Workspace ────────────────────────────────

    def create_workspace(
        self,
        country: str,
        flag_emoji: str,
        year: str,
    ) -> str:
        """Create the trip library folder. Returns the library path."""
        folder_name = f"{self._slug(country)}-{year}"
        workspace_path = os.path.join(self.base_path, folder_name)
        self._ensure_dir(workspace_path)
        self._ensure_dir(os.path.join(workspace_path, "cities"))
        self._ensure_dir(os.path.join(workspace_path, "feedback"))
        logger.info("Created workspace at %s", workspace_path)
        return workspace_path

    # ─── Index ────────────────────────────────────

    def write_index(
        self,
        workspace_path: str,
        country: str,
        flag_emoji: str,
        cities: list[dict],
        dates: dict,
        has_guide: bool = False,
        has_priorities: bool = False,
        has_plan: bool = False,
        has_budget: bool = False,
    ) -> None:
        """Write or update the master INDEX.md."""
        total_days = dates.get("total_days", "?")
        start = dates.get("start", "?")
        end = dates.get("end", "?")

        lines = [
            f"# {flag_emoji} {country} Trip",
            f"",
            f"**Dates:** {start} to {end} ({total_days} days)",
            f"",
            f"---",
            f"",
            f"## Contents",
            f"",
        ]

        if has_guide:
            lines.append(f"- [Destination Guide](destination-guide.md)")

        for city in cities:
            name = city.get("name", "?")
            days = city.get("days", "?")
            slug = self._slug(name)
            lines.append(f"- [{name} ({days} days)](cities/{slug}/overview.md)")
            lines.append(f"  - [Places](cities/{slug}/places.md)")
            lines.append(f"  - [Food & Drink](cities/{slug}/food.md)")
            lines.append(f"  - [Activities](cities/{slug}/activities.md)")
            lines.append(f"  - [Logistics](cities/{slug}/logistics.md)")
            lines.append(f"  - [Hidden Gems](cities/{slug}/hidden-gems.md)")

        lines.append("")

        if has_priorities:
            lines.append("- [Priorities](priorities.md)")
        if has_plan:
            lines.append("- [Itinerary](itinerary.md)")
        if has_budget:
            lines.append("- [Budget](budget.md)")

        lines.extend([
            "",
            "---",
            f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        ])

        self._write(os.path.join(workspace_path, "INDEX.md"), "\n".join(lines))

    # ─── Destination Guide ────────────────────────

    def write_destination_guide(
        self,
        workspace_path: str,
        dest: dict,
    ) -> None:
        """Write the destination guide markdown file."""
        country = dest.get("country", "Unknown")
        flag = dest.get("flag_emoji", "")

        lines = [
            f"# {flag} {country} — Destination Guide",
            "",
            "---",
            "",
            "## Essential Phrases",
            "",
            "| English | Local |",
            "|---------|-------|",
        ]

        phrases = dest.get("useful_phrases", {})
        for eng, local in phrases.items():
            lines.append(f"| {eng} | {local} |")

        lines.extend([
            "",
            "## Currency & Money",
            "",
            f"- **Currency:** {dest.get('currency_code', '?')} ({dest.get('currency_symbol', '?')})",
            f"- **Exchange rate:** 1 USD = {dest.get('exchange_rate_to_usd', '?')} {dest.get('currency_code', '?')}",
            f"- **Payment norms:** {dest.get('payment_norms', '?')}",
            f"- **Tipping:** {dest.get('tipping_culture', '?')}",
            "",
            "## Safety",
            "",
        ])
        for note in dest.get("safety_notes", []):
            lines.append(f"- {note}")

        scams = dest.get("common_scams", [])
        if scams:
            lines.extend(["", "### Common Scams", ""])
            for scam in scams:
                lines.append(f"- {scam}")

        lines.extend(["", "## Emergency Numbers", ""])
        for service, number in dest.get("emergency_numbers", {}).items():
            lines.append(f"- **{service}:** {number}")

        lines.extend(["", "## Cultural Notes", ""])
        for note in dest.get("cultural_notes", []):
            lines.append(f"- {note}")

        lines.extend([
            "",
            "## Transport",
            "",
            f"- **Apps:** {', '.join(dest.get('transport_apps', []))}",
            f"- **Intercity:** {', '.join(dest.get('common_intercity_transport', []))}",
            f"- **Booking platforms:** {', '.join(dest.get('booking_platforms', []))}",
            "",
            "## Practical Info",
            "",
            f"- **Plug type:** {dest.get('plug_type', '?')}",
            f"- **Voltage:** {dest.get('voltage', '?')}",
            f"- **Time zone:** {dest.get('time_zone', '?')}",
            f"- **SIM/Connectivity:** {dest.get('sim_connectivity', '?')}",
            f"- **Climate:** {dest.get('climate_type', '?')}",
            f"- **Season notes:** {dest.get('current_season_notes', '?')}",
        ])

        health = dest.get("health_advisories", [])
        if health:
            lines.extend(["", "## Health Advisories", ""])
            for h in health:
                lines.append(f"- {h}")

        benchmarks = dest.get("daily_budget_benchmarks", {})
        if benchmarks:
            lines.extend([
                "",
                "## Daily Budget Benchmarks (per person, USD)",
                "",
                "| Style | Amount |",
                "|-------|--------|",
            ])
            for style, amount in benchmarks.items():
                lines.append(f"| {style} | ${amount} |")

        self._write(os.path.join(workspace_path, "destination-guide.md"), "\n".join(lines))

    # ─── City Research ────────────────────────────

    def write_city_research(
        self,
        workspace_path: str,
        city_name: str,
        city_research: dict,
        currency_symbol: str,
        currency_code: str,
        exchange_rate: float,
        priorities: list[dict] | None = None,
    ) -> int:
        """Write all research files for a city. Returns number of items written."""
        slug = self._slug(city_name)
        city_dir = os.path.join(workspace_path, "cities", slug)
        self._ensure_dir(city_dir)

        # Build priority lookup
        priority_map: dict[str, str] = {}
        if priorities:
            tier_emoji = {"must_do": "🔴", "nice_to_have": "🟡", "if_nearby": "🟢", "skip": "⚪"}
            for p in priorities:
                priority_map[p.get("item_id", "")] = tier_emoji.get(p.get("tier", ""), "")

        total = 0

        # Overview
        total_items = sum(len(city_research.get(k, [])) for k in ("places", "activities", "food", "logistics", "tips", "hidden_gems"))
        overview = [
            f"# {city_name}",
            "",
            f"**Total research items:** {total_items}",
            f"**Last updated:** {city_research.get('last_updated', '?')}",
            "",
            f"- [Places](places.md) ({len(city_research.get('places', []))})",
            f"- [Food & Drink](food.md) ({len(city_research.get('food', []))})",
            f"- [Activities](activities.md) ({len(city_research.get('activities', []))})",
            f"- [Logistics](logistics.md) ({len(city_research.get('logistics', []))})",
            f"- [Hidden Gems](hidden-gems.md) ({len(city_research.get('hidden_gems', []))})",
        ]
        self._write(os.path.join(city_dir, "overview.md"), "\n".join(overview))

        # Category files
        category_files = {
            "places": "places.md",
            "food": "food.md",
            "activities": "activities.md",
            "logistics": "logistics.md",
            "tips": "logistics.md",  # tips merged into logistics
            "hidden_gems": "hidden-gems.md",
        }

        # Group tips into logistics
        merged: dict[str, list[dict]] = {}
        for cat, filename in category_files.items():
            items = city_research.get(cat, [])
            merged.setdefault(filename, []).extend(items)

        category_titles = {
            "places.md": "Places to Visit",
            "food.md": "Food & Drink",
            "activities.md": "Activities",
            "logistics.md": "Logistics & Tips",
            "hidden-gems.md": "Hidden Gems",
        }

        for filename, items in merged.items():
            if not items:
                continue
            title = category_titles.get(filename, filename)
            lines = [f"# {city_name} — {title}", ""]

            for item in items:
                total += 1
                item_id = item.get("id", "")
                priority = priority_map.get(item_id, "")
                name = item.get("name", "Unknown")
                name_local = item.get("name_local")

                header = f"## {priority} {name}".strip()
                if name_local:
                    header += f" ({name_local})"
                lines.append(header)
                lines.append("")

                desc = item.get("description", "")
                if desc:
                    lines.append(desc)
                    lines.append("")

                # Details table
                details: list[str] = []
                if item.get("cost_local") is not None:
                    local_str = format_local(item["cost_local"], currency_symbol, currency_code)
                    usd = item.get("cost_usd")
                    cost_str = f"{local_str}" + (f" (~${usd:,.2f})" if usd else "")
                    details.append(f"**Cost:** {cost_str}")
                elif item.get("cost_usd") is not None:
                    details.append(f"**Cost:** ${item['cost_usd']:,.2f}")
                if item.get("time_needed_hrs"):
                    details.append(f"**Time:** {item['time_needed_hrs']}h")
                if item.get("best_time"):
                    details.append(f"**Best time:** {item['best_time']}")
                if item.get("location"):
                    details.append(f"**Location:** {item['location']}")
                if item.get("getting_there"):
                    details.append(f"**Getting there:** {item['getting_there']}")
                if item.get("advance_booking"):
                    lead = item.get("booking_lead_time", "advance")
                    details.append(f"**Booking:** Required ({lead})")

                if details:
                    lines.extend(details)
                    lines.append("")

                tags = item.get("tags", [])
                if tags:
                    lines.append(f"*Tags: {', '.join(tags)}*")
                    lines.append("")

                notes = item.get("notes")
                if notes:
                    lines.append(f"> {notes}")
                    lines.append("")

                must_try = item.get("must_try_items", [])
                if must_try:
                    lines.append(f"**Must try:** {', '.join(must_try)}")
                    lines.append("")

                lines.append("---")
                lines.append("")

            self._write(os.path.join(city_dir, filename), "\n".join(lines))

        return total

    # ─── Priorities ───────────────────────────────

    def write_priorities(
        self,
        workspace_path: str,
        priorities: dict[str, list[dict]],
    ) -> None:
        """Write priorities.md with all cities and tiers."""
        lines = ["# Priorities", ""]
        tier_emoji = {"must_do": "🔴 Must Do", "nice_to_have": "🟡 Nice to Have", "if_nearby": "🟢 If Nearby", "skip": "⚪ Skip"}

        for city_name, items in priorities.items():
            lines.append(f"## {city_name}")
            lines.append("")

            by_tier: dict[str, list[dict]] = {}
            for item in items:
                tier = item.get("tier", "skip")
                by_tier.setdefault(tier, []).append(item)

            for tier_key in ("must_do", "nice_to_have", "if_nearby", "skip"):
                tier_items = by_tier.get(tier_key, [])
                if not tier_items:
                    continue
                lines.append(f"### {tier_emoji.get(tier_key, tier_key)} ({len(tier_items)})")
                lines.append("")
                for item in tier_items:
                    override = " *(user override)*" if item.get("user_override") else ""
                    lines.append(f"- **{item.get('name', '?')}** [{item.get('category', '?')}] — {item.get('reason', '')}{override}")
                lines.append("")

        self._write(os.path.join(workspace_path, "priorities.md"), "\n".join(lines))

    # ─── Itinerary ────────────────────────────────

    def write_itinerary(
        self,
        workspace_path: str,
        plan: list[dict],
        dest: dict,
    ) -> None:
        """Write itinerary.md from the high-level plan."""
        flag = dest.get("flag_emoji", "")
        country = dest.get("country", "?")

        lines = [f"# {flag} {country} — Itinerary", ""]

        for day in plan:
            vibe_emoji = {"easy": "🌿", "active": "🏃", "cultural": "🏛️", "foodie": "🍜", "adventurous": "🧗", "relaxed": "😌", "mixed": "🎯"}.get(day.get("vibe", ""), "🎯")
            lines.append(f"## Day {day.get('day', '?')} — {day.get('city', '?')} — \"{day.get('theme', '')}\" {vibe_emoji}")
            lines.append(f"*{day.get('date', '?')}*")
            lines.append("")

            travel = day.get("travel")
            if travel:
                lines.append(f"🚆 **Travel:** {travel.get('mode', '?')} from {travel.get('from_city', '?')} to {travel.get('to_city', '?')} ({travel.get('duration_hrs', '?')}h)")
                lines.append("")

            activities = day.get("key_activities", [])
            if activities:
                lines.append("**Activities:**")
                for act in activities:
                    tier_icon = {"must_do": "🔴", "nice_to_have": "🟡"}.get(act.get("tier", ""), "⚪")
                    lines.append(f"- {tier_icon} {act.get('name', '?')}")
                lines.append("")

            meals = day.get("meals", {})
            if meals:
                for meal_type in ("breakfast", "lunch", "dinner"):
                    slot = meals.get(meal_type)
                    if slot:
                        name = slot.get("name") or slot.get("type", "explore")
                        icon = {"breakfast": "☕", "lunch": "🍜", "dinner": "🍽️"}.get(meal_type, "🍽️")
                        lines.append(f"- {icon} **{meal_type.title()}:** {name}")
                lines.append("")

            moment = day.get("special_moment")
            if moment:
                lines.append(f"✨ **Special moment:** {moment}")
                lines.append("")

            notes = day.get("notes")
            if notes:
                lines.append(f"> {notes}")
                lines.append("")

            lines.append("---")
            lines.append("")

        self._write(os.path.join(workspace_path, "itinerary.md"), "\n".join(lines))

    # ─── Feedback ─────────────────────────────────

    def write_feedback(
        self,
        workspace_path: str,
        feedback: dict,
    ) -> None:
        """Write a single day's feedback to feedback/day-N.md."""
        day = feedback.get("day", 0)
        city = feedback.get("city", "?")
        date = feedback.get("date", "?")

        lines = [
            f"# Day {day} Feedback — {city}",
            f"*{date}*",
            "",
        ]

        if feedback.get("highlight"):
            lines.append(f"**Highlight:** {feedback['highlight']}")
        if feedback.get("lowlight"):
            lines.append(f"**Lowlight:** {feedback['lowlight']}")
        lines.append(f"**Energy:** {feedback.get('energy_level', '?')}")
        lines.append(f"**Food:** {feedback.get('food_rating', '?')}")
        lines.append(f"**Budget:** {feedback.get('budget_status', '?')}")
        lines.append(f"**Weather:** {feedback.get('weather', '?')}")
        lines.append(f"**Sentiment:** {feedback.get('sentiment', '?')}")
        lines.append("")

        completed = feedback.get("completed_items", [])
        if completed:
            lines.append("**Completed:**")
            for item in completed:
                lines.append(f"- ✅ {item}")
            lines.append("")

        skipped = feedback.get("skipped_items", [])
        if skipped:
            lines.append("**Skipped:**")
            for item in skipped:
                lines.append(f"- ⏭️ {item}")
            lines.append("")

        discoveries = feedback.get("discoveries", [])
        if discoveries:
            lines.append("**Discoveries:**")
            for d in discoveries:
                lines.append(f"- ✨ {d}")
            lines.append("")

        adjustments = feedback.get("adjustments_made", [])
        if adjustments:
            lines.append("**Adjustments:**")
            for a in adjustments:
                lines.append(f"- {a}")

        filepath = os.path.join(workspace_path, "feedback", f"day-{day}.md")
        self._write(filepath, "\n".join(lines))

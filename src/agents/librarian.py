"""Librarian agent — creates and manages a local markdown knowledge base for the trip.

DEPRECATED (v2): This agent has been demoted to a tool function.
Use src/tools/library_sync.library_sync() instead.
The graph no longer includes a librarian node. This file is kept for backward compatibility.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.state import TripState
from src.tools.markdown_sync import MarkdownLibrary

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Librarian Agent for a travel planning assistant. You manage a local markdown \
knowledge base that serves as the trip's library. You work for ANY destination.

YOUR RESPONSIBILITIES:
1. Create a structured folder of markdown files adapted to the destination
2. Sync research findings into organized, readable markdown pages
3. Keep the library updated as research, priorities, and plans change
4. Maintain a master INDEX.md that links to all content
"""


class LibrarianAgent(BaseAgent):
    agent_name = "librarian"

    def __init__(self) -> None:
        super().__init__()
        self.library = MarkdownLibrary()

    def get_system_prompt(self, state=None) -> str:
        return SYSTEM_PROMPT

    async def handle(self, state: TripState, user_message: str) -> dict:
        """Sync research data to markdown library."""
        dest = state.get("destination", {})
        country = dest.get("country", "Unknown")
        flag = dest.get("flag_emoji", "")
        currency_code = dest.get("currency_code", "USD")
        currency_symbol = dest.get("currency_symbol", "$")
        exchange_rate = dest.get("exchange_rate_to_usd", 1)
        library_config = dict(state.get("library") or {})

        messages: list[str] = []

        # 1. Create workspace folder if it doesn't exist
        workspace_path = library_config.get("workspace_path", "")
        if not workspace_path:
            year = datetime.now().strftime("%Y")
            workspace_path = self.library.create_workspace(country, flag, year)
            library_config["workspace_path"] = workspace_path
            messages.append(f"📁 Created library folder: {workspace_path}")

        # 2. Write destination guide
        if dest.get("researched_at") and not library_config.get("guide_written"):
            self.library.write_destination_guide(workspace_path, dest)
            library_config["guide_written"] = True
            messages.append(f"📖 Destination guide written for {flag} {country}")

        # 3. Sync research per city
        research = state.get("research", {})
        cities = state.get("cities", [])
        priorities = state.get("priorities", {})
        synced_cities = library_config.get("synced_cities", {})
        total_items = 0

        for city_name, city_research in research.items():
            city_priorities = priorities.get(city_name)
            count = self.library.write_city_research(
                workspace_path,
                city_name,
                city_research,
                currency_symbol,
                currency_code,
                exchange_rate,
                priorities=city_priorities,
            )
            total_items += count
            synced_cities[city_name] = datetime.now(timezone.utc).isoformat()
            messages.append(f"  {city_name}: {count} items written")

        library_config["synced_cities"] = synced_cities

        # 4. Write priorities file
        if priorities:
            self.library.write_priorities(workspace_path, priorities)
            messages.append("🎯 Priorities written")

        # 5. Write itinerary
        plan = state.get("high_level_plan", [])
        if plan:
            self.library.write_itinerary(workspace_path, plan, dest)
            messages.append("📅 Itinerary written")

        # 6. Write feedback entries
        feedback_log = state.get("feedback_log", [])
        written_feedback = library_config.get("feedback_days_written", [])
        for entry in feedback_log:
            day = entry.get("day")
            if day and day not in written_feedback:
                self.library.write_feedback(workspace_path, entry)
                written_feedback.append(day)
        library_config["feedback_days_written"] = written_feedback

        # 7. Update index
        self.library.write_index(
            workspace_path,
            country,
            flag,
            cities,
            state.get("dates", {}),
            has_guide=library_config.get("guide_written", False),
            has_priorities=bool(priorities),
            has_plan=bool(plan),
            has_budget=bool(state.get("cost_tracker", {}).get("daily_log")),
        )
        messages.append("📋 INDEX.md updated")

        # Update library config
        library_config["last_synced"] = datetime.now(timezone.utc).isoformat()

        if not messages:
            messages.append("📚 Library is up to date!")

        response_lines = [f"📚 Library Sync — {workspace_path}"]
        response_lines.extend(messages)
        response_lines.append(f"\nTotal items: {total_items}")

        return {
            "response": "\n".join(response_lines),
            "state_updates": {"library": library_config},
        }

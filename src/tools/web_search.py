"""Tavily web search wrapper for destination and city research."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tavily import AsyncTavilyClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 30.0  # seconds per search call


class WebSearchTool:
    """Wraps Tavily API for travel research queries."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Run a single search query and return results."""
        try:
            response = await asyncio.wait_for(
                self.client.search(query=query, max_results=max_results, search_depth="advanced"),
                timeout=SEARCH_TIMEOUT,
            )
            return response.get("results", [])
        except asyncio.TimeoutError:
            logger.warning("Tavily search timed out after %.0fs for query: %s", SEARCH_TIMEOUT, query)
            return []
        except Exception:
            logger.exception("Tavily search failed for query: %s", query)
            return []

    async def search_destination_intel(self, country: str, travel_dates: str = "") -> list[dict[str, Any]]:
        """Run 6 queries to build country-level destination intelligence."""
        queries = [
            f"{country} travel guide essentials 2025 2026",
            f"{country} tourist tips safety scams",
            f"{country} visa currency payment tipping",
            f"{country} cultural etiquette customs dress code",
            f"{country} transport apps getting around",
            f"{country} weather {travel_dates}" if travel_dates else f"{country} weather travel seasons",
        ]

        all_results: list[dict[str, Any]] = []
        for q in queries:
            results = await self.search(q, max_results=3)
            all_results.extend(results)
            logger.info("Destination intel search: %s → %d results", q, len(results))

        return all_results

    async def search_city(
        self,
        city: str,
        country: str,
        interests: list[str] | None = None,
        traveler_type: str = "",
    ) -> list[dict[str, Any]]:
        """Run 6+ queries for city-level research."""
        year = "2026"
        queries = [
            f"{city} {country} top things to do {year}",
            f"{city} best food restaurants local dishes",
            f"{city} hidden gems off beaten path",
            f"{city} travel tips logistics {traveler_type}".strip(),
            f"{city} day trips from",
        ]

        if interests:
            interest_str = " ".join(interests[:2])
            queries.append(f"{city} {interest_str}")

        all_results: list[dict[str, Any]] = []
        for q in queries:
            results = await self.search(q, max_results=3)
            all_results.extend(results)
            logger.info("City research search: %s → %d results", q, len(results))

        return all_results

"""Transit time estimation — optional external API integration."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# OpenRouteService API
_ORS_BASE_URL = "https://api.openrouteservice.org/v2/directions"

# Mode mapping from user-friendly to ORS profile names
_ORS_PROFILES = {
    "walking": "foot-walking",
    "cycling": "cycling-regular",
    "driving": "driving-car",
}


async def get_transit_time(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    mode: str = "walking",
    api_key: str = "",
) -> dict | None:
    """Estimate transit time between two points.

    Args:
        origin_coords: (lat, lng) of the origin
        dest_coords: (lat, lng) of the destination
        mode: "walking", "cycling", or "driving"
        api_key: OpenRouteService API key

    Returns dict with duration_min, distance_km, mode
    or None if API key not set or request fails.

    Graceful degradation: returns None silently if no API key configured.
    """
    if not api_key:
        return None

    profile = _ORS_PROFILES.get(mode, "foot-walking")

    try:
        # ORS expects coordinates as [lng, lat]
        start = [origin_coords[1], origin_coords[0]]
        end = [dest_coords[1], dest_coords[0]]

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_ORS_BASE_URL}/{profile}/json",
                json={"coordinates": [start, end]},
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        routes = data.get("routes", [])
        if not routes:
            return None

        route = routes[0]
        summary = route.get("summary", {})

        duration_sec = summary.get("duration", 0)
        distance_m = summary.get("distance", 0)

        return {
            "duration_min": round(duration_sec / 60),
            "distance_km": round(distance_m / 1000, 1),
            "mode": mode,
        }

    except Exception:
        logger.warning(
            "Transit API request failed for %s -> %s (%s)",
            origin_coords, dest_coords, mode,
        )
        return None


def estimate_walking_time(distance_km: float) -> int:
    """Fallback estimate: walking time in minutes at ~5 km/h."""
    return round(distance_km / 5.0 * 60)


def format_transit_for_slot(transit: dict | None) -> str:
    """Format transit data for display in agenda slots."""
    if not transit:
        return ""

    mode_emoji = {"walking": "\U0001f6b6", "cycling": "\U0001f6b2", "driving": "\U0001f697"}.get(
        transit.get("mode", ""), "\U0001f6b6"
    )
    return (
        f"{mode_emoji} {transit['duration_min']}min "
        f"({transit['distance_km']}km {transit.get('mode', 'walking')})"
    )

"""Weather forecast tool — optional external API integration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# WeatherAPI.com (free tier: 3 days forecast)
_BASE_URL = "https://api.weatherapi.com/v1/forecast.json"


async def get_forecast(city: str, date: str, api_key: str = "") -> dict | None:
    """Fetch weather forecast for a city on a given date.

    Returns dict with temp_c, temp_f, condition, rain_chance, humidity, wind_kph
    or None if API key not set or request fails.

    Graceful degradation: returns None silently if no API key configured.
    """
    if not api_key:
        return None

    try:
        # Parse date to check if it's within forecast range (3 days for free tier)
        target = datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        days_ahead = (target - today).days

        if days_ahead < 0 or days_ahead > 2:
            # Out of free tier range
            return None

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                _BASE_URL,
                params={
                    "key": api_key,
                    "q": city,
                    "days": days_ahead + 1,
                    "aqi": "no",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract the target day's forecast
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        for day in forecast_days:
            if day.get("date") == date:
                day_data = day.get("day", {})
                return {
                    "temp_high_c": day_data.get("maxtemp_c"),
                    "temp_low_c": day_data.get("mintemp_c"),
                    "temp_high_f": day_data.get("maxtemp_f"),
                    "temp_low_f": day_data.get("mintemp_f"),
                    "condition": day_data.get("condition", {}).get("text", "Unknown"),
                    "rain_chance": day_data.get("daily_chance_of_rain", 0),
                    "humidity": day_data.get("avghumidity"),
                    "wind_kph": day_data.get("maxwind_kph"),
                    "uv_index": day_data.get("uv"),
                }

        return None

    except Exception:
        logger.warning("Weather API request failed for %s on %s", city, date)
        return None


def format_weather_for_prompt(forecast: dict | None) -> str:
    """Format weather data for injection into agent prompts."""
    if not forecast:
        return ""

    condition = forecast.get("condition", "Unknown")
    high_c = forecast.get("temp_high_c", "?")
    low_c = forecast.get("temp_low_c", "?")
    rain = forecast.get("rain_chance", 0)

    parts = [f"Weather: {condition}, {low_c}-{high_c}\u00b0C"]
    if rain and int(rain) > 30:
        parts.append(f"Rain chance: {rain}% \u2014 have a backup plan!")
    if forecast.get("uv_index") and float(forecast["uv_index"]) > 6:
        parts.append(f"UV: {forecast['uv_index']} \u2014 bring sunscreen!")

    return " | ".join(parts)

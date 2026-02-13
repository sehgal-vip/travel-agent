"""Shared test fixtures — destination-agnostic test states."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.state import TripState


@pytest.fixture
def empty_state() -> TripState:
    """A brand new state with no data — pre-onboarding."""
    return TripState(
        trip_id="",
        created_at="",
        updated_at="",
        destination={},
        cities=[],
        travelers={},
        budget={},
        dates={},
        interests=[],
        must_dos=[],
        deal_breakers=[],
        accommodation_pref="",
        transport_pref=[],
        research={},
        priorities={},
        high_level_plan=[],
        plan_status="not_started",
        plan_version=0,
        detailed_agenda=[],
        feedback_log=[],
        cost_tracker={},
        library={},
        current_agent="orchestrator",
        conversation_history=[],
        onboarding_complete=False,
        onboarding_step=0,
        current_trip_day=None,
        agent_scratch={},
        messages=[],
        _next="",
        _user_message="",
    )


@pytest.fixture
def japan_state() -> TripState:
    """A fully onboarded Japan trip state for testing post-onboarding agents."""
    return TripState(
        trip_id="japan-2026",
        trip_title="Ramen & Temples Run",
        created_at="2026-02-01T00:00:00Z",
        updated_at="2026-02-01T00:00:00Z",
        destination={
            "country": "Japan",
            "country_code": "JP",
            "region": "East Asia",
            "flag_emoji": "\U0001f1ef\U0001f1f5",
            "language": "Japanese",
            "useful_phrases": {
                "thank you": "arigatou gozaimasu",
                "hello": "konnichiwa",
                "excuse me": "sumimasen",
                "how much": "ikura desu ka",
            },
            "currency_code": "JPY",
            "currency_symbol": "\u00a5",
            "exchange_rate_to_usd": 148.0,
            "tipping_culture": "Not customary. Do not tip.",
            "visa_requirements": "Visa-free for most nationalities (90 days)",
            "safety_notes": ["Very safe country", "Natural disaster preparedness important"],
            "common_scams": ["Overcharged at tourist shops"],
            "transport_apps": ["Google Maps", "Navitime"],
            "payment_norms": "Mix of cash and card; many places still cash-preferred",
            "emergency_numbers": {"police": "110", "ambulance": "119"},
            "plug_type": "Type A/B",
            "voltage": "100V",
            "time_zone": "Asia/Tokyo",
            "climate_type": "temperate",
            "current_season_notes": "Cherry blossom season in early April",
            "cultural_notes": ["Remove shoes indoors", "Bow when greeting", "No tipping"],
            "health_advisories": [],
            "pricing_tier": "expensive",
            "daily_budget_benchmarks": {"backpacker": 50, "midrange": 150, "luxury": 400},
            "common_intercity_transport": ["shinkansen", "bus", "domestic_flight"],
            "booking_platforms": ["Klook", "Japan Rail Pass"],
            "sim_connectivity": "eSIM recommended, pocket WiFi available at airports",
            "researched_at": "2026-02-01T10:00:00Z",
        },
        cities=[
            {"name": "Tokyo", "country": "Japan", "days": 4, "order": 1},
            {"name": "Kyoto", "country": "Japan", "days": 4, "order": 2},
            {"name": "Osaka", "country": "Japan", "days": 3, "order": 3},
        ],
        travelers={"count": 2, "type": "couple", "dietary": [], "accessibility": []},
        budget={"style": "midrange", "total_estimate_usd": 4000, "splurge_on": ["food"], "save_on": ["transport"]},
        dates={"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
        interests=["food", "culture", "nature", "photography"],
        must_dos=["Fushimi Inari at sunrise", "Tsukiji outer market"],
        deal_breakers=["large tour groups"],
        accommodation_pref="boutique_and_ryokans",
        transport_pref=["train", "public_transit"],
        research={},
        priorities={},
        high_level_plan=[],
        plan_status="not_started",
        plan_version=0,
        detailed_agenda=[],
        feedback_log=[],
        cost_tracker={
            "budget_total_usd": 4000,
            "budget_daily_target_usd": 286,
            "local_currency": "JPY",
            "currency_symbol": "\u00a5",
            "exchange_rate": 148.0,
            "pricing_benchmarks": {},
            "daily_log": [],
            "totals": {"spent_usd": 0, "remaining_usd": 4000, "daily_avg_usd": 0, "projected_total_usd": 0, "status": "on_track"},
            "by_category": {},
            "by_city": {},
            "savings_tips": [],
        },
        library={"workspace_path": None, "guide_written": False, "synced_cities": {}, "feedback_days_written": [], "last_synced": None},
        current_agent="orchestrator",
        conversation_history=[],
        onboarding_complete=True,
        onboarding_step=11,
        current_trip_day=1,
        agent_scratch={},
        messages=[],
        _next="",
        _user_message="",
    )


@pytest.fixture
def morocco_state() -> TripState:
    """A fully onboarded Morocco trip state — different continent, currency, climate."""
    return TripState(
        trip_id="morocco-2026",
        trip_title="Souks Spices & Sunsets",
        created_at="2026-03-01T00:00:00Z",
        updated_at="2026-03-01T00:00:00Z",
        destination={
            "country": "Morocco",
            "country_code": "MA",
            "region": "North Africa",
            "flag_emoji": "\U0001f1f2\U0001f1e6",
            "language": "Arabic / French",
            "useful_phrases": {
                "thank you": "shukran",
                "hello": "salam alaikum",
                "no thank you": "la shukran",
                "how much": "bshhal",
            },
            "currency_code": "MAD",
            "currency_symbol": "DH",
            "exchange_rate_to_usd": 10.0,
            "tipping_culture": "Expected. 10-15% at restaurants, small tips for services.",
            "visa_requirements": "Visa-free for most nationalities (90 days)",
            "safety_notes": ["Generally safe", "Watch for pickpockets in medinas"],
            "common_scams": ["Fake guides in medinas", "Carpet shop pressure"],
            "transport_apps": ["InDrive", "Careem"],
            "payment_norms": "Cash dominant, especially in medinas",
            "emergency_numbers": {"police": "19", "ambulance": "15"},
            "plug_type": "Type C/E",
            "voltage": "220V",
            "time_zone": "Africa/Casablanca",
            "climate_type": "arid",
            "current_season_notes": "March is pleasant, warm days and cool nights",
            "cultural_notes": ["Dress modestly", "Remove shoes in homes/riads", "Use right hand for eating"],
            "health_advisories": ["Drink bottled water"],
            "pricing_tier": "budget",
            "daily_budget_benchmarks": {"backpacker": 30, "midrange": 80, "luxury": 250},
            "common_intercity_transport": ["train (ONCF)", "CTM bus", "grand taxi"],
            "booking_platforms": ["GetYourGuide", "Viator"],
            "sim_connectivity": "SIM cards available at airports, cheap data",
            "researched_at": "2026-03-01T10:00:00Z",
        },
        cities=[
            {"name": "Marrakech", "country": "Morocco", "days": 3, "order": 1},
            {"name": "Fes", "country": "Morocco", "days": 2, "order": 2},
        ],
        travelers={"count": 2, "type": "couple", "dietary": [], "accessibility": []},
        budget={"style": "midrange", "total_estimate_usd": 1500, "splurge_on": ["accommodation"], "save_on": []},
        dates={"start": "2026-03-15", "end": "2026-03-20", "total_days": 5},
        interests=["food", "culture", "architecture", "photography"],
        must_dos=["Jardin Majorelle", "Fes medina"],
        deal_breakers=[],
        accommodation_pref="riad",
        transport_pref=["train", "taxi"],
        research={},
        priorities={},
        high_level_plan=[],
        plan_status="not_started",
        plan_version=0,
        detailed_agenda=[],
        feedback_log=[],
        cost_tracker={
            "budget_total_usd": 1500,
            "budget_daily_target_usd": 300,
            "local_currency": "MAD",
            "currency_symbol": "DH",
            "exchange_rate": 10.0,
            "pricing_benchmarks": {},
            "daily_log": [],
            "totals": {"spent_usd": 0, "remaining_usd": 1500, "daily_avg_usd": 0, "projected_total_usd": 0, "status": "on_track"},
            "by_category": {},
            "by_city": {},
            "savings_tips": [],
        },
        library={"workspace_path": None, "guide_written": False, "synced_cities": {}, "feedback_days_written": [], "last_synced": None},
        current_agent="orchestrator",
        conversation_history=[],
        onboarding_complete=True,
        onboarding_step=11,
        current_trip_day=1,
        agent_scratch={},
        messages=[],
        _next="",
        _user_message="",
    )


@pytest_asyncio.fixture
async def async_db(tmp_path):
    """Create an in-memory async database for testing."""
    from src.db.persistence import TripRepository

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    repo = TripRepository(db_url)
    await repo.init_db()
    yield repo
    await repo.close()

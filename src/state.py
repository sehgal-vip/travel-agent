"""
State schema — LangGraph shared state (destination-agnostic).

Root TripState is a TypedDict (required by LangGraph).
Nested structures use TypedDict for JSON-serialisability.
Enums use str mixin for easy serialisation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict


# ─── Enums ────────────────────────────────────────


class BudgetStyle(str, Enum):
    BACKPACKER = "backpacker"
    MIDRANGE = "midrange"
    LUXURY = "luxury"
    MIXED = "mixed"


class Tier(str, Enum):
    MUST_DO = "must_do"
    NICE_TO_HAVE = "nice_to_have"
    IF_NEARBY = "if_nearby"
    SKIP = "skip"


class Vibe(str, Enum):
    EASY = "easy"
    ACTIVE = "active"
    CULTURAL = "cultural"
    FOODIE = "foodie"
    ADVENTUROUS = "adventurous"
    RELAXED = "relaxed"
    MIXED = "mixed"


class ItemStatus(str, Enum):
    NOT_STARTED = "not_started"
    PLANNED = "planned"
    DONE = "done"
    SKIPPED = "skipped"


class EnergyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SlotType(str, Enum):
    ACTIVITY = "activity"
    MEAL = "meal"
    TRANSPORT = "transport"
    REST = "rest"
    FREE = "free"
    SPECIAL = "special"


class PricingTier(str, Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"
    VERY_EXPENSIVE = "very_expensive"


class ClimateType(str, Enum):
    TROPICAL = "tropical"
    TEMPERATE = "temperate"
    COLD = "cold"
    ARID = "arid"
    HIGH_ALTITUDE = "high_altitude"


# ─── Destination Intelligence ─────────────────────


class DestinationIntel(TypedDict, total=False):
    """Country/region-level intelligence. Researched once per destination."""

    country: str
    country_code: str  # ISO 3166-1 alpha-2
    region: str  # "Southeast Asia", "Western Europe"
    flag_emoji: str
    language: str
    useful_phrases: dict[str, str]
    currency_code: str  # "JPY", "EUR", "VND"
    currency_symbol: str  # "¥", "€", "₫"
    exchange_rate_to_usd: float
    tipping_culture: str
    visa_requirements: str
    safety_notes: list[str]
    common_scams: list[str]
    transport_apps: list[str]
    payment_norms: str
    emergency_numbers: dict[str, str]
    plug_type: str
    voltage: str
    time_zone: str
    climate_type: str  # ClimateType value
    current_season_notes: str
    cultural_notes: list[str]
    health_advisories: list[str]
    pricing_tier: str  # PricingTier value
    daily_budget_benchmarks: dict[str, float]
    common_intercity_transport: list[str]
    booking_platforms: list[str]
    sim_connectivity: str
    researched_at: str  # ISO timestamp


# ─── Trip Configuration ──────────────────────────


class CityConfig(TypedDict, total=False):
    name: str
    country: str
    days: int
    order: int
    arrival_date: Optional[str]
    departure_date: Optional[str]


class TravelerConfig(TypedDict, total=False):
    count: int
    type: str  # "couple", "solo", "friends", "family"
    ages: Optional[list[int]]
    dietary: list[str]
    accessibility: list[str]
    nationalities: Optional[list[str]]


class BudgetConfig(TypedDict, total=False):
    style: str  # BudgetStyle value
    total_estimate_usd: Optional[float]
    daily_target_usd: Optional[float]
    splurge_on: list[str]
    save_on: list[str]


class DateConfig(TypedDict, total=False):
    start: str
    end: str
    total_days: int


# ─── Research ────────────────────────────────────


class Coordinates(TypedDict, total=False):
    lat: float
    lng: float


class ResearchItem(TypedDict, total=False):
    id: str  # "{city_slug}-{category}-{number}"
    name: str
    name_local: Optional[str]
    category: str  # place | activity | food | logistics | tip
    subcategory: str
    description: str
    cost_local: Optional[float]
    cost_usd: Optional[float]
    time_needed_hrs: Optional[float]
    best_time: Optional[str]
    location: Optional[str]
    coordinates: Optional[Coordinates]
    getting_there: Optional[str]
    traveler_suitability_score: Optional[int]  # 1-5
    advance_booking: bool
    booking_lead_time: Optional[str]
    tags: list[str]
    seasonal_relevance: Optional[str]
    sources: list[str]
    notes: Optional[str]
    must_try_items: Optional[list[str]]


class CityResearch(TypedDict, total=False):
    last_updated: str
    places: list[ResearchItem]
    activities: list[ResearchItem]
    food: list[ResearchItem]
    logistics: list[ResearchItem]
    tips: list[ResearchItem]
    hidden_gems: list[ResearchItem]


# ─── Priorities ──────────────────────────────────


class PrioritizedItem(TypedDict, total=False):
    item_id: str
    name: str
    category: str
    tier: str  # Tier value
    score: int
    reason: str
    group: Optional[str]
    user_override: bool


# ─── Itinerary ───────────────────────────────────


class TravelSegment(TypedDict, total=False):
    mode: str
    from_city: str
    to_city: str
    duration_hrs: float
    cost_usd: Optional[float]
    cost_local: Optional[float]
    booking_platform: Optional[str]
    notes: Optional[str]


class MealSlot(TypedDict, total=False):
    item_id: Optional[str]
    name: Optional[str]
    type: str  # "specific" | "explore" | "hotel"
    area: Optional[str]


class DayPlan(TypedDict, total=False):
    day: int
    date: str
    city: str
    theme: str
    vibe: str  # Vibe value
    travel: Optional[TravelSegment]
    key_activities: list[dict]
    meals: dict  # {breakfast, lunch, dinner} → MealSlot
    special_moment: Optional[str]
    notes: Optional[str]
    free_time_blocks: list[str]
    estimated_cost_usd: Optional[float]


# ─── Detailed Agenda ─────────────────────────────


class TransportDetail(TypedDict, total=False):
    mode: str
    line: Optional[str]
    duration_min: int
    cost_local: Optional[float]
    cost_usd: Optional[float]
    directions: Optional[str]


class AgendaSlot(TypedDict, total=False):
    time: str
    end_time: str
    type: str  # SlotType value
    name: str
    item_id: Optional[str]
    address: Optional[str]
    coordinates: Optional[Coordinates]
    transport_from_prev: Optional[TransportDetail]
    duration_min: int
    cost_local: Optional[float]
    cost_usd: Optional[float]
    cost_for: str
    tips: Optional[str]
    local_phrase: Optional[str]
    payment: Optional[str]
    booking_required: bool
    booking_urgency: Optional[str]
    rain_backup: Optional[str]
    tags: list[str]


class DailyCostEstimate(TypedDict, total=False):
    food: float
    food_usd: float
    activities: float
    activities_usd: float
    transport: float
    transport_usd: float
    accommodation: float
    accommodation_usd: float
    other: float
    other_usd: float
    total_local: float
    total_usd: float
    currency_code: str


class QuickReference(TypedDict, total=False):
    hotel: Optional[str]
    emergency: dict[str, str]
    transport_app: str
    exchange_rate: str
    weather: Optional[str]
    key_phrases: dict[str, str]
    reminders: list[str]


class DetailedDay(TypedDict, total=False):
    day: int
    date: str
    city: str
    theme: str
    slots: list[AgendaSlot]
    daily_cost_estimate: DailyCostEstimate
    booking_alerts: list[dict]
    quick_reference: QuickReference


# ─── Feedback ────────────────────────────────────


class DestinationFeedback(TypedDict, total=False):
    transport: str
    language: str
    safety: str
    accommodation: str
    food_quality: str
    connectivity: str


class FeedbackEntry(TypedDict, total=False):
    day: int
    date: str
    city: str
    completed_items: list[str]
    skipped_items: list[str]
    highlight: Optional[str]
    lowlight: Optional[str]
    energy_level: str  # EnergyLevel value
    food_rating: str
    budget_status: str
    weather: str
    discoveries: list[str]
    preference_shifts: list[str]
    destination_feedback: DestinationFeedback
    adjustments_made: list[str]
    sentiment: str
    actual_spend_usd: Optional[float]
    actual_spend_local: Optional[float]


# ─── Cost Tracking ───────────────────────────────


class PricingBenchmarks(TypedDict, total=False):
    accommodation: dict[str, float]
    meal: dict[str, float]
    street_food: dict[str, float]
    local_transport: dict[str, float]
    day_tour: dict[str, float]
    coffee: dict[str, float]
    beer: dict[str, float]


class DailyCostLog(TypedDict, total=False):
    day: int
    date: str
    city: str
    estimated_usd: float
    actual_usd: Optional[float]
    breakdown_usd: dict[str, float]
    breakdown_local: Optional[dict[str, float]]
    notes: Optional[str]


class CostTotals(TypedDict, total=False):
    spent_usd: float
    remaining_usd: float
    daily_avg_usd: float
    projected_total_usd: float
    status: str


class CostTracker(TypedDict, total=False):
    budget_total_usd: float
    budget_daily_target_usd: float
    local_currency: str
    currency_symbol: str
    exchange_rate: float
    pricing_benchmarks: PricingBenchmarks
    daily_log: list[DailyCostLog]
    totals: CostTotals
    by_category: dict[str, dict]
    by_city: dict[str, dict]
    savings_tips: list[str]


# ─── Library (Markdown Knowledge Base) ───────────


class LibraryConfig(TypedDict, total=False):
    workspace_path: Optional[str]
    guide_written: bool
    synced_cities: dict[str, str]  # city_name → last_synced ISO timestamp
    feedback_days_written: list[int]
    last_synced: Optional[str]


# ─── Conversation ────────────────────────────────


class Message(TypedDict, total=False):
    role: str
    content: str
    timestamp: str
    agent: Optional[str]


# ═══ ROOT STATE ══════════════════════════════════


class TripState(TypedDict, total=False):
    """Root state object for LangGraph. TypedDict at root, typed nested dicts."""

    # Identity
    trip_id: str
    created_at: str
    updated_at: str

    # Destination intelligence
    destination: DestinationIntel

    # Configuration
    cities: list[CityConfig]
    travelers: TravelerConfig
    budget: BudgetConfig
    dates: DateConfig
    interests: list[str]
    must_dos: list[str]
    deal_breakers: list[str]
    accommodation_pref: str
    transport_pref: list[str]

    # Research
    research: dict[str, CityResearch]

    # Priorities
    priorities: dict[str, list[PrioritizedItem]]

    # Planning
    high_level_plan: list[DayPlan]
    plan_status: str
    plan_version: int

    # Scheduling
    detailed_agenda: list[DetailedDay]

    # Feedback
    feedback_log: list[FeedbackEntry]

    # Costs
    cost_tracker: CostTracker

    # Library (local markdown knowledge base)
    library: LibraryConfig

    # Conversation management
    current_agent: str
    conversation_history: list[Message]
    onboarding_complete: bool
    onboarding_step: Optional[int]
    current_trip_day: Optional[int]
    agent_scratch: dict

    # LangGraph messages (for internal LLM conversation)
    messages: list

    # Internal routing (used by graph nodes)
    _next: str
    _user_message: str

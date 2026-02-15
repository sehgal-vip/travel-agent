"""Per-agent memory — each specialist agent owns a single .md file per trip.

Folder layout::

    data/trips/{trip_id}/
    +-- research.md    <- research agent's memory
    +-- planner.md     <- planner agent's memory
    +-- scheduler.md   <- scheduler agent's memory
    +-- prioritizer.md <- prioritizer agent's memory
    +-- feedback.md    <- feedback agent's memory
    +-- cost.md        <- cost agent's memory
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────

NOTES_MARKER = "## Agent Notes [accumulated] <!-- mem:notes -->"
_OLD_NOTES_MARKER = "## Agent Notes [accumulated]"  # fallback for reading old files
MAX_NOTES_LINES = 30  # Cap to prevent unbounded growth
PINNED_TAG = "[pinned]"
SHARED_TAG = "[shared]"
MAX_PINNED_LINES = 10
MAX_EPHEMERAL_LINES = 25
SCHEDULER_FEEDBACK_ENTRIES = 3  # named constant for scheduler feedback lookback
MEMORY_FORMAT_VERSION = "1"
_MEMORY_SIZE_WARN_BYTES = 50 * 1024  # warn if memory file exceeds 50KB


# ─── Async Lock Registry ────────────────────────────────

_memory_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _get_lock(trip_id: str, agent_name: str) -> asyncio.Lock:
    """Get or create an async lock for a (trip_id, agent_name) pair.

    No await → atomic in single-threaded asyncio; safe without additional synchronization.
    """
    key = (trip_id, agent_name)
    if key not in _memory_locks:
        _memory_locks[key] = asyncio.Lock()
    return _memory_locks[key]


# ─── Atomic Write Helper ────────────────────────────────


def _atomic_write(filepath: Path, content: str) -> None:
    """Write content atomically via temp file + os.replace. Prevents corruption on interrupted writes."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(filepath))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── New Public API — per-agent memory ──────────────────


async def refresh_agent_memory(trip_id: str, agent_name: str, content: str, base_dir: str = "./data/trips") -> None:
    """Write agent's memory file with async locking and atomic write.

    Idempotency: skips write if content is unchanged (MD5 hash comparison).
    """
    async with _get_lock(trip_id, agent_name):
        folder = Path(base_dir) / trip_id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{agent_name}.md"
        # Idempotency: skip write if content unchanged
        if target.is_file():
            existing_hash = hashlib.md5(target.read_bytes()).hexdigest()
            new_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            if existing_hash == new_hash:
                return
        _atomic_write(target, content)
        # Observability: log size and warn if large
        size = len(content.encode("utf-8"))
        est_tokens = len(content) // 3
        logger.debug(
            "Memory written: %s/%s (%d bytes, ~%d tokens)",
            trip_id, agent_name, size, est_tokens,
        )
        if size > _MEMORY_SIZE_WARN_BYTES:
            logger.warning(
                "Memory file %s/%s exceeds %dKB (%d bytes, ~%d tokens)",
                trip_id, agent_name, _MEMORY_SIZE_WARN_BYTES // 1024, size, est_tokens,
            )


def get_memory_stats(trip_id: str, base_dir: str = "./data/trips") -> dict[str, dict]:
    """Return size/token stats for all agent memory files in a trip. For observability."""
    from src.agents.constants import MEMORY_AGENTS

    folder = Path(base_dir) / trip_id
    stats: dict[str, dict] = {}
    if not folder.is_dir():
        return stats
    for agent in MEMORY_AGENTS:
        filepath = folder / f"{agent}.md"
        if filepath.is_file():
            content = filepath.read_text(encoding="utf-8")
            size = len(content.encode("utf-8"))
            stats[agent] = {
                "size_bytes": size,
                "estimated_tokens": len(content) // 3,
                "has_notes": NOTES_MARKER in content or _OLD_NOTES_MARKER in content,
            }
    return stats


def read_agent_notes(trip_id: str, agent_name: str, base_dir: str = "./data/trips") -> str | None:
    """Read just the Agent Notes section from an existing agent file.

    Notes are always the LAST section in the file (after all ## headings).
    Sync read — acceptable for small files (<50KB).
    Supports both new and old NOTES_MARKER formats.
    Separates pinned (durable) notes from ephemeral notes, capping each independently.
    """
    filepath = Path(base_dir) / trip_id / f"{agent_name}.md"
    if not filepath.is_file():
        return None
    text = filepath.read_text(encoding="utf-8")
    # Support both old and new marker
    marker = NOTES_MARKER if NOTES_MARKER in text else (_OLD_NOTES_MARKER if _OLD_NOTES_MARKER in text else None)
    if marker is None:
        return None
    # Notes are always last — take everything after the marker
    start = text.index(marker) + len(marker)
    rest = text[start:].strip()
    if not rest:
        return None
    lines = rest.splitlines()
    # Separate pinned from ephemeral
    pinned = [l for l in lines if PINNED_TAG in l]
    ephemeral = [l for l in lines if PINNED_TAG not in l]
    # Cap each independently
    pinned = pinned[-MAX_PINNED_LINES:]
    ephemeral = ephemeral[-MAX_EPHEMERAL_LINES:]
    result = pinned + ephemeral
    if not result:
        return None
    return "\n".join(result)


async def append_agent_notes(trip_id: str, agent_name: str, new_notes: str, base_dir: str = "./data/trips") -> None:
    """Append LLM-generated notes to an agent's file with async locking and atomic write.

    Sanitizes incoming notes to prevent marker injection.
    """
    # Sanitize: strip any marker strings from incoming notes
    new_notes = new_notes.replace(NOTES_MARKER, "").replace(_OLD_NOTES_MARKER, "").strip()
    if not new_notes:
        return
    async with _get_lock(trip_id, agent_name):
        filepath = Path(base_dir) / trip_id / f"{agent_name}.md"
        if not filepath.is_file():
            return
        text = filepath.read_text(encoding="utf-8")
        # Support both old and new marker
        if NOTES_MARKER in text or _OLD_NOTES_MARKER in text:
            text += f"\n{new_notes}"
        else:
            text += f"\n\n{NOTES_MARKER}\n{new_notes}"
        _atomic_write(filepath, text)


def read_shared_insights(trip_id: str, exclude_agent: str, base_dir: str = "./data/trips") -> str:
    """Read [shared]-tagged notes from all agents except exclude_agent.

    Returns a formatted section for injection into an agent's memory, or empty string.
    Each line is prefixed with the source agent name.

    Timing note: The handler writes agent X's memory (including new [shared] notes) *after*
    agent X responds. So if LangGraph routes through multiple agents in a single pass,
    agent Y won't see agent X's [shared] notes until the *next* request.
    """
    from src.agents.constants import MEMORY_AGENTS

    folder = Path(base_dir) / trip_id
    if not folder.is_dir():
        return ""
    lines: list[str] = []
    for agent in sorted(MEMORY_AGENTS):
        if agent == exclude_agent:
            continue
        notes = read_agent_notes(trip_id, agent, base_dir)
        if not notes:
            continue
        shared = [l for l in notes.splitlines() if SHARED_TAG in l]
        for line in shared:
            lines.append(f"[{agent}] {line}")
    if not lines:
        return ""
    # Cap at 15 lines
    return "## Cross-Agent Insights [auto-refreshed]\n" + "\n".join(lines[:15])


# Module-level builder dispatch — single source of truth for agent→builder mapping
_ALL_BUILDERS: dict[str, callable] = {}  # populated after builder functions are defined

_validated = False


def _validate_agent_lists() -> None:
    """Lazily validate that builder keys match MEMORY_AGENTS and NOTES_ELIGIBLE is a subset."""
    global _validated
    if _validated:
        return
    from src.agents.constants import MEMORY_AGENTS, NOTES_ELIGIBLE_AGENTS

    if set(_ALL_BUILDERS.keys()) != MEMORY_AGENTS:
        raise RuntimeError(
            f"Builder/MEMORY_AGENTS mismatch: builders={set(_ALL_BUILDERS.keys())}, "
            f"MEMORY_AGENTS={MEMORY_AGENTS}"
        )
    if not NOTES_ELIGIBLE_AGENTS <= MEMORY_AGENTS:
        raise RuntimeError(
            f"NOTES_ELIGIBLE_AGENTS {NOTES_ELIGIBLE_AGENTS} is not a subset of "
            f"MEMORY_AGENTS {MEMORY_AGENTS}"
        )
    _validated = True


def build_agent_memory_content(trip_id: str, agent_name: str, state: dict, base_dir: str = "./data/trips") -> str | None:
    """Build memory content for an agent without writing to disk.

    Returns the full memory string (programmatic sections + notes), or None if
    agent_name is not a memory agent. Used by both _ensure_memory_fresh (for the
    prompt) and build_and_write_agent_memory (for persistence).

    Note: This calls read_agent_notes which performs a sync file read. This is
    acceptable for small files (<50KB). In rapid multi-message conversations, the
    agent's prompt may be missing the most recent note from the immediately preceding
    interaction (because append_agent_notes is async and may not have completed).
    This is a conscious design tradeoff — the note will appear on the next invocation.
    """
    _validate_agent_lists()
    from src.agents.constants import MEMORY_AGENTS

    if agent_name not in MEMORY_AGENTS:
        return None

    builder = _ALL_BUILDERS.get(agent_name)
    if not builder:
        return None

    title = state.get("trip_title") or _fallback_title(state)
    programmatic = f"<!-- memory_format: {MEMORY_FORMAT_VERSION} -->\n"
    programmatic += f"# {agent_name.title()} Memory — {title}\n\n"
    programmatic += builder(state)

    # Cross-agent insights: [shared]-tagged notes from other agents
    shared_insights = read_shared_insights(trip_id, agent_name, base_dir)
    if shared_insights:
        programmatic += f"\n\n{shared_insights}"

    # Preserve existing notes
    existing_notes = read_agent_notes(trip_id, agent_name, base_dir)
    if existing_notes:
        programmatic += f"\n\n{NOTES_MARKER}\n{existing_notes}"

    return programmatic


async def build_and_write_agent_memory(trip_id: str, agent_name: str, state: dict, base_dir: str = "./data/trips") -> None:
    """Build programmatic sections for an agent, preserve notes, write file.

    Called by handlers.py after agent completes work. Avoids instantiating agent classes.
    """
    content = build_agent_memory_content(trip_id, agent_name, state, base_dir)
    if content is None:
        return

    await refresh_agent_memory(trip_id, agent_name, content, base_dir)




# ─── Helpers ─────────────────────────────────────────────


def _fallback_title(state: dict) -> str:
    dest = state.get("destination", {})
    country = dest.get("country", "Trip")
    return f"{country} Trip"


SLIM_FIELDS = {"destination", "dates", "cities", "travelers", "interests"}
EXTENDED_FIELDS = SLIM_FIELDS | {"must_dos", "budget"}
COST_FIELDS = EXTENDED_FIELDS | {"currency"}  # adds currency details


def _build_context(state: dict, fields: set[str] | None = None) -> str:
    """Build a trip context block using the specified field set.

    Replaces the 3 inline context blocks across planner, prioritizer, and cost builders.
    Defaults to SLIM_FIELDS if not specified.
    """
    if fields is None:
        fields = SLIM_FIELDS

    dest = state.get("destination", {})
    dates = state.get("dates", {})
    cities = state.get("cities", [])
    travelers = state.get("travelers", {})
    interests = state.get("interests", [])

    route = " → ".join(f"{c.get('name', '?')} ({c.get('days', '?')}d)" for c in cities)
    lines = [
        "## Trip Context [auto-refreshed]",
        f"Destination: {dest.get('flag_emoji', '')} {dest.get('country', '?')} | "
        f"Dates: {dates.get('start', '?')} to {dates.get('end', '?')} ({dates.get('total_days', '?')} days) | "
        f"Travelers: {travelers.get('type', '?')}, {travelers.get('count', '?')}",
        f"Route: {route}",
    ]
    if interests and "interests" in fields:
        lines.append(f"Interests: {', '.join(interests)}")
    if "must_dos" in fields:
        must_dos = state.get("must_dos", [])
        if must_dos:
            lines.append(f"Must-dos: {', '.join(must_dos)}")
    if "budget" in fields:
        budget = state.get("budget", {})
        if budget:
            lines.append(f"Budget: {budget.get('style', '?')} · ${budget.get('total_estimate_usd', '?'):,} total")
    if "currency" in fields:
        if dest.get("currency_code"):
            rate = dest.get("exchange_rate_to_usd", "?")
            lines.append(f"Currency: {dest['currency_code']} ({dest.get('currency_symbol', '?')}) · 1 USD = {rate}")

    return "\n".join(lines)


def _build_slim_context(state: dict) -> str:
    """Compact trip context block — backwards-compatible wrapper around _build_context."""
    return _build_context(state, SLIM_FIELDS)


def _build_research_status(state: dict) -> str:
    """Research progress table for research/prioritizer agents."""
    cities = state.get("cities", [])
    research = state.get("research", {})
    if not cities:
        return ""

    lines = [
        "## Research Progress [auto-refreshed]",
        "| City | Status | Items | Updated |",
        "|------|--------|-------|---------|",
    ]
    for city in cities:
        name = city.get("name", "?")
        city_data = research.get(name)
        if city_data:
            total = sum(len(city_data.get(k, [])) for k in ("places", "activities", "food", "logistics", "tips", "hidden_gems"))
            updated = city_data.get("last_updated", "?")
            if isinstance(updated, str) and "T" in updated:
                updated = updated.split("T")[0]
            lines.append(f"| {name} | ✅ | {total} items | {updated} |")
        else:
            lines.append(f"| {name} | ⏳ | — | — |")

    return "\n".join(lines)


# ─── Per-Agent Builder Functions ─────────────────────────


def _build_research_memory(state: dict) -> str:
    """Programmatic sections for the research agent."""
    sections = []
    sections.append(_build_slim_context(state))
    if state.get("destination", {}).get("researched_at"):
        sections.append(build_destination_intel(state))
    sections.append(_build_research_status(state))
    research = state.get("research", {})
    if research:
        sections.append(build_research_findings(state))
    return "\n\n".join(s for s in sections if s)


def _build_planner_memory(state: dict) -> str:
    """Programmatic sections for the planner agent."""
    sections = []

    # Extended context with interests, must-dos, budget
    sections.append(_build_context(state, EXTENDED_FIELDS))

    # Priorities summary
    priorities = state.get("priorities")
    if priorities:
        sections.append(build_priorities_knowledge(state))

    # Food highlights for meal planning
    research = state.get("research", {})
    food_lines = ["## Food Highlights [auto-refreshed]"]
    has_food = False
    for city_name, city_data in research.items():
        food_items = city_data.get("food", [])
        if food_items:
            has_food = True
            names = ", ".join(f.get("name", "?") for f in food_items[:5])
            food_lines.append(f"**{city_name}**: {names}")
    if has_food:
        sections.append("\n".join(food_lines))

    # Itinerary (if already exists — for adjustments)
    plan = state.get("high_level_plan")
    if plan:
        sections.append(build_itinerary_knowledge(state))

    return "\n\n".join(s for s in sections if s)


def _build_scheduler_memory(state: dict) -> str:
    """Programmatic sections for the scheduler agent."""
    sections = []
    sections.append(_build_slim_context(state))

    # Itinerary
    plan = state.get("high_level_plan")
    if plan:
        sections.append(build_itinerary_knowledge(state))

    # Practical info
    dest = state.get("destination", {})
    practical_lines = ["## Practical Info [auto-refreshed]"]
    if dest.get("climate_type"):
        climate = dest["climate_type"]
        season = dest.get("current_season_notes")
        if season:
            climate += f" — {season}"
        practical_lines.append(f"Climate: {climate}")
    if dest.get("transport_apps"):
        practical_lines.append(f"Transport apps: {', '.join(dest['transport_apps'])}")
    if dest.get("payment_norms"):
        practical_lines.append(f"Payment: {dest['payment_norms']}")
    phrases = dest.get("useful_phrases", {})
    if phrases:
        phrase_parts = [f'"{k}" = "{v}"' for k, v in list(phrases.items())[:5]]
        practical_lines.append(f"Key phrases: {', '.join(phrase_parts)}")
    emergency = dest.get("emergency_numbers", {})
    if emergency:
        practical_lines.append(f"Emergency: {', '.join(f'{k}: {v}' for k, v in emergency.items())}")
    if len(practical_lines) > 1:
        sections.append("\n".join(practical_lines))

    # Booking deadlines
    deadlines = build_booking_deadlines(state)
    if deadlines:
        sections.append(deadlines)

    # Recent feedback (last N entries)
    feedback_log = state.get("feedback_log", [])
    if feedback_log:
        fb_lines = ["## Recent Feedback [auto-refreshed]"]
        for entry in feedback_log[-SCHEDULER_FEEDBACK_ENTRIES:]:
            day = entry.get("day", "?")
            highlight = entry.get("highlight", "")
            energy = entry.get("energy_level", "?")
            fb_lines.append(f"- Day {day}: {highlight} (energy: {energy})")
        sections.append("\n".join(fb_lines))

    return "\n\n".join(s for s in sections if s)


def _build_prioritizer_memory(state: dict) -> str:
    """Programmatic sections for the prioritizer agent."""
    sections = []

    # Context with interests and must-dos
    sections.append(_build_context(state, EXTENDED_FIELDS))

    # Research progress + findings
    sections.append(_build_research_status(state))
    research = state.get("research", {})
    if research:
        sections.append(build_research_findings(state))

    return "\n\n".join(s for s in sections if s)


def _build_feedback_memory(state: dict) -> str:
    """Programmatic sections for the feedback agent."""
    sections = []
    sections.append(_build_slim_context(state))

    # Today's plan excerpt (timezone-aware)
    plan = state.get("high_level_plan", [])
    current_day = _get_local_trip_day(state) or state.get("current_trip_day") or 1
    today_plan = next((d for d in plan if d.get("day") == current_day), None)
    if today_plan:
        activities = [a.get("name", "?") for a in today_plan.get("key_activities", [])]
        today_lines = [
            "## Today's Plan [auto-refreshed]",
            f"Day {current_day}: {today_plan.get('city', '?')} — \"{today_plan.get('theme', '?')}\"",
        ]
        if activities:
            today_lines.append(f"Activities: {', '.join(activities)}")
        sections.append("\n".join(today_lines))

    # Full feedback history
    feedback_log = state.get("feedback_log", [])
    if feedback_log:
        fb_lines = [
            "## Feedback History [auto-refreshed]",
            "| Day | Highlight | Energy | Discoveries | Adjustments |",
            "|-----|-----------|--------|-------------|-------------|",
        ]
        for entry in feedback_log:
            day = entry.get("day", "?")
            highlight = entry.get("highlight", "") or ""
            energy = entry.get("energy_level", "?")
            discoveries = ", ".join(entry.get("discoveries", [])[:2])
            adjustments = ", ".join(entry.get("adjustments_made", [])[:2])
            fb_lines.append(f"| {day} | {highlight} | {energy} | {discoveries} | {adjustments} |")
        sections.append("\n".join(fb_lines))

    return "\n\n".join(s for s in sections if s)


def _build_cost_memory(state: dict) -> str:
    """Programmatic sections for the cost agent."""
    sections = []

    # Context with budget and currency
    sections.append(_build_context(state, COST_FIELDS))

    # Budget overview
    cost_tracker = state.get("cost_tracker", {})
    if cost_tracker.get("totals"):
        sections.append(build_cost_knowledge(state))

    # Spending by category
    by_cat = cost_tracker.get("by_category", {})
    if by_cat:
        cat_lines = ["## Spending by Category [auto-refreshed]"]
        for cat, data in by_cat.items():
            spent = data.get("spent_usd", 0)
            cat_lines.append(f"- {cat.title()}: ${spent:,.0f}")
        sections.append("\n".join(cat_lines))

    # Spending by city
    by_city = cost_tracker.get("by_city", {})
    if by_city:
        city_lines = ["## Spending by City [auto-refreshed]"]
        for city, data in by_city.items():
            spent = data.get("spent_usd", 0)
            city_lines.append(f"- {city}: ${spent:,.0f}")
        sections.append("\n".join(city_lines))

    # Pricing benchmarks
    dest = state.get("destination", {})
    benchmarks = dest.get("daily_budget_benchmarks", {})
    if benchmarks:
        bm_lines = ["## Pricing Benchmarks [auto-refreshed]"]
        for level, amount in benchmarks.items():
            bm_lines.append(f"- {level.title()}: ~${amount}/day/person")
        sections.append("\n".join(bm_lines))

    # Savings tips
    tips = cost_tracker.get("savings_tips", [])
    if tips:
        tip_lines = ["## Savings Tips [auto-refreshed]"]
        for tip in tips[:5]:
            tip_lines.append(f"- {tip}")
        sections.append("\n".join(tip_lines))

    return "\n\n".join(s for s in sections if s)


# ─── Populate module-level builder dispatch ─────────────
_ALL_BUILDERS.update({
    "research": _build_research_memory,
    "planner": _build_planner_memory,
    "scheduler": _build_scheduler_memory,
    "prioritizer": _build_prioritizer_memory,
    "feedback": _build_feedback_memory,
    "cost": _build_cost_memory,
})


# ─── Timezone Helper ────────────────────────────────────


def _get_local_trip_day(state: dict) -> int | None:
    """Compute the current trip day using the destination's timezone.

    Falls back to state['current_trip_day'] if timezone unavailable.
    Uses zoneinfo.ZoneInfo with the time_zone field from destination.
    """
    dest = state.get("destination", {})
    dates = state.get("dates", {})
    tz_name = dest.get("time_zone")
    start_date = dates.get("start")
    if not tz_name or not start_date:
        return state.get("current_trip_day")
    try:
        from datetime import date, datetime, timezone
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
        now_local = datetime.now(tz).date()
        trip_start = date.fromisoformat(start_date)
        delta = (now_local - trip_start).days + 1  # day 1 = start date
        return max(1, delta) if delta >= 1 else None
    except Exception:
        return state.get("current_trip_day")


# ─── Memory Cleanup ─────────────────────────────────────

import shutil
from datetime import datetime as _dt, timezone as _tz


async def cleanup_trip_memory(trip_id: str, base_dir: str = "./data/trips") -> None:
    """Remove all memory files for a trip. Also cleans up associated lock entries."""
    folder = Path(base_dir) / trip_id
    if folder.is_dir():
        shutil.rmtree(folder)
    # Clean up lock entries to prevent unbounded growth
    keys_to_remove = [k for k in _memory_locks if k[0] == trip_id]
    for key in keys_to_remove:
        del _memory_locks[key]


async def cleanup_stale_trips(max_age_days: int = 90, base_dir: str = "./data/trips") -> list[str]:
    """Remove trip memory folders older than max_age_days. Returns list of cleaned trip IDs."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    cleaned: list[str] = []
    cutoff = _dt.now(_tz.utc).timestamp() - (max_age_days * 86400)
    for trip_folder in base.iterdir():
        if not trip_folder.is_dir():
            continue
        # Use most recent file modification time as activity indicator
        latest_mtime = max(
            (f.stat().st_mtime for f in trip_folder.iterdir() if f.is_file()),
            default=0,
        )
        if latest_mtime and latest_mtime < cutoff:
            await cleanup_trip_memory(trip_folder.name, base_dir)
            cleaned.append(trip_folder.name)
    return cleaned


# ─── Phase Detection ────────────────────────────────────


def _detect_current_phase(state: dict) -> tuple[int, str, str]:
    """Return (phase_number, phase_name, description)."""
    if not state.get("onboarding_complete"):
        return (1, "Dreaming & Anchoring", "Locking in destination, dates, budget, and travel style")
    if not state.get("research"):
        return (3, "Research", "Deep dive into what to do, eat, see in each city")
    if not state.get("high_level_plan"):
        if not state.get("priorities"):
            return (5, "Structuring", "Ready to build your itinerary — run /plan (priorities auto-assigned, or /priorities to set manually)")
        return (5, "Structuring", "Sequencing cities and days into a coherent itinerary")
    if not state.get("detailed_agenda"):
        return (7, "Pre-trip Planning", "Detailed agenda, booking deadlines, prep checklist")
    if state.get("feedback_log"):
        return (9, "Mid-trip Adaptation", "Plan meets reality — adjusting based on experience")
    return (8, "Day-to-day Execution", "Living the trip — what to do today")


# ─── Public Section Builders (importable by agents) ─────


def build_destination_intel(state: dict) -> str:
    dest = state.get("destination", {})

    rows = []
    if dest.get("language"):
        rows.append(("Language", dest["language"]))
    if dest.get("currency_code"):
        currency = f"{dest['currency_code']} ({dest.get('currency_symbol', '?')})"
        rate = dest.get("exchange_rate_to_usd")
        if rate:
            currency += f" · 1 USD ≈ {dest.get('currency_symbol', '')}{rate}"
        rows.append(("Currency", currency))
    if dest.get("tipping_culture"):
        rows.append(("Tipping", dest["tipping_culture"]))
    if dest.get("payment_norms"):
        rows.append(("Payment", dest["payment_norms"]))
    if dest.get("climate_type"):
        climate = dest["climate_type"]
        season = dest.get("current_season_notes")
        if season:
            climate += f" — {season}"
        rows.append(("Climate", climate))

    phrases = dest.get("useful_phrases", {})
    if phrases:
        phrase_parts = [f'"{v}" = {k}' for k, v in list(phrases.items())[:5]]
        rows.append(("Phrases", ", ".join(phrase_parts)))

    if not rows:
        return ""

    lines = ["## Destination Intel", "", "| Field | Value |", "|-------|-------|"]
    for field, value in rows:
        lines.append(f"| {field} | {value} |")

    return "\n".join(lines)


def build_research_findings(state: dict) -> str:
    research = state.get("research", {})
    if not research:
        return ""

    lines = ["## Research Findings"]
    cities = state.get("cities", [])
    city_names = [c.get("name", "") for c in cities]

    # Show research in city order
    for city_name in city_names:
        city_data = research.get(city_name)
        if not city_data:
            continue

        city_config = next((c for c in cities if c.get("name") == city_name), {})
        days = city_config.get("days", "?")

        places = city_data.get("places", [])
        food = city_data.get("food", [])
        activities = city_data.get("activities", [])
        hidden_gems = city_data.get("hidden_gems", [])

        total = sum(len(city_data.get(k, [])) for k in ("places", "activities", "food", "logistics", "tips", "hidden_gems"))

        top_places = ", ".join(p.get("name", "?") for p in places[:3])
        top_food = ", ".join(f.get("name", "?") for f in food[:3])

        detail = f"Items: {total}"
        if places:
            detail += f" · Places: {len(places)}"
            if top_places:
                detail += f" ({top_places})"
        if food:
            detail += f" · Food: {len(food)}"
            if top_food:
                detail += f" ({top_food})"
        if activities:
            detail += f" · Activities: {len(activities)}"
        if hidden_gems:
            detail += f" · Hidden gems: {len(hidden_gems)}"

        lines.append(f"\n### {city_name} ({days} days)")
        lines.append(detail)

    # Also show any researched cities not in the main route
    for city_name, city_data in research.items():
        if city_name not in city_names:
            total = sum(len(city_data.get(k, [])) for k in ("places", "activities", "food", "logistics", "tips", "hidden_gems"))
            lines.append(f"\n### {city_name}")
            lines.append(f"Items: {total}")

    return "\n".join(lines)


def build_priorities_knowledge(state: dict) -> str:
    priorities = state.get("priorities", {})
    if not priorities:
        return ""

    lines = ["## Priorities"]

    for city_name, items in priorities.items():
        lines.append(f"\n### {city_name}")
        lines.append("| Tier | Items |")
        lines.append("|------|-------|")

        tiers: dict[str, list[str]] = {}
        for item in items:
            tier = item.get("tier", "unknown")
            name = item.get("name", "?")
            tiers.setdefault(tier, []).append(name)

        tier_labels = {
            "must_do": "Must Do",
            "nice_to_have": "Nice to Have",
            "if_nearby": "If Nearby",
            "skip": "Skip",
        }
        tier_icons = {
            "must_do": "\U0001f534",
            "nice_to_have": "\U0001f7e1",
            "if_nearby": "\U0001f7e2",
            "skip": "\u26aa",
        }

        for tier_key in ("must_do", "nice_to_have", "if_nearby", "skip"):
            items_list = tiers.get(tier_key, [])
            if items_list:
                icon = tier_icons.get(tier_key, "")
                label = tier_labels.get(tier_key, tier_key)
                names = ", ".join(items_list[:5])
                if len(items_list) > 5:
                    names += f" +{len(items_list) - 5} more"
                lines.append(f"| {icon} {label} ({len(items_list)}) | {names} |")

    return "\n".join(lines)


def build_itinerary_knowledge(state: dict) -> str:
    plan = state.get("high_level_plan", [])
    if not plan:
        return ""

    lines = [
        "## Itinerary",
        "",
        "| Day | Date | City | Theme |",
        "|-----|------|------|-------|",
    ]

    for day_plan in plan:
        day = day_plan.get("day", "?")
        date = day_plan.get("date", "?")
        city = day_plan.get("city", "?")
        theme = day_plan.get("theme", "?")
        lines.append(f"| {day} | {date} | {city} | {theme} |")

    return "\n".join(lines)


def build_cost_knowledge(state: dict) -> str:
    cost_tracker = state.get("cost_tracker", {})
    totals = cost_tracker.get("totals", {})
    budget = state.get("budget", {})

    if not totals:
        return ""

    budget_total = budget.get("total_estimate_usd") or cost_tracker.get("budget_total_usd", 0)
    spent = totals.get("spent_usd", 0)
    daily_avg = totals.get("daily_avg_usd", 0)
    status = totals.get("status", "unknown")

    rows = []
    if budget_total:
        rows.append(("Budget", f"${budget_total:,.0f}"))
    rows.append(("Spent", f"${spent:,.0f}"))
    if daily_avg:
        rows.append(("Daily Avg", f"${daily_avg:,.0f}/day"))
    rows.append(("Status", status))

    lines = ["## Cost Knowledge", "", "| Metric | Value |", "|--------|-------|"]
    for field, value in rows:
        lines.append(f"| {field} | {value} |")

    return "\n".join(lines)


def build_booking_deadlines(state: dict) -> str:
    """Find items with advance_booking=True from research."""
    research = state.get("research", {})
    deadlines: list[str] = []

    for city_name, city_data in research.items():
        for category in ("places", "activities", "food"):
            for item in city_data.get(category, []):
                if item.get("advance_booking"):
                    name = item.get("name", "?")
                    lead = item.get("booking_lead_time", "book in advance")
                    deadlines.append(f"{name} ({city_name}): {lead}")

    if not deadlines:
        return ""

    lines = ["## Booking Deadlines"]
    for d in deadlines:
        lines.append(f"- {d}")

    return "\n".join(lines)


# ─── Private Section Builders (internal only) ───────────


def _build_trip_profile(state: dict) -> str:
    dest = state.get("destination", {})
    cities = state.get("cities", [])
    dates = state.get("dates", {})
    travelers = state.get("travelers", {})
    budget = state.get("budget", {})
    interests = state.get("interests", [])
    must_dos = state.get("must_dos", [])
    deal_breakers = state.get("deal_breakers", [])
    accommodation = state.get("accommodation_pref", "")
    transport = state.get("transport_pref", [])

    flag = dest.get("flag_emoji", "")
    country = dest.get("country", "Unknown")
    region = dest.get("region", "")

    route = " > ".join(
        f"{c.get('name', '?')} ({c.get('days', '?')}d)" for c in cities
    ) if cities else "Not set"

    date_str = ""
    if dates.get("start"):
        date_str = f"{dates['start']} to {dates.get('end', '?')} ({dates.get('total_days', '?')} days)"

    traveler_str = ""
    if travelers:
        traveler_str = f"{travelers.get('count', '?')} {travelers.get('type', '')}"

    budget_str = ""
    if budget:
        parts = [budget.get("style", "")]
        total = budget.get("total_estimate_usd")
        if total:
            parts.append(f"${total:,.0f} total")
        budget_str = " · ".join(p for p in parts if p)

    rows = [
        ("Destination", f"{flag} {country}" + (f" ({region})" if region else "")),
        ("Route", route),
    ]
    if date_str:
        rows.append(("Dates", date_str))
    if traveler_str:
        rows.append(("Travelers", traveler_str))
    if budget_str:
        rows.append(("Budget", budget_str))
    if interests:
        rows.append(("Interests", ", ".join(interests)))
    if must_dos:
        rows.append(("Must-dos", ", ".join(must_dos)))
    if deal_breakers:
        rows.append(("Deal-breakers", ", ".join(deal_breakers)))
    if accommodation:
        rows.append(("Accommodation", accommodation))
    if transport:
        rows.append(("Transport", ", ".join(transport)))

    lines = ["## Trip Profile", "", "| Field | Value |", "|-------|-------|"]
    for field, value in rows:
        lines.append(f"| {field} | {value} |")

    return "\n".join(lines)


def _build_trip_learnings(state: dict) -> str:
    feedback_log = state.get("feedback_log", [])
    if not feedback_log:
        return ""

    lines = [
        "## Trip Learnings",
        "",
        "| Day | Highlight | Discovery |",
        "|-----|-----------|-----------|",
    ]

    for entry in feedback_log:
        day = entry.get("day", "?")
        highlight = entry.get("highlight", "") or ""
        discoveries = entry.get("discoveries", [])
        discovery = discoveries[0] if discoveries else ""
        lines.append(f"| {day} | {highlight} | {discovery} |")

    return "\n".join(lines)


# ─── Tactical Section Builders ──────────────────────────


def _build_current_phase(state: dict) -> str:
    phase_num, phase_name, desc = _detect_current_phase(state)
    return f"## Current Phase\n\n**Phase {phase_num}: {phase_name}**\n{desc}"


def _build_phase_checklist(state: dict) -> str:
    phase_num, _, _ = _detect_current_phase(state)

    phases = [
        (1, "Phase 1-2: Anchoring", state.get("onboarding_complete", False)),
        (3, "Phase 3: Research", bool(state.get("research"))),
        (4, "Phase 4: Priorities (optional)", bool(state.get("priorities"))),
        (5, "Phase 5: Structuring", bool(state.get("high_level_plan"))),
        (7, "Phase 6: Pre-trip prep", bool(state.get("detailed_agenda"))),
        (8, "Phase 7-10: On trip", bool(state.get("feedback_log"))),
    ]

    lines = ["## Completed"]
    for _, label, done in phases:
        mark = "x" if done else " "
        lines.append(f"- [{mark}] {label}")

    return "\n".join(lines)


def _build_next_actions(state: dict) -> str:
    phase_num, _, _ = _detect_current_phase(state)
    cities = state.get("cities", [])
    research = state.get("research", {})

    actions: list[str] = []

    if phase_num <= 1:
        actions = [
            "Complete trip setup — answer the onboarding questions",
            "Lock in destination, dates, and budget",
        ]
    elif phase_num == 3:
        unresearched = [c.get("name", "?") for c in cities if c.get("name") not in research]
        if unresearched:
            actions.append(f"Research cities: {', '.join(unresearched)} \u2192 /research all")
        else:
            actions.append("All cities researched \u2192 /plan to build your itinerary")
            actions.append("Optional: /priorities to fine-tune what matters most first")
    elif phase_num == 5:
        actions = [
            "Generate day-by-day itinerary \u2192 /plan",
            "Review and approve the plan",
        ]
    elif phase_num == 7:
        actions = [
            "Get detailed agenda for first days \u2192 /agenda",
            "Review booking deadlines",
            "Complete pre-trip checklist",
        ]
    elif phase_num == 8:
        actions = [
            "Check today's agenda \u2192 /agenda",
            "Log your day \u2192 /feedback",
            "Track spending \u2192 /costs today",
        ]
    elif phase_num == 9:
        actions = [
            "Review adjustments from feedback",
            "Check updated agenda \u2192 /agenda",
            "Log today \u2192 /feedback",
        ]

    if not actions:
        actions = ["You're all set! Enjoy your trip."]

    lines = ["## Next Actions"]
    for i, action in enumerate(actions, 1):
        lines.append(f"{i}. {action}")

    return "\n".join(lines)


def _build_pending_decisions(state: dict) -> str:
    """Extract open decisions from must-dos, research items that need timing, etc."""
    decisions: list[str] = []

    must_dos = state.get("must_dos", [])
    plan = state.get("high_level_plan", [])

    # If we have must-dos but no plan yet, they're pending decisions about timing
    if must_dos and not plan:
        for item in must_dos:
            decisions.append(f"When to schedule: {item}")

    if not decisions:
        return ""

    lines = ["## Pending Decisions"]
    for d in decisions:
        lines.append(f"- {d}")

    return "\n".join(lines)


def _build_active_adjustments(state: dict) -> str:
    """Extract adjustments from feedback log."""
    feedback_log = state.get("feedback_log", [])
    adjustments: list[str] = []

    for entry in feedback_log:
        for adj in entry.get("adjustments_made", []):
            day = entry.get("day", "?")
            adjustments.append(f"Day {day}: {adj}")

    if not adjustments:
        return ""

    lines = ["## Active Adjustments"]
    for a in adjustments:
        lines.append(f"- {a}")

    return "\n".join(lines)


def _build_pretrip_checklist(state: dict) -> str:
    """Destination-specific pre-trip checklist."""
    # Only show after planning phase
    if not state.get("high_level_plan"):
        return ""

    dest = state.get("destination", {})
    country = dest.get("country", "")

    items = [
        "Book flights",
        "Reserve accommodation",
        "Travel insurance",
        "Download offline maps",
    ]

    # Destination-specific additions
    transport = dest.get("common_intercity_transport", [])
    if any("rail" in t.lower() or "train" in t.lower() or "shinkansen" in t.lower() for t in transport):
        items.append("Buy rail pass (if applicable)")

    if dest.get("sim_connectivity"):
        items.append("Get eSIM/pocket WiFi")

    if dest.get("visa_requirements") and "visa" in dest["visa_requirements"].lower():
        items.append("Check visa requirements")

    if dest.get("health_advisories"):
        items.append("Check health advisories / vaccinations")

    lines = ["## Pre-trip Checklist"]
    for item in items:
        lines.append(f"- [ ] {item}")

    return "\n".join(lines)



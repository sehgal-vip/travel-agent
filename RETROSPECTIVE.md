# Travel Agent: Technical Documentation

**Project:** Multi-Agent Travel Planning Bot (Telegram + LangGraph + Anthropic Claude)
**Codebase:** ~11,300 LOC (src ~6,800 + tests ~4,500) across 35 Python files
**Last updated:** 2026-02-15

---

## Table of Contents

1. [Product Requirements Document (PRD)](#1-product-requirements-document)
2. [High-Level Design (HLD)](#2-high-level-design)
3. [Low-Level Design (LLD)](#3-low-level-design)
4. [System Architecture](#4-system-architecture)
5. [Agent Reference](#5-agent-reference)
6. [Testing Strategy](#6-testing-strategy)
7. [Retrospective](#7-retrospective)
8. [Appendices](#8-appendices)

---

## 1. Product Requirements Document

### 1.1 Problem Statement

A Telegram-based travel planning assistant using multiple LLM-powered agents coordinated by LangGraph. The system handles the full trip lifecycle: onboarding, destination research, priority ranking, itinerary generation, hour-by-hour scheduling, on-trip feedback, and budget tracking. It works for any destination worldwide, adapts to local currencies and customs, and maintains per-agent memory across sessions.

### 1.2 User Stories

**US-1: End-to-End Trip Planning**
> As a traveler, I can plan a complete trip through conversational Telegram messages, from "I want to go to Japan" through a detailed day-by-day agenda with times, costs, and local phrases.

Acceptance criteria:
- 13-field onboarding via natural conversation (`src/agents/onboarding.py:17-120`)
- Destination intelligence for any country (`src/agents/research.py:17-84`)
- City-level research with configurable depth (`src/agents/research.py:87-94`)
- Priority-ranked activities with 4 tiers (`src/agents/prioritizer.py:15-58`)
- Day-by-day itinerary with 10 planning principles (`src/agents/planner.py:15-77`)
- Hour-by-hour agenda with transport, costs, local phrases (`src/agents/scheduler.py:15-94`)

**US-2: Destination-Agnostic Design**
> As a traveler going anywhere in the world, the system adapts to my destination's currency, language, climate, and customs without hardcoded assumptions.

Acceptance criteria:
- No hardcoded country names, currencies, or cultural assumptions in prompts
- `DestinationIntel` TypedDict with 32 fields populated by research (`src/state.py:82-113`)
- Dynamic currency formatting with 16 zero-decimal currencies (`src/tools/currency.py:12-15`)
- Climate-adaptive scheduling (tropical, temperate, cold, arid, high altitude) (`src/agents/scheduler.py:29-33`)
- Automated test scanning for hardcoded values (`tests/test_destination_agnostic.py`)

**US-3: Per-Agent Memory with Cross-Session Continuity**
> As a returning user, agents remember qualitative insights from previous interactions so recommendations improve over time.

Acceptance criteria:
- 6 specialist agents each own a `.md` memory file per trip (`src/tools/trip_memory.py:1-12`)
- Programmatic sections rebuilt from state every invocation (`trip_memory.py:389-600`)
- LLM-generated notes accumulated across sessions via Haiku (`src/telegram/handlers.py:553-598`)
- Pinned/ephemeral note split with independent caps (10/25 lines) (`trip_memory.py:31-34`)
- Cross-agent `[shared]` tag propagation (`trip_memory.py:187-215`)
- Atomic writes via `tempfile.mkstemp` + `os.replace` (`trip_memory.py:59-74`)

**US-4: Multi-Trip Management**
> As a user planning multiple trips, I can create, switch between, archive, and share trips with travel companions.

Acceptance criteria:
- `/trip new`, `/trip switch <id>`, `/trip archive <id>` commands (`src/telegram/handlers.py:260-324`)
- `/join <id>` for shared trips with member access control (`handlers.py:210-257`)
- Rich trip listing with inline switch buttons (`handlers.py:402-472`)
- SQLite persistence with `TripRepository` (`src/db/persistence.py:17-146`)

**US-5: On-Trip Adaptation**
> As a traveler currently on my trip, the system shifts from planning mode to execution mode, tracking my daily experience and adapting the remaining itinerary.

Acceptance criteria:
- Orchestrator dual personality: TRAVEL STRATEGIST vs EXECUTIVE ASSISTANT (`src/agents/orchestrator.py:58-106`)
- End-of-day feedback collection via conversational check-in (`src/agents/feedback.py:16-86`)
- Budget tracking with natural-language spending parsing (`src/agents/cost.py:280-335`)
- Scheduler integrates recent feedback for adaptive scheduling (`src/agents/scheduler.py:149`)
- On-trip intent classification bias toward scheduler/feedback/cost (`orchestrator.py:259-265`)

**US-6: Write Safety Under Concurrency**
> As a system handling concurrent Telegram messages from multiple users, memory file writes must not corrupt data.

Acceptance criteria:
- Atomic writes via `tempfile.mkstemp` + `os.replace` (`trip_memory.py:59-74`)
- Async locks per `(trip_id, agent_name)` pair (`trip_memory.py:42-53`)
- MD5 idempotency check skips redundant writes (`trip_memory.py:90-94`)
- Handler is the sole writer; agents are read-only (`handlers.py:124`, `base.py:147-163`)

### 1.3 Non-Functional Requirements

| Requirement | Target | Implementation |
|-------------|--------|----------------|
| LLM response quality | Opus for complex reasoning (planner only) | `settings.py:28` |
| LLM cost efficiency | Sonnet for most agents, Haiku for notes | `settings.py:22-32`, `handlers.py:523` |
| Memory latency | < 5ms for file read + write | Sync reads, async locked writes |
| Input token safety | 160K conservative limit | `base.py:97`, truncation preserving notes |
| Notes cost | < $0.002 per invocation | Haiku with `max_tokens=300` |
| Notes growth | Bounded per agent | 10 pinned + 25 ephemeral lines |
| Write safety | No corruption on concurrent access | Atomic writes + async locking |
| Telegram message limit | 4096 chars per message | `formatters.py:split_message()` |

### 1.4 Out of Scope

- Multi-language UI (system operates in English; destination phrases are multilingual)
- Real-time collaboration (users share trip state, not live editing)
- Payment processing or booking integration
- Image/photo handling in Telegram
- Voice message support

---

## 2. High-Level Design

### 2.1 Architecture Overview

```
    Telegram Bot API (python-telegram-bot)
          |
          v
    handlers.py: process_message()          [SOLE WRITER]
          |
          +-- Load prior state from TripRepository (SQLite/aiosqlite)
          |
          +-- graph.ainvoke(input_state, config={thread_id: trip_id})
          |       |
          |       +-- StateGraph(TripState) ── hub-and-spoke topology
          |       |       |
          |       |       +-- orchestrator_node ── routes to exactly 1 specialist
          |       |       +-- specialist_node ── handles task, returns {response, state_updates}
          |       |       +-- END
          |       |
          |       v
          |   Result: {messages, state_updates, current_agent, ...}
          |
          +-- repo.merge_state(trip_id, state_updates)
          |
          +-- build_and_write_agent_memory(trip_id, agent, state)  [async, locked, atomic]
          |
          +-- _generate_and_append_notes(...)  [async, Haiku LLM, notes-eligible agents only]
          |
          +-- split_message(response) --> Telegram reply (4096 char chunks)
```

### 2.2 LangGraph Topology

**Defined in:** `src/graph.py:224-263`

The system uses a **hub-and-spoke** `StateGraph(TripState)` where every user message enters through the orchestrator, gets routed to exactly one specialist agent, and terminates. No multi-agent chains or loops -- each request is a single `orchestrator --> specialist --> END` pass.

```
                         Telegram (handlers.py)
                               |
                               v
                    +-------------------------+
                    |   StateGraph(TripState)  |
                    +-------------------------+
                               |
                    +-------------------------+
                    |   ORCHESTRATOR (entry)   |
                    |   .route(state, msg)     |
                    |   3-tier classification  |
                    +-----------+-------------+
                                |
              route_from_orchestrator(state["_next"])
                                |
        +-------+-------+------+------+--------+--------+--------+
        |       |       |      |      |        |        |        |
        v       v       v      v      v        v        v        v
    Onboard Research Librarian Plan  Priority Schedule Feedback  Cost
        |       |       |      |      |        |        |        |
        +-------+-------+------+------+--------+--------+--------+
                                |
                               END
```

**9 nodes**, all async. 1 conditional edge (orchestrator --> specialists via `route_from_orchestrator`). 8 fixed edges (specialists --> END).

**Graph construction** (`graph.py:224-263`):

```python
def build_graph(checkpointer=None) -> StateGraph:
    graph = StateGraph(TripState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("research", research_node)
    # ... 6 more specialist nodes
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", route_from_orchestrator, {...})
    for node_name in ("onboarding", "research", ...):
        graph.add_edge(node_name, END)
    return graph
```

**Routing function** (`graph.py:212-218`):

```python
def route_from_orchestrator(state: TripState) -> str:
    next_node = state.get("_next", END)
    if next_node == END or next_node == "orchestrator":
        return END
    valid = {"onboarding", "research", "librarian", "prioritizer",
             "planner", "scheduler", "feedback", "cost"}
    return next_node if next_node in valid else END
```

### 2.3 Routing: Three-Tier Decision

The orchestrator (`orchestrator.py:108-160`) uses a three-tier system:

**Tier 1: Command Detection** -- Slash commands map directly via `COMMAND_MAP` (15 entries, `orchestrator.py:15-31`):

| Command | Agent | Notes |
|---------|-------|-------|
| `/start` | onboarding | Redirects to welcome-back if already onboarded |
| `/research [city\|all]` | research | |
| `/library` | librarian | Markdown knowledge base sync |
| `/priorities` | prioritizer | Prerequisite: research |
| `/plan` | planner | Prerequisite: research |
| `/agenda` | scheduler | Prerequisite: high_level_plan |
| `/feedback`, `/adjust` | feedback | |
| `/costs [subcmd]` | cost | 6 sub-commands |
| `/status`, `/help` | orchestrator | Handled directly |
| `/trips`, `/mytrips` | orchestrator | Handled by handlers.py |

**Tier 2: Onboarding Guard** -- If `onboarding_complete` is false, all non-command messages go to onboarding (`orchestrator.py:156-157`).

**Tier 3: LLM Intent Classification** -- For natural language, Claude classifies intent against 9 agent descriptions with cue examples and on-trip bias (`orchestrator.py:227-297`).

### 2.4 Prerequisite Soft Guards

```python
# orchestrator.py:34-52
PREREQUISITES = {
    "prioritizer": [("research", "I'd need research findings...")],
    "planner":     [("research", "I'll need research on your cities...")],
    "scheduler":   [("high_level_plan", "I need a day-by-day plan...")],
}
```

Soft guards return a helpful offer rather than blocking. The user can always override.

### 2.5 Phase-Aware Behavior

The orchestrator shifts personality based on trip phase (`orchestrator.py:58-106`):

| Condition | Mode | Behavior |
|-----------|------|----------|
| No agenda or feedback | TRAVEL STRATEGIST | Brainstorming partner, reads cues, bounces ideas |
| Has agenda or feedback | EXECUTIVE ASSISTANT | Day runner, times, locations, routes, budget tracking |

Phase detection for memory/next-actions (`trip_memory.py:672-686`):

| Phase | Condition | Name |
|-------|-----------|------|
| 1 | Not onboarded | Dreaming & Anchoring |
| 3 | No research | Research |
| 5 | No plan (priorities optional) | Structuring |
| 7 | No detailed agenda | Pre-trip Planning |
| 8 | Has agenda, no feedback | Day-to-day Execution |
| 9 | Has feedback | Mid-trip Adaptation |

### 2.6 Data Flow: Memory System

```
    Agent.build_system_prompt(state)          [base.py:165-186]
         |
         +-- Is agent in MEMORY_AGENTS?
         |       |
         |   YES |                              NO
         |       v                               v
         | _ensure_memory_fresh(state)       get_destination_context(state)
         |   [READ-ONLY -- no disk write]        [thin context block]
         |       |
         |       +-- build_agent_memory_content(trip_id, agent, state)
         |       |       |
         |       |       +-- _ALL_BUILDERS[agent](state)  [programmatic sections]
         |       |       +-- read_shared_insights()       [cross-agent [shared] notes]
         |       |       +-- read_agent_notes()           [pinned + ephemeral notes]
         |       |       +-- Return combined content string
         |       |
         |       +-- _truncate_memory(memory, available_tokens)
         |       +-- Inject: "--- AGENT MEMORY ---\n{memory}\n--- END ---"
         |
         v
    System prompt = TONE_PREAMBLE + AGENT_PROMPT + MEMORY_OR_CONTEXT
```

### 2.7 Component Interaction Matrix

| Component | Reads | Writes | Dispatches To |
|-----------|-------|--------|---------------|
| `BaseAgent._ensure_memory_fresh()` | `build_agent_memory_content()` | **Nothing** | -- |
| `BaseAgent.build_system_prompt()` | `_ensure_memory_fresh()` or `get_destination_context()` | -- | `_truncate_memory()` |
| `handlers.process_message()` | Graph result, DB state | DB merge, memory files, notes | `build_and_write_agent_memory()`, `_generate_and_append_notes()` |
| `build_agent_memory_content()` | `read_agent_notes()`, `read_shared_insights()` | **Nothing** | Per-agent builder via `_ALL_BUILDERS` |
| `build_and_write_agent_memory()` | `build_agent_memory_content()` | `refresh_agent_memory()` | -- |
| `_generate_and_append_notes()` | `read_agent_notes()` (dedup) | `append_agent_notes()` | Haiku LLM |

---

## 3. Low-Level Design

### 3.1 State Schema (`src/state.py`, 476 LOC)

**Root type:** `TripState(TypedDict, total=False)` -- required by LangGraph. All fields optional.

**Enums** (9, all `str` mixin for JSON serialization):

| Enum | Values | Used By |
|------|--------|---------|
| `BudgetStyle` | backpacker, midrange, luxury, mixed | Onboarding, Cost |
| `Tier` | must_do, nice_to_have, if_nearby, skip | Prioritizer, Planner |
| `Vibe` | easy, active, cultural, foodie, adventurous, relaxed, mixed | Planner |
| `ItemStatus` | not_started, planned, done, skipped | Feedback |
| `EnergyLevel` | low, medium, high | Feedback |
| `SlotType` | activity, meal, transport, rest, free, special | Scheduler |
| `PricingTier` | budget, moderate, expensive, very_expensive | Research |
| `ClimateType` | tropical, temperate, cold, arid, high_altitude | Research, Scheduler |

**Nested TypedDicts** (18):

| TypedDict | Fields | Key Detail |
|-----------|--------|------------|
| `DestinationIntel` | 32 | Country-level intel. Populated once by research. |
| `CityConfig` | 6 | name, country, days, order, arrival/departure dates |
| `TravelerConfig` | 6 | count, type, ages, dietary, accessibility, nationalities |
| `BudgetConfig` | 5 | style, total_estimate_usd, daily_target, splurge_on, save_on |
| `DateConfig` | 3 | start, end, total_days |
| `ResearchItem` | 21 | id, name, name_local, category, coordinates, cost_local/usd, tags, ... |
| `CityResearch` | 7 | last_updated + 6 category lists (places, activities, food, logistics, tips, hidden_gems) |
| `PrioritizedItem` | 8 | item_id, name, tier, score, reason, user_override |
| `DayPlan` | 12 | day, date, city, theme, vibe, travel, key_activities, meals, special_moment, ... |
| `AgendaSlot` | 18 | time, end_time, type, name, address, coordinates, transport_from_prev, cost, tips, local_phrase, ... |
| `DetailedDay` | 8 | day, date, city, theme, slots, daily_cost_estimate, booking_alerts, quick_reference |
| `FeedbackEntry` | 18 | day, date, city, completed/skipped items, highlight, lowlight, energy, discoveries, adjustments, ... |
| `CostTracker` | 11 | budget totals, exchange rate, daily_log, by_category, by_city, savings_tips |
| `TransportDetail` | 6 | mode, line, duration_min, cost_local/usd, directions |
| `DailyCostEstimate` | 13 | food, activities, transport, accommodation, other (each local + usd) + total |
| `QuickReference` | 7 | hotel, emergency, transport_app, exchange_rate, weather, key_phrases, reminders |
| `LibraryConfig` | 5 | workspace_path, guide_written, synced_cities, feedback_days_written, last_synced |
| `Message` | 4 | role, content, timestamp, agent |

**Root TripState fields** (`state.py:417-477`):

| Domain | Fields | Populated By |
|--------|--------|-------------|
| Identity | `trip_id`, `trip_title`, `created_at`, `updated_at` | Onboarding |
| Destination | `destination: DestinationIntel` | Research (Mode 1) |
| Configuration | `cities`, `travelers`, `budget`, `dates`, `interests`, `must_dos`, `deal_breakers`, `accommodation_pref`, `transport_pref` | Onboarding |
| Research | `research: dict[str, CityResearch]` | Research (Mode 2) |
| Priorities | `priorities: dict[str, list[PrioritizedItem]]` | Prioritizer |
| Planning | `high_level_plan: list[DayPlan]`, `plan_status`, `plan_version` | Planner |
| Scheduling | `detailed_agenda: list[DetailedDay]` | Scheduler |
| Feedback | `feedback_log: list[FeedbackEntry]` | Feedback |
| Costs | `cost_tracker: CostTracker` | Cost |
| Library | `library: LibraryConfig` | Librarian |
| Conversation | `current_agent`, `conversation_history`, `onboarding_complete`, `onboarding_step`, `current_trip_day`, `agent_scratch` | All agents |
| Internal | `messages`, `_next`, `_user_message` | Graph nodes |

### 3.2 Agent Configuration (`src/agents/constants.py`, 12 LOC)

```python
MEMORY_AGENTS = {"research", "planner", "scheduler", "prioritizer", "feedback", "cost"}
AGENT_MAX_TOKENS: dict[str, int] = {
    "research": 16384, "planner": 16384, "scheduler": 8192, "prioritizer": 8192,
}
# All other agents default to 4096
NOTES_ELIGIBLE_AGENTS = {"research", "planner", "feedback"}
```

These constants are the single source of truth. They break the circular import between `base.py` and `trip_memory.py`.

### 3.3 Base Agent (`src/agents/base.py`, 212 LOC)

**`BaseAgent`** wraps `ChatAnthropic` with model selection and memory injection.

| Method | Purpose |
|--------|---------|
| `__init__()` | Selects model from `settings.AGENT_MODELS`, sets per-agent `max_tokens` from `AGENT_MAX_TOKENS` |
| `get_system_prompt(state)` | Override in subclasses. Returns agent-specific prompt. |
| `build_system_prompt(state)` | Combines `_TONE_PREAMBLE` + agent prompt + memory (or thin dest context) |
| `_ensure_memory_fresh(state)` | **Read-only.** Calls `build_agent_memory_content()`. No disk write. |
| `invoke(state, user_message)` | Single LLM call with system prompt + last 20 conversation history messages |

**Token budgeting** (`base.py:95-122`):

```python
_DEFAULT_INPUT_BUDGET = 160_000  # conservative for 200K context models

def _estimate_tokens(text: str) -> int:
    return len(text) // 3  # safe for CJK, emoji

def _truncate_memory(memory: str, available_tokens: int) -> str:
    # Preserves notes section when truncating
    # available = _DEFAULT_INPUT_BUDGET - base_prompt_tokens - output_budget
```

**Helper functions:**
- `get_destination_context(state)` (`base.py:19-67`): Thin context block for non-memory agents (orchestrator, onboarding, librarian).
- `format_money(amount, symbol, code)` (`base.py:70-78`): Handles 8 zero-decimal currencies.
- `add_to_conversation_history(state, role, content, agent)` (`base.py:81-92`): Timestamped append.

### 3.4 Trip Memory System (`src/tools/trip_memory.py`, 1150 LOC)

**Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `NOTES_MARKER` | `"## Agent Notes [accumulated] <!-- mem:notes -->"` | Section delimiter with HTML comment |
| `_OLD_NOTES_MARKER` | `"## Agent Notes [accumulated]"` | Backward compat for reading old files |
| `MAX_PINNED_LINES` | 10 | Durable preferences cap |
| `MAX_EPHEMERAL_LINES` | 25 | Session-specific insights cap |
| `SCHEDULER_FEEDBACK_ENTRIES` | 3 | Feedback lookback for scheduler |
| `MEMORY_FORMAT_VERSION` | `"1"` | Version header in memory files |
| `_MEMORY_SIZE_WARN_BYTES` | 50KB | Warning threshold for large files |

**Public API:**

| Function | Async? | Purpose |
|----------|--------|---------|
| `build_agent_memory_content(trip_id, agent, state)` | No | Build memory string without writing. Used by prompt builder and persistence. |
| `build_and_write_agent_memory(trip_id, agent, state)` | Yes | Build content + write to disk. Called by handler only. |
| `refresh_agent_memory(trip_id, agent, content)` | Yes | Async locked, atomic write, MD5 idempotency. |
| `read_agent_notes(trip_id, agent)` | No | Sync read. Pinned/ephemeral split with independent caps. |
| `append_agent_notes(trip_id, agent, new_notes)` | Yes | Async locked, atomic. Sanitizes marker strings. |
| `read_shared_insights(trip_id, exclude_agent)` | No | `[shared]`-tagged notes from other agents. Capped at 15 lines. |
| `get_memory_stats(trip_id)` | No | Size/token stats per agent for observability. |
| `cleanup_trip_memory(trip_id)` | Yes | Remove all files + clean lock entries. |
| `cleanup_stale_trips(max_age_days)` | Yes | Remove folders older than threshold. |

**Write safety infrastructure** (`trip_memory.py:42-74`):

```python
_memory_locks: dict[tuple[str, str], asyncio.Lock] = {}  # per (trip_id, agent_name)
_get_lock(trip_id, agent_name)  # lazy creation, atomic in single-threaded asyncio

def _atomic_write(filepath, content):
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    # write + fsync + os.replace (atomic on POSIX)
    # cleanup tmp on failure
```

**Per-agent builder dispatch** (`trip_memory.py:218-219, 593-600`):

```python
_ALL_BUILDERS: dict[str, callable] = {
    "research": _build_research_memory,     # slim context + dest intel + research status + findings
    "planner": _build_planner_memory,       # extended context + priorities + food highlights + itinerary
    "scheduler": _build_scheduler_memory,   # slim context + itinerary + practical info + deadlines + feedback
    "prioritizer": _build_prioritizer_memory, # extended context + research status + findings
    "feedback": _build_feedback_memory,     # slim context + today's plan + feedback history
    "cost": _build_cost_memory,             # cost context + budget + spending + benchmarks + tips
}
```

Validation (`trip_memory.py:224-241`): `_validate_agent_lists()` raises `RuntimeError` at first invocation if `_ALL_BUILDERS` keys don't match `MEMORY_AGENTS` or `NOTES_ELIGIBLE_AGENTS` is not a subset.

**Context field sets** (`trip_memory.py:308-310`):

```python
SLIM_FIELDS = {"destination", "dates", "cities", "travelers", "interests"}
EXTENDED_FIELDS = SLIM_FIELDS | {"must_dos", "budget"}
COST_FIELDS = EXTENDED_FIELDS | {"currency"}
```

**Memory file structure:**

```markdown
<!-- memory_format: 1 -->
# {Agent} Memory -- {Trip Title}

## Trip Context [auto-refreshed]
Destination: JP Japan | Dates: 2026-04-01 to 2026-04-14 (14 days)
Route: Tokyo (5d) --> Kyoto (4d) --> Osaka (3d)

## {Agent-Specific Sections} [auto-refreshed]
...

## Cross-Agent Insights [auto-refreshed]
[research] - [shared] User strongly prefers street food
[feedback] - [shared] User energy drops after 3PM

## Agent Notes [accumulated] <!-- mem:notes -->
- [pinned] User is vegetarian
- Fushimi Inari research was very detailed
```

Three zones: (1) Programmatic `[auto-refreshed]` -- rebuilt from state every invocation. (2) Cross-agent insights -- `[shared]` tags from other agents. (3) LLM-generated `[accumulated]` -- preserved across rebuilds.

### 3.5 Persistence (`src/db/persistence.py`, 146 LOC)

**`TripRepository`** -- async CRUD backed by SQLAlchemy + aiosqlite.

| Method | Purpose |
|--------|---------|
| `create_trip(trip_id, user_id, state)` | Insert new trip |
| `get_trip(trip_id)` | Lookup by PK |
| `list_trips(user_id)` | All trips where user is owner or member |
| `merge_state(trip_id, updates)` | JSON merge (not replace) into existing state |
| `archive_trip(trip_id)` | Soft delete |
| `add_member(trip_id, user_id)` | Idempotent join |
| `is_member(trip_id, user_id)` | Owner or member check |

**Models** (`src/db/models.py`, 52 LOC):
- `Trip`: `trip_id` (PK), `user_id`, `destination_country`, `state_json` (TEXT), `archived` (bool), `created_at`, `updated_at`
- `TripMember`: `trip_id` + `user_id` (unique constraint)

### 3.6 Settings (`src/config/settings.py`, 39 LOC)

Pydantic `BaseSettings` loaded from `.env`:

```python
AGENT_MODELS: dict[str, str] = {
    "planner": "claude-opus-4-6",          # Only agent on Opus
    "orchestrator": "claude-sonnet-4-5-20250929",
    "onboarding": "claude-sonnet-4-5-20250929",
    # ... all others: Sonnet 4.5
}
```

Notes generation uses `claude-haiku-4-5-20251001` (`handlers.py:523`).

### 3.7 Web Search (`src/tools/web_search.py`, 86 LOC)

Tavily wrapper with two query strategies:
- `search_destination_intel(country, dates)`: 6 queries (essentials, safety, visa/currency, culture, transport, weather)
- `search_city(city, country, interests, type)`: 5-6 queries (top things, food, hidden gems, logistics, day trips, + interest-specific)

Each query: `max_results=3`, `search_depth="advanced"`, 30s timeout.

### 3.8 Currency Tools (`src/tools/currency.py`, 78 LOC)

- 16 zero-decimal currencies in `ZERO_DECIMAL_CURRENCIES` frozenset
- `format_local()`, `format_usd()`, `format_dual()` for display
- `convert()` / `convert_to_local()` using destination exchange rate
- `refresh_exchange_rate()` via open.er-api.com (async, with timeout)

### 3.9 Markdown Library (`src/tools/markdown_sync.py`, 508 LOC)

`MarkdownLibrary` creates a structured folder of `.md` files:

```
trips/Japan_2026/
+-- INDEX.md
+-- destination_guide.md
+-- cities/
|   +-- Tokyo.md
|   +-- Kyoto.md
+-- priorities.md
+-- itinerary.md
+-- feedback/
    +-- day_1.md
```

Methods: `create_workspace`, `write_destination_guide`, `write_city_research`, `write_priorities`, `write_itinerary`, `write_feedback`, `write_index`.

### 3.10 Telegram Formatters (`src/telegram/formatters.py`, 250 LOC)

- `format_agenda_slot(slot, dest)`: Rich single-slot display with transport, cost, local phrases
- `format_budget_report(state)`: Full budget dashboard
- `split_message(text, limit=4096)`: Split long responses at paragraph/line boundaries

### 3.11 Telegram Handlers (`src/telegram/handlers.py`, 628 LOC)

**`process_message()`** (`handlers.py:31-207`) -- the central bridge between Telegram and LangGraph:

1. Get user ID, load active trip
2. Handle special commands (`/join`, `/trip`, `/trips`)
3. Start background typing indicator
4. Load prior state from `TripRepository` as fallback
5. `graph.ainvoke(input_state, config={thread_id: trip_id})`
6. Save state to DB via `repo.merge_state()`
7. Write agent memory via `build_and_write_agent_memory()`
8. Generate notes if eligible via `_generate_and_append_notes()`
9. Handle onboarding completion (new trip ID, auto-research)
10. Split and send response

**Notes generation** (`handlers.py:518-609`):

| Function | Purpose |
|----------|---------|
| `_should_generate_notes(result, agent)` | Check `NOTES_ELIGIBLE_AGENTS` + meaningful state updates |
| `_extract_notes_context(response, max_chars=1500)` | First line + tail (not blind head truncation) |
| `_generate_and_append_notes(trip_id, agent, result, append_fn)` | Haiku LLM with structured bullets, `[pinned]`/`[shared]` tags, dedup via `<existing_notes>` XML |
| `_log_notes_failure(agent)` | Log first 3, then every 10th (with `exc_info=True`) |

---

## 4. System Architecture

### 4.1 Node Execution Flow

All specialist nodes use `_specialist_node(agent_name, state)` (`graph.py:129-183`):

1. Lazy-load agent singleton via `_get_agent(name)` (`graph.py:26-50`)
2. Extract `_user_message` from state
3. Call `agent.handle(state, user_msg)` --> `{response, state_updates}`
4. Prepend `_routing_echo` if present (orchestrator's preamble)
5. Append to `conversation_history` with timestamp + agent attribution
6. Return `{messages, conversation_history, current_agent, **state_updates, _next: END}`

The orchestrator node is special (`graph.py:56-95`) -- it can terminate directly for `/status`, `/help`, welcome-back, or set `_next` to route to a specialist.

### 4.2 Agent Lazy Loading

Agents are imported lazily in `_get_agent()` (`graph.py:26-50`) to avoid circular imports at module level. Only `OrchestratorAgent` and `OnboardingAgent` are imported eagerly (lines 12-13) since they're needed on every request.

### 4.3 State Flow

```
User message
    |
    v
graph.ainvoke({**prior_state, messages: [{role: "user", content: msg}]})
    |
    v
orchestrator_node: reads messages[-1], calls route()
    |-- Direct response (status/help/welcome-back) --> return {_next: END}
    |-- Route to specialist --> return {_next: "agent_name", _user_message: msg}
    |
    v
specialist_node: reads _user_message, calls agent.handle()
    |-- Returns {response, state_updates}
    |
    v
Graph merges state_updates into TripState, returns result
    |
    v
handlers.py: persist state, write memory, generate notes, reply
```

### 4.4 Per-Agent Memory Layout

```
data/trips/{trip_id}/
+-- research.md      <- slim context + dest intel + research progress + findings
+-- planner.md       <- extended context + priorities + food highlights + itinerary
+-- scheduler.md     <- slim context + itinerary + practical info + deadlines + feedback
+-- prioritizer.md   <- extended context + research progress + findings
+-- feedback.md      <- slim context + today's plan + feedback history
+-- cost.md          <- cost context + budget + spending + benchmarks + tips
```

Not in `MEMORY_AGENTS` (no file written): orchestrator, onboarding, librarian. These use thin `get_destination_context()` instead.

---

## 5. Agent Reference

### 5.1 Agent Summary Table

| Agent | File | LOC | Model | max_tokens | Memory? | Notes? |
|-------|------|-----|-------|-----------|---------|--------|
| Orchestrator | `orchestrator.py` | 469 | Sonnet 4.5 | 4096 | No | No |
| Onboarding | `onboarding.py` | 387 | Sonnet 4.5 | 4096 | No | No |
| Research | `research.py` | 423 | Sonnet 4.5 | 16384 | Yes | Yes |
| Planner | `planner.py` | 376 | **Opus 4.6** | 16384 | Yes | Yes |
| Prioritizer | `prioritizer.py` | 263 | Sonnet 4.5 | 8192 | Yes | No |
| Scheduler | `scheduler.py` | 282 | Sonnet 4.5 | 8192 | Yes | No |
| Feedback | `feedback.py` | 193 | Sonnet 4.5 | 4096 | Yes | Yes |
| Cost | `cost.py` | 353 | Sonnet 4.5 | 4096 | Yes | No |
| Librarian | `librarian.py` | 134 | Sonnet 4.5 | 4096 | No | No |

### 5.2 Orchestrator (`src/agents/orchestrator.py`)

**Responsibilities:** Route every message, handle meta-commands (`/status`, `/help`), welcome-back, phase-aware personality switching.

**Key methods:**
- `route(state, message)` -- 3-tier routing (command --> onboarding guard --> LLM classification)
- `_classify_intent(state, message)` -- LLM-based intent classification with on-trip bias
- `_welcome_back_with_ideas(state)` -- Contextual welcome using `generate_status(state)` directly (no file I/O)
- `_check_prerequisites(state, target)` -- Soft guard returning helpful suggestions
- `_get_state_summary(state)` -- Enriched routing context capped at 12 lines

**Module-level functions:**
- `generate_status(state)` (`orchestrator.py:382-434`) -- Dynamic status dashboard
- `generate_help()` (`orchestrator.py:437-469`) -- Command reference text

### 5.3 Onboarding (`src/agents/onboarding.py`)

**Responsibilities:** Collect 13 trip parameters through conversational flow across 11 steps.

**Steps** (`onboarding.py:86-98`): destination --> cities --> dates --> days per city --> travelers --> budget --> interests --> dietary/accessibility --> must-dos/deal-breakers --> accommodation --> transport --> confirmation.

**Key methods:**
- `handle(state, user_message)` -- Builds context of filled/missing fields, invokes LLM, checks for confirmed JSON
- `_extract_config(text)` -- Parse JSON from `` ```json `` blocks
- `_build_state_from_config(config, state)` -- Initializes all state containers (research, priorities, plan, cost_tracker, library, etc.)
- `_estimate_step(state, message, current_step)` -- Jumps to first unfilled step
- `_generate_trip_title(state_updates)` -- Witty title via LLM (max 6 words), with deterministic fallback

### 5.4 Research (`src/agents/research.py`)

**Two modes:**
1. **Destination Intelligence** (`_research_destination`, line 149): 32-field `DestinationIntel` from 6 web search queries. Runs once when destination is set.
2. **City Research** (`_research_city`, line 235): 6-category research items (places, activities, food, logistics, tips, hidden_gems). Depth scales by city days:

| Days | Places | Activities | Food |
|------|--------|-----------|------|
| 4+ | 20 | 15 | 25 |
| 2-3 | 10 | 8 | 15 |
| 1 | 5 | 5 | 8 |
| <1 | 3 | 2 | 5 |

**Key methods:**
- `_research_all_cities(state)` (`research.py:330-379`) -- Chains destination intel then all cities sequentially
- `_merge_research(existing, new)` (`research.py:381-392`) -- Deduplicates by name (case-insensitive)
- `_parse_json(text)` (`research.py:394-423`) -- Tries direct parse, markdown block, first-`{`-to-last-`}` with diagnostic logging

Both modes generate a conversational abstract via a second LLM call (not just raw JSON).

### 5.5 Planner (`src/agents/planner.py`)

**The only agent using Opus 4.6** (`settings.py:28`).

**10 planning principles** in system prompt: pacing, travel days, city flow, priority cascade, meal planning, golden hours, traveler moments, destination-adaptive, first/last day, day-of-week.

**Key methods:**
- `handle(state, user_message)` -- Detects adjust requests vs new plans. Auto-prioritizes if no explicit priorities.
- `_auto_prioritize(state)` (`planner.py:225-273`) -- Heuristic scoring with 30% must-do cap:
  - User's explicit must-dos: score 95
  - Baseline: 50
  - Suitability (1-5): -20 to +20
  - Tag overlap with interests: +8 each
  - Food category: +5
  - Advance booking: +5
- `_score_to_tier(score)` -- >= 75: must_do, >= 55: nice_to_have, >= 35: if_nearby, else skip
- `_adjust_plan(state, message, current_plan)` -- Modify existing plan based on user feedback

### 5.6 Scheduler (`src/agents/scheduler.py`)

**Produces 2-day rolling window** of `DetailedDay` objects with minute-level `AgendaSlot`s.

**Key features:**
- Climate-adaptive energy curves (tropical: early start + siesta, cold: prioritize daylight)
- Per-slot: address, transport from previous, cost in dual currency, local phrase, rain backup
- Booking alerts: book NOW, book today, walk-in fine
- Quick reference per day: hotel, emergency, transport app, exchange rate, key phrases
- Integrates last 2 feedback entries for adaptive scheduling (`scheduler.py:149`)
- Merges into existing `detailed_agenda` (replaces same-day entries, `scheduler.py:179-186`)

### 5.7 Prioritizer (`src/agents/prioritizer.py`)

**6 weighted scoring criteria** in system prompt:
1. Uniqueness (30%)
2. Match to interests (25%)
3. Reviews & reputation (15%)
4. Time-value ratio (15%)
5. Traveler-type suitability (10%)
6. Logistics (5%)

**Rules:** 30% must-do cap per city, 2-3 food items always must-do, tourist traps to skip, cross-city redundancy flagging.

**User override preservation** (`prioritizer.py:153-166`): When re-prioritizing, items with `user_override: True` keep their user-set tier.

### 5.8 Feedback (`src/agents/feedback.py`)

**Conversational end-of-day check-in** producing `FeedbackEntry` (18 fields).

Captures: completed/skipped items, highlight/lowlight, energy level, food rating, budget status, weather, discoveries, preference shifts, destination feedback (transport, language, safety, accommodation, food quality, connectivity), adjustments, sentiment, actual spend.

**Adjustment triggers:** "exhausted" --> lighten tomorrow, "loved X" --> add more, "overspent" --> flag + suggest cheaper, "weather was bad" --> plan backups.

### 5.9 Cost (`src/agents/cost.py`)

**6 sub-commands:**

| Sub-command | Method | Purpose |
|-------------|--------|---------|
| `/costs` | `_full_report()` | Full budget dashboard via `format_budget_report()` |
| `/costs today` | `_today_report()` | Today's estimated vs actual |
| `/costs save` | `_savings_tips()` | Destination-specific tips via LLM (cached) |
| `/costs convert X` | `_convert()` | Quick currency conversion |
| `/costs {city}` | `_city_report()` | City breakdown |
| `/costs {category}` | `_category_report()` | Category breakdown |

**Natural language spending** (`cost.py:280-335`): LLM parses "spent 3000 yen on ramen" into `{amount_local, amount_usd, category, description}`, updates tracker totals and by_category.

### 5.10 Librarian (`src/agents/librarian.py`)

**Syncs trip data to local markdown files** using `MarkdownLibrary`.

Sync order: workspace folder --> destination guide --> city research (with priorities) --> priorities file --> itinerary --> feedback entries --> INDEX.md.

Tracks sync state in `LibraryConfig` to avoid redundant writes.

---

## 6. Testing Strategy

### 6.1 Test Architecture

| Test File | LOC | Tests | Coverage |
|-----------|-----|-------|----------|
| `test_functional_flows.py` | 1964 | ~80 | All agent routing and flow scenarios |
| `test_trip_memory.py` | 716 | ~63 | Memory system: builders, notes, file I/O, prompts |
| `test_graph_integration.py` | 420 | ~20 | Full graph execution with mocked LLMs |
| `test_orchestrator.py` | 194 | ~12 | Routing, prerequisites, status, help |
| `test_persistence.py` | 190 | ~10 | DB CRUD, merge_state, add_member |
| `test_destination_agnostic.py` | 153 | ~8 | No hardcoded country/currency in source |
| `test_mytrips.py` | 107 | ~6 | Trip listing, switching, join |
| `test_graph.py` | 94 | ~5 | Graph compilation, node count |
| `test_onboarding.py` | 94 | ~5 | Onboarding step progression |
| `conftest.py` | 416 | -- | 7 fixtures |

### 6.2 Fixture Strategy (`tests/conftest.py`)

Progressive fixture chain mirroring real trip progression:

```
japan_state
    |-- Onboarding complete, destination set, no research
    |
    +-- researched_japan
    |       |-- Research data for Tokyo + Kyoto
    |       |
    |       +-- prioritized_japan
    |               |-- Priorities for Tokyo (8 items across tiers)
    |               |
    |               +-- planned_japan
    |                       |-- 5-day itinerary
    |                       |
    |                       +-- on_trip_japan
    |                               |-- Detailed agenda (Day 1)
    |                               |-- Feedback log (Day 1)

morocco_state -- Different continent, currency (MAD), climate (arid)
paris_state -- Single-city trip (3 days)
long_trip_state -- 28-day 6-city European tour
large_research_state -- Japan with 60+ items per city
```

### 6.3 Test Classes in `test_trip_memory.py`

| Class | Tests | Coverage |
|-------|-------|---------|
| `TestAgentMemorySections` | 8 | Each agent's builder via `build_agent_memory_content` |
| `TestPhaseDetection` | 9 | Phase detection, next actions, checklists |
| `TestBuildSections` | 9 | Individual section builders (profile, intel, itinerary, etc.) |
| `TestFileIO` | 13 | Async file creation, overwrite, idempotency, notes, legacy API removal |
| `TestNotesAccumulation` | 3 | Notes persistence across refreshes, pinned/ephemeral caps |
| `TestSystemPromptIntegration` | 7 | Prompt injection for memory agents, orchestrator exclusion |
| `TestWelcomeBack` | 4 | Welcome-back flow, title/flag display |
| `TestPrerequisiteGuards` | 4 | Soft guards with helpful messages |
| `TestPhasePrompt` | 5 | Phase-appropriate prompts |

### 6.4 What Is NOT Tested

| Area | Reason |
|------|--------|
| Live Haiku notes generation | Requires API key; wrapped in try/except; cost-prohibitive for CI |
| Actual memory file sizes | Depends on real trip state; unit tests verify structure |
| `_parse_json` truncation | Would require mocking truncated LLM response |
| Concurrent async lock contention | Standard `asyncio.Lock`; would need multi-task harness |
| Cross-agent `[shared]` end-to-end | Requires multi-agent invocation; unit tested via `read_shared_insights` |
| Token budget truncation in production | Requires 50KB+ files; `large_research_state` fixture available |

---

## 7. Retrospective

### 7.1 What Went Well

**Per-agent memory architecture is clean.** The separation between programmatic sections (rebuilt from state) and accumulated notes (preserved across rebuilds) avoids the stale-data problem. Each agent sees only relevant context, and the `_ALL_BUILDERS` dispatch dict makes the mapping explicit and validated.

**Centralizing builders eliminated dual-path bugs.** Moving all memory building to `trip_memory.py` and making agents read-only (they build memory content for prompts but never write to disk) means there's exactly one code path. The handler is the sole writer.

**Structured bullet format for notes was a reliability win.** Requiring `- ` prefix and using `- NONE` as a sentinel eliminated false positives from Haiku responses like "No new patterns observed." -- those lines don't start with `- ` and are silently ignored.

**`constants.py` cleanly broke the circular import.** The function-level imports between `base.py` and `trip_memory.py` were fragile. Extracting `MEMORY_AGENTS`, `AGENT_MAX_TOKENS`, and `NOTES_ELIGIBLE_AGENTS` to a dedicated module eliminated the risk.

**Auto-prioritization with 30% cap** (`planner.py:225-273`). The heuristic scoring + cap means `/plan` works without requiring `/priorities` first, lowering the barrier to getting an itinerary. Explicit priorities override the heuristic when present.

**Destination-agnostic design is enforced by tests.** `test_destination_agnostic.py` scans source files for hardcoded country names, currencies, and cultural assumptions, preventing regression.

### 7.2 What Was Tricky

**Async/sync boundary in memory system.** `build_agent_memory_content` must be sync (called from `build_system_prompt` in the agent's hot path), while `refresh_agent_memory` and `append_agent_notes` must be async (for locking). The boundary is clean -- sync reads state, async writes files -- but required careful `await` propagation through `handlers.py`.

**Eventual consistency of notes.** The agent sees notes from the _previous_ invocation, not the current one (handler writes notes _after_ the agent responds). For rapid multi-message conversations, a note might take one extra round-trip to appear. This is documented as a conscious design tradeoff (`trip_memory.py:252-255`).

**Pinned vs ephemeral note caps.** Separate caps (10 pinned, 25 ephemeral) changed the behavioral contract from the original flat 30-line cap. Tests needed updating to reflect the new split.

**Cross-agent shared insights ordering.** Agent X's `[shared]` notes are written _after_ X responds. So if the graph routes X then Y in sequence, Y won't see X's shared notes until the next request. This is eventual consistency by design (`trip_memory.py:193-196`).

**JSON parsing robustness.** LLM responses may include markdown code blocks, preamble text, or incomplete JSON. All agents share a 3-strategy parser: direct parse, markdown extraction, first-`{`/`[` to last-`}`/`]`. Research agent additionally logs char counts and token estimates for calibration.

### 7.3 Architecture Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Hub-and-spoke (no multi-agent chains) | Simpler debugging, predictable latency | Can't chain research --> plan in one pass |
| Opus only for planner | Complex multi-day reasoning needs it | Higher cost per plan generation |
| Haiku for notes | Cost efficiency (~$0.001/call) | Lower quality notes (acceptable) |
| Per-agent `.md` files (not DB) | Human-readable, easy debugging | No ACID guarantees (mitigated by atomic writes) |
| Handler as sole writer | No concurrent write conflicts from agents | Agent can't persist its own scratch state |
| Sync reads in build_agent_memory_content | Avoids async in prompt building hot path | Blocking I/O (acceptable for <50KB files) |
| 160K conservative input budget | Safe margin for 200K context models | Wastes ~40K tokens of capacity |

### 7.4 Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| New agent without `_ALL_BUILDERS` entry | `_validate_agent_lists()` raises `RuntimeError` at first invocation |
| Notes marker in LLM-generated content | HTML comment suffix; `append_agent_notes` sanitizes markers |
| File system I/O failures | Atomic writes + try/except; agent falls back to destination context |
| Haiku notes adds latency | Post-response; `max_tokens=300`; failure logging throttled |
| Lock dict grows unboundedly | `cleanup_trip_memory` cleans lock entries; `cleanup_stale_trips` for old trips |
| Token budget too conservative | `160K` for 200K models; `//3` estimation safe but wasteful for ASCII |
| Rapid messages miss notes | Eventual consistency by design; appears on next invocation |
| Research JSON truncation | `max_tokens=16384` (4x headroom for largest observed city research) |

### 7.5 v2 Architecture (February 2026)

Major rewrite implementing `langgraph.types.Command`-based routing, loopback edges, and 12 product optimizations across 5 phases. **32 files** modified/created (10 new, 22 modified). Test count: 318 → 339.

#### Graph Topology Changes

| Feature | v1 | v2 |
|---------|----|----|
| Routing | Conditional edges | `langgraph.types.Command` (requires >=0.2.28) |
| Topology | Hub-and-spoke, single pass | Hub-and-spoke with loopback edges |
| Loopback agents | None | research, planner, feedback (can return to orchestrator) |
| Error handling | Try/except in specialist node | Dedicated `error_handler` node |
| Librarian | Full graph agent | Demoted to standalone tool (`src/tools/library_sync.py`) |
| Slash commands | Orchestrator classifies | Pre-dispatch in `handlers.py` via `COMMAND_DISPATCH` |
| Graph control keys | `_next`, `_user_message` | + `_awaiting_input`, `_callback`, `_delegate_to`, `_chain`, `_routing_echo`, `_error_agent`, `_error_context`, `_loopback_depth` |

**Loopback routing priority** (in `orchestrator.py`):
1. `_awaiting_input` → route directly (no LLM)
2. `_callback` → return to delegating agent
3. `_delegate_to` → forward to delegate
4. `_chain` → pop next agent
5. Normal command/LLM routing
6. Depth enforcement: `_loopback_depth > 5` → force END

#### New Files (v2)

| File | LOC | Phase | Purpose |
|------|-----|-------|---------|
| `src/tools/library_sync.py` | ~80 | 1 | Standalone library sync (no LLM) |
| `src/db/user_profile.py` | ~100 | 3 | Cross-trip user preferences |
| `src/tools/trip_synthesis.py` | ~115 | 4 | One-screen trip summary |
| `src/tools/drift_detector.py` | ~95 | 4 | Feedback trend analysis |
| `src/telegram/nudge.py` | ~180 | 4 | Proactive nudge system |
| `src/tools/weather.py` | ~80 | 5 | Weather API (graceful degradation) |
| `src/tools/transit.py` | ~80 | 5 | Transit time estimation |
| `tests/test_drift_detector.py` | ~80 | 4 | 12 drift detector tests |
| `tests/test_nudge.py` | ~100 | 4 | 9 nudge system tests |
| `tests/test_library_sync.py` | varies | 1 | Library sync tests |
| `tests/test_loopback.py` | varies | 1 | Loopback routing tests |

#### Product Optimizations Implemented

| # | Optimization | Phase | Key Changes |
|---|-------------|-------|-------------|
| 1 | Proactive nudges | 4 | `nudge.py`: booking deadlines, stale research, budget drift, must-do crunch, stale trip |
| 2 | Progressive profiling | 3 | 3-field minimum viable onboarding (destination, dates, travelers); `onboarding_depth` field |
| 3 | Cross-trip learning | 3 | `UserProfileModel` with dietary, pace, budget_tendency, interests_history, visited_countries |
| 4 | Planner negotiation | 2 | Conflict detection (>4 must-dos/day); `_awaiting_input` loopback for user resolution |
| 5 | Conversation compression | 3 | `_compress_history()`: extractive summary of older messages, keep last 15 verbatim |
| 6 | Spending inference | 2 | Feedback agent cross-references completed activities with research cost data |
| 7 | Trip synthesis | 4 | `/summary` command: route, must-dos, budget, booking alerts — no LLM call |
| 8 | Research freshness | 3 | `_check_freshness()`: flags stale (>14d) research for trips within 30 days |
| 9 | Weather API | 5 | WeatherAPI.com integration with graceful degradation (returns None if no key) |
| 10 | Drift detection | 4 | Energy/budget/food/schedule/preference/sentiment trend analysis across feedback |
| 11 | Transit API | 5 | OpenRouteService integration with graceful degradation |
| 12 | Confidence signaling | 4 | `source_recency`, `corroborating_sources`, `review_volume`, `confidence_score` on ResearchItem |

#### State Schema Additions (v2)

```python
# Graph Control
_awaiting_input: Optional[str]
_callback: Optional[str]
_delegate_to: Optional[str]
_chain: list
_routing_echo: str
_error_agent: Optional[str]
_error_context: Optional[str]
_loopback_depth: int

# Progressive profiling
onboarding_depth: str  # "minimal" | "standard" | "complete"

# Confidence signaling (on ResearchItem)
source_recency: Optional[str]
corroborating_sources: Optional[int]
review_volume: Optional[str]
confidence_score: Optional[float]
```

#### Settings Additions (v2)

| Variable | Default | Purpose |
|----------|---------|---------|
| `WEATHER_API_KEY` | `""` | WeatherAPI.com (optional) |
| `TRANSIT_API_KEY` | `""` | OpenRouteService (optional) |

#### New Commands (v2)

| Command | Handler | Description |
|---------|---------|-------------|
| `/summary` | `handlers.py` (direct) | One-screen trip overview via `synthesize_trip()` |
| `/library` | `handlers.py` (direct) | Calls `library_sync()` tool directly (no graph) |

---

## 8. Appendices

### 8.1 Token Budget Reference

| Agent | max_tokens | Typical Output | Headroom |
|-------|-----------|---------------|----------|
| Research (city) | 16,384 | ~4,000-4,600 tokens | ~3.5x |
| Planner | 16,384 | ~1,700-3,300 tokens | ~5x |
| Scheduler | 8,192 | ~1,000-2,000 tokens | ~4x |
| Prioritizer | 8,192 | ~700-1,700 tokens | ~5x |
| Feedback | 4,096 | ~170-670 tokens | ~6x |
| Cost | 4,096 | ~170-670 tokens | ~6x |
| Orchestrator | 4,096 | ~70-330 tokens | ~12x |

Input budget: `_DEFAULT_INPUT_BUDGET = 160,000` tokens. Token estimation: `len(text) // 3` (conservative for CJK/emoji).

### 8.2 LLM Notes Cost

| Parameter | Value |
|-----------|-------|
| Model | claude-haiku-4-5-20251001 |
| Input tokens per call | ~300-500 |
| Output tokens per call | ~50-150 (max 300) |
| Estimated cost per call | ~$0.001 |
| Calls per trip (typical) | ~5-10 |
| Cost per trip | ~$0.005-$0.010 |
| Failure mode | Warning logged (first 3, then every 10th) |

### 8.3 Command Reference

| Command | Agent | Description |
|---------|-------|-------------|
| `/start` | Onboarding | Begin trip planning |
| `/join <id>` | -- | Join shared trip |
| `/trip new` | Onboarding | Start new trip |
| `/trip switch <id>` | -- | Switch active trip |
| `/trip id` | -- | Show current trip ID |
| `/trip archive <id>` | -- | Archive trip |
| `/research <city>` | Research | Research one city |
| `/research all` | Research | Research all cities |
| `/library` | Librarian | Sync markdown knowledge base |
| `/priorities` | Prioritizer | View/set priority tiers |
| `/plan` | Planner | Generate itinerary |
| `/adjust` | Feedback | Adjust plan from feedback |
| `/agenda` | Scheduler | Get 2-day detailed agenda |
| `/feedback` | Feedback | End-of-day check-in |
| `/costs` | Cost | Full budget report |
| `/costs today` | Cost | Today's spending |
| `/costs save` | Cost | Savings tips |
| `/costs convert X` | Cost | Currency conversion |
| `/costs <city>` | Cost | City breakdown |
| `/costs <category>` | Cost | Category breakdown |
| `/status` | Orchestrator | Planning progress |
| `/mytrips` | -- | Browse/switch trips |
| `/help` | Orchestrator | Command reference |

### 8.4 File Index

| File | LOC | Purpose |
|------|-----|---------|
| `src/state.py` | 476 | TripState schema, enums, nested TypedDicts |
| `src/graph.py` | 271 | LangGraph definition, nodes, routing |
| `src/main.py` | 94 | Entry point: DB init, graph compile, bot polling |
| `src/agents/base.py` | 212 | BaseAgent, token budgeting, memory injection |
| `src/agents/constants.py` | 12 | MEMORY_AGENTS, AGENT_MAX_TOKENS, NOTES_ELIGIBLE |
| `src/agents/orchestrator.py` | 469 | Routing, status, help, welcome-back |
| `src/agents/onboarding.py` | 387 | 13-field conversational collection |
| `src/agents/research.py` | 423 | Destination + city research |
| `src/agents/planner.py` | 376 | Day-by-day itinerary (Opus) |
| `src/agents/scheduler.py` | 282 | Hour-by-hour agenda |
| `src/agents/prioritizer.py` | 263 | Priority tier ranking |
| `src/agents/feedback.py` | 193 | End-of-day check-in |
| `src/agents/cost.py` | 353 | Budget tracking, spending parsing |
| `src/agents/librarian.py` | 134 | Markdown knowledge base sync |
| `src/tools/trip_memory.py` | 1150 | Per-agent memory system |
| `src/tools/web_search.py` | 86 | Tavily search wrapper |
| `src/tools/currency.py` | 78 | Currency conversion and formatting |
| `src/tools/markdown_sync.py` | 508 | Markdown library writer |
| `src/telegram/handlers.py` | 628 | Telegram <-> LangGraph bridge |
| `src/telegram/formatters.py` | 250 | Display formatting, message splitting |
| `src/telegram/bot.py` | 51 | Bot setup, command handlers |
| `src/config/settings.py` | 39 | Pydantic settings from .env |
| `src/db/persistence.py` | 146 | Async CRUD for trip state |
| `src/db/models.py` | 52 | SQLAlchemy ORM models |
| `tests/conftest.py` | 416 | 7 fixtures, 4 destination states |
| `tests/test_trip_memory.py` | 716 | 63 memory system tests |
| `tests/test_functional_flows.py` | 1964 | 80+ agent flow tests |
| `tests/test_graph_integration.py` | 420 | Full graph execution tests |
| `tests/test_orchestrator.py` | 194 | Routing and meta-command tests |
| `tests/test_persistence.py` | 190 | DB CRUD tests |
| `tests/test_destination_agnostic.py` | 153 | Hardcode scanning tests |
| `tests/test_mytrips.py` | 107 | Trip management tests |
| `tests/test_graph.py` | 94 | Graph compilation tests |
| `tests/test_onboarding.py` | 94 | Onboarding step tests |

### 8.5 Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes | -- | Claude API access |
| `TELEGRAM_BOT_TOKEN` | Yes | -- | Telegram bot access |
| `TAVILY_API_KEY` | No | `""` | Web search (Tavily) |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/trip_state.db` | SQLite path |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `DEFAULT_MODEL` | No | `claude-sonnet-4-5-20250929` | Default LLM model |
| `PLANNER_MODEL` | No | `claude-opus-4-6` | Planner-specific model |
| `LLM_TIMEOUT` | No | `120.0` | LLM call timeout (seconds) |
| `LLM_MAX_RETRIES` | No | `1` | LLM retry count |

### 8.6 Adding a New Agent

1. Create `src/agents/{name}.py` extending `BaseAgent`. Set `agent_name = "{name}"`.
2. Define `get_system_prompt(self, state)` and `async handle(self, state, user_message)`.
3. Add `"{name}"` to `MEMORY_AGENTS` in `src/agents/constants.py`.
4. Add `_build_{name}_memory(state)` in `src/tools/trip_memory.py`.
5. Add to `_ALL_BUILDERS` dict (`trip_memory.py:593-600`).
6. Add lazy import in `_get_agent()` (`graph.py:26-50`).
7. Add node + edges in `build_graph()` (`graph.py:224-263`).
8. Add command mapping in `COMMAND_MAP` (`orchestrator.py:15-31`).
9. If > 4096 output tokens needed, add to `AGENT_MAX_TOKENS` (`constants.py`).
10. If should generate notes, add to `NOTES_ELIGIBLE_AGENTS` (`constants.py`).
11. `_validate_agent_lists()` will raise `RuntimeError` if builders/MEMORY_AGENTS mismatch.
12. Add tests in `test_trip_memory.py` and `test_functional_flows.py`.

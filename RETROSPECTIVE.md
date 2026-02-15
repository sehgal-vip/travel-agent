# Per-Agent Memory Architecture: Retrospective

**Date:** 2026-02-15
**Project:** Multi-Agent Travel Planning Bot (Telegram + LangGraph + Anthropic Claude)
**Scope:** Architecture-level redesign of the memory and token-budget system across all specialist agents

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Requirements Document (PRD)](#2-product-requirements-document-prd)
3. [High-Level Design (HLD)](#3-high-level-design-hld)
4. [Low-Level Design (LLD)](#4-low-level-design-lld)
5. [Retrospective](#5-retrospective)
6. [Testing Strategy](#6-testing-strategy)

---

## 1. Executive Summary

### The Problem

Two compounding failures were causing a **100% city research failure rate** in production:

1. **JSON Truncation** -- A hardcoded `max_tokens=4096` ceiling applied uniformly to every agent, including the research agent whose city research responses consistently produce 12,994--13,826 characters (~4,000--4,600 tokens). The response was severed mid-JSON, producing unparseable output every time.

2. **Monolithic Shared Memory** -- Every agent received the full strategic + tactical memory (~1,100--1,700 tokens) injected into its system prompt regardless of relevance. No agent retained its own learnings across sessions. The orchestrator, planner, feedback agent, and research agent all received identical memory blobs, wasting context window budget and providing no specialization.

### The Solution

A **per-agent memory system** where each specialist agent owns a single `.md` file per trip, combining auto-refreshed programmatic sections (rebuilt from state every invocation) with accumulated LLM-generated notes (preserved across refreshes). Each agent also received a tailored `max_tokens` budget based on its output profile.

### Key Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Research agent `max_tokens` | 4,096 | 16,384 |
| City research success rate | 0% | 100% (JSON no longer truncated) |
| System prompt token waste | ~1,100--1,700 tokens of irrelevant context per agent | 0 (each agent sees only its own memory) |
| Agent continuity across sessions | None | Accumulated notes preserved per agent |
| Agent continuity quality | Raw append, no dedup | Pinned/ephemeral notes, dedup, structured bullets |
| Cross-agent awareness | None (siloed) | `[shared]` tag propagation between agents |
| Write safety | Bare `write_text()` | Atomic writes + async locking + idempotency |
| Input token guard | None (unbounded) | Conservative budget with truncation preserving notes |
| Test suite | Partial coverage | 289 tests, all passing |
| Files changed | -- | 14 files (1 new) |

---

## 2. Product Requirements Document (PRD)

### 2.1 Background

The travel planning bot uses a multi-agent architecture coordinated by LangGraph. An orchestrator routes user messages to specialist agents (research, planner, scheduler, prioritizer, feedback, cost), each powered by Anthropic Claude. The system manages trip state as a `TripState` TypedDict flowing through a state graph.

Prior to this change, two shared files (`strategic.md` and `tactical.md`) were generated from state and injected wholesale into every agent's system prompt. This created three problems:

- **Irrelevant context**: The cost agent received research findings; the feedback agent received booking deadlines. None of it was useful.
- **Token budget pressure**: The shared memory consumed 1,100--1,700 tokens before the agent even began its own reasoning, compressing the space available for complex outputs.
- **No learning**: Agents had no mechanism to record qualitative observations (e.g., "user prefers street food over fine dining") that persist across interactions.

### 2.2 User Stories

**US-1: Research Agent Produces Complete JSON**
> As a user who triggers `/research all`, I expect the research agent to return complete, parseable JSON for every city, so that the planning pipeline can proceed without manual intervention.

Acceptance Criteria:
- Research agent receives `max_tokens=16384` (sufficient for the largest observed city research response).
- JSON parsing succeeds for cities requiring 2+ days of research depth (10 places, 8 activities, 15 food items).
- `_parse_json` logs diagnostic char counts and previews on failure for debugging.

**US-2: Each Agent Sees Only Relevant Context**
> As a specialist agent, my system prompt should contain only the memory sections relevant to my function, so that I can use my full context window for reasoning about my domain.

Acceptance Criteria:
- Research agent sees: trip context, destination intel, research progress, research findings.
- Planner agent sees: trip context (with interests/must-dos/budget), priorities, food highlights, itinerary.
- Scheduler agent sees: trip context, itinerary, practical info, booking deadlines, recent feedback.
- Prioritizer agent sees: trip context (with interests/must-dos), research progress, findings.
- Feedback agent sees: trip context, today's plan excerpt, feedback history.
- Cost agent sees: trip context (with budget/currency), cost knowledge, spending by category/city, benchmarks, savings tips.
- Orchestrator sees: thin destination context (no file written).

**US-3: Agent Notes Accumulate Across Sessions**
> As a returning user, the agents should remember qualitative insights from previous interactions (e.g., "user loves street food," "user was exhausted after Day 2") so that recommendations improve over time.

Acceptance Criteria:
- After meaningful work (research/planner/feedback), a Haiku LLM call generates 2--3 bullet-point notes.
- Notes append to the agent's `.md` file under a `## Agent Notes [accumulated]` section.
- Notes survive memory refreshes (programmatic sections are rebuilt; notes are read and re-appended).
- Notes are capped at 30 lines (`MAX_NOTES_LINES`) to prevent unbounded growth.
- Note generation uses Claude Haiku for cost efficiency (~$0.001/call) and fails silently on error.

**US-4: Orchestrator Remains Lightweight**
> As the orchestrator agent (which runs on every single user message for routing), I should not write a memory file, since that would be wasteful I/O on every interaction.

Acceptance Criteria:
- Orchestrator is excluded from `MEMORY_AGENTS`.
- Orchestrator's system prompt uses thin `get_destination_context()` instead of a memory file.
- `_welcome_back_with_ideas()` reads from `generate_status(state)` directly, not from memory files.

**US-5: Legacy API Removal** *(updated in v2)*
> As a developer, legacy functions (`generate_strategic_memory`, `generate_tactical_memory`, `write_trip_memory`, `read_trip_memory`) should be removed since no production callers exist. Tests verify they are gone.

Acceptance Criteria:
- Legacy functions are deleted from `trip_memory.py`.
- Backward-compatible aliases (`_build_*`) are removed.
- A test (`test_legacy_functions_removed`) confirms they are no longer importable.

**US-6: Write Safety** *(new in v2)*
> As a system handling concurrent Telegram messages, memory file writes must not corrupt data.

Acceptance Criteria:
- All writes use atomic `tempfile.mkstemp` + `os.replace` pattern.
- Async locks per `(trip_id, agent_name)` prevent concurrent read-modify-write races.
- Idempotency: MD5 hash comparison skips redundant writes.
- Handler is the **sole writer** — agents build memory content for prompts without writing.

**US-7: Cross-Agent Awareness** *(new in v2)*
> As a specialist agent, I should see insights from other agents that are relevant to my work.

Acceptance Criteria:
- Notes tagged with `[shared]` propagate to other agents' memory.
- Each agent sees a "Cross-Agent Insights" section with source attribution.
- Capped at 15 lines to prevent noise.

**US-8: Input Token Budget** *(new in v2)*
> As a system operating within a 200K context window, memory injection must not exceed limits.

Acceptance Criteria:
- Conservative `len(text) // 3` token estimation (safe for CJK, emoji).
- Memory truncated if it would exceed 160K input budget, preserving notes section.
- Warning logged on truncation.

### 2.3 Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Latency impact of memory refresh | < 5ms (file read + write, local disk) |
| LLM notes generation cost | < $0.002 per agent invocation |
| Notes storage growth per trip | Bounded: 10 pinned + 25 ephemeral lines per agent (210 lines max across 6 agents) |
| Input token budget | 160K conservative limit with truncation + notes preservation |
| Write safety | Atomic writes + async locking; no data corruption on concurrent access |
| Test coverage | All public API functions tested; integration tests for system prompt injection |

### 2.4 Out of Scope

- Migration of existing `strategic.md` / `tactical.md` files to the new per-agent format.
- UI changes in the Telegram bot (no user-facing changes).
- Compression or summarization of notes when the 30-line cap is reached (simple truncation of oldest lines is used).

---

## 3. High-Level Design (HLD)

### 3.1 Architecture Overview

```
                         User Message (Telegram)
                               |
                               v
                    +---------------------+
                    |   handlers.py       |         constants.py
                    |   (sole writer)     |   <--- MEMORY_AGENTS, AGENT_MAX_TOKENS,
                    +---------------------+         NOTES_ELIGIBLE_AGENTS
                               |
                               v
                    +---------------------+
                    |   LangGraph State   |
                    |   Graph (TripState)  |
                    +---------------------+
                               |
              +--------+-------+-------+--------+--------+--------+
              |        |       |       |        |        |        |
              v        v       v       v        v        v        v
         Orchestrator  Research Planner Scheduler Prioritizer Feedback Cost
              |        |       |       |        |        |        |
              |        +-------+-------+--------+--------+--------+
              |               |
              |               v
              |    +-----------------------+
              |    | BaseAgent             |
              |    | ._ensure_memory_fresh | (READ-ONLY: builds content, no disk write)
              |    | .build_system_prompt  | (token budget + truncation)
              |    +-----------------------+
              |               |
              |               v
              |    +-----------------------+
              |    | trip_memory.py        |
              |    | build_agent_memory_content (sync, read-only)
              |    | build_and_write_agent_memory (async, sole writer)
              |    | refresh_agent_memory  | (async, locked, atomic, idempotent)
              |    | append_agent_notes    | (async, locked, atomic)
              |    | read_shared_insights  | (cross-agent [shared] tags)
              |    +-----------------------+
              |               |
              v               v
    Thin destination    Per-agent .md files
    context (no file)   in data/trips/{id}/
```

### 3.2 Data Flow: Agent Invocation

```
    Agent.build_system_prompt(state)
         |
         +-- Is agent in MEMORY_AGENTS?
         |       |
         |   YES |                          NO
         |       v                           v
         | _ensure_memory_fresh(state)   get_destination_context(state)
         |   (READ-ONLY — no disk write)     |
         |       |                           +-- Return thin context block
         |       +-- build_agent_memory_content(trip_id, agent, state)
         |       |       |
         |       |       +-- Dispatch to per-agent builder function
         |       |       |   (module-level _ALL_BUILDERS dict)
         |       |       |
         |       |       +-- read_shared_insights (cross-agent [shared] notes)
         |       |       |
         |       |       +-- read_agent_notes (sync file read, pinned/ephemeral split)
         |       |       |
         |       |       +-- Return combined content string
         |       |
         |       +-- _truncate_memory (token budget guard)
         |       |
         |       +-- Inject into system prompt
         |                                   |
         v                                   v
    System prompt =                    System prompt =
    TONE + AGENT_PROMPT +              TONE + AGENT_PROMPT +
    --- AGENT MEMORY ---               --- DEST CONTEXT ---
    [version header]                   [thin context block]
    [programmatic sections]            --- END DEST CONTEXT ---
    [cross-agent insights]
    [pinned + ephemeral notes]
    --- END AGENT MEMORY ---
```

### 3.3 Data Flow: Post-Invocation Memory Write

```
    handlers.py: process_message()   (SOLE WRITER)
         |
         +-- Graph returns result
         |
         +-- Save state to DB (repo.merge_state)
         |
         +-- await build_and_write_agent_memory(trip_id, agent_name, state)
         |       |
         |       +-- build_agent_memory_content (sync: builders + notes + shared insights)
         |       |
         |       +-- await refresh_agent_memory
         |       |       |
         |       |       +-- async lock: _get_lock(trip_id, agent_name)
         |       |       +-- MD5 idempotency check (skip if unchanged)
         |       |       +-- _atomic_write (tempfile + os.replace)
         |       |       +-- Observability: log size, warn if >50KB
         |
         +-- If _should_generate_notes(result, agent):
                 |
                 +-- _generate_and_append_notes:
                 |       +-- _extract_notes_context (first line + tail, not blind head truncation)
                 |       +-- Haiku LLM call with structured bullet format
                 |       |   (dedup via <existing_notes>, [pinned]/[shared] tags)
                 |       +-- Parse bullet lines, filter "- NONE" sentinel
                 |       +-- await append_agent_notes (async locked, atomic, sanitized)
                 |
                 +-- On failure: _log_notes_failure (first 3, then every 10th)
```

### 3.4 Memory File Structure

Each agent's `.md` file follows a consistent multi-zone structure:

```markdown
<!-- memory_format: 1 -->
# {Agent} Memory — {Trip Title}

## Trip Context [auto-refreshed]
Destination: JP Japan | Dates: 2026-04-01 to 2026-04-14 (14 days)
Route: Tokyo (5d) → Kyoto (4d) → Osaka (3d)
Interests: food, temples, photography

## {Agent-Specific Section 1} [auto-refreshed]
...

## {Agent-Specific Section N} [auto-refreshed]
...

## Cross-Agent Insights [auto-refreshed]
[research] - [shared] User strongly prefers street food over fine dining
[feedback] - [shared] User energy drops significantly after 3PM

## Agent Notes [accumulated] <!-- mem:notes -->
- [pinned] User is vegetarian — affects all meal recommendations
- [pinned] User has knee issues — avoid steep climbs
- Fushimi Inari research was very detailed; user asked follow-up questions
- Consider lighter itinerary — user mentioned fatigue on Day 3
```

**Zone 1 (Programmatic `[auto-refreshed]`):** Rebuilt from `TripState` every time the agent runs. Deterministic. No stale data. Uses `_build_context()` with field sets (`SLIM_FIELDS`, `EXTENDED_FIELDS`, `COST_FIELDS`) for DRY context building.

**Zone 2 (Cross-Agent Insights `[auto-refreshed]`):** `[shared]`-tagged notes from other agents, auto-collected by `read_shared_insights()`. Capped at 15 lines. Source agent attribution.

**Zone 3 (LLM-generated `[accumulated]`):** Qualitative observations appended after meaningful work. Preserved across rebuilds. Split into:
- **Pinned notes** (`[pinned]` tag): Durable preferences (dietary, mobility, strong likes/dislikes). Capped at 10 lines.
- **Ephemeral notes**: Session-specific insights. Capped at 25 lines.

### 3.5 Per-Agent Data Layout

```
data/trips/{trip_id}/
+-- research.md      <- Trip context + destination intel + research progress + findings
+-- planner.md       <- Trip context (extended) + priorities + food highlights + itinerary
+-- scheduler.md     <- Trip context + itinerary + practical info + deadlines + feedback
+-- prioritizer.md   <- Trip context (extended) + research progress + findings
+-- feedback.md      <- Trip context + today's plan + feedback history
+-- cost.md          <- Trip context (extended) + cost knowledge + spending + benchmarks + tips
```

### 3.6 Token Budget Assignment

Output budgets (defined in `src/agents/constants.py`):
```
AGENT_MAX_TOKENS: dict[str, int] = {
    "research":    16384,   # City research JSON can be 13K+ chars
    "planner":     16384,   # Full itinerary JSON for multi-city trips
    "scheduler":    8192,   # 2-day detailed agenda
    "prioritizer":  8192,   # Priority scoring for all cities
}
# All other agents default to 4096
```

Input budget (defined in `src/agents/base.py`):
```
_DEFAULT_INPUT_BUDGET = 160_000  # conservative for 200K context models

def _estimate_tokens(text: str) -> int:
    return len(text) // 3  # safe for CJK, emoji

def _truncate_memory(memory: str, available_tokens: int) -> str:
    # Preserves notes section when truncating
    # available = _DEFAULT_INPUT_BUDGET - base_prompt_tokens - output_budget
```

### 3.7 Component Interaction Matrix

| Component | Reads | Writes | Dispatches To |
|-----------|-------|--------|---------------|
| `BaseAgent._ensure_memory_fresh()` | `build_agent_memory_content()` (sync) | **Nothing** (read-only) | -- |
| `BaseAgent.build_system_prompt()` | `_ensure_memory_fresh()` or `get_destination_context()` | -- | `_truncate_memory()` |
| `handlers.process_message()` | Graph result | DB state | `build_and_write_agent_memory()`, `append_agent_notes()` |
| `build_agent_memory_content()` | `read_agent_notes()`, `read_shared_insights()` | **Nothing** (sync, read-only) | Per-agent builder functions via `_ALL_BUILDERS` |
| `build_and_write_agent_memory()` | `build_agent_memory_content()` | `refresh_agent_memory()` (async, locked, atomic) | -- |
| `OrchestratorAgent._welcome_back_with_ideas()` | `generate_status(state)` | -- | LLM |
| `_generate_and_append_notes()` | `read_agent_notes()` (for dedup) | `append_agent_notes()` (async, locked, atomic) | Haiku LLM |

---

## 4. Low-Level Design (LLD)

### 4.1 `src/agents/constants.py` *(new file)*

Single source of truth for agent configuration. Breaks the circular import between `base.py` and `trip_memory.py`.

```python
MEMORY_AGENTS = {"research", "planner", "scheduler", "prioritizer", "feedback", "cost"}
AGENT_MAX_TOKENS: dict[str, int] = {
    "research": 16384, "planner": 16384, "scheduler": 8192, "prioritizer": 8192,
}
NOTES_ELIGIBLE_AGENTS = {"research", "planner", "feedback"}
```

### 4.2 `src/agents/base.py`

Imports `MEMORY_AGENTS` and `AGENT_MAX_TOKENS` from `constants.py` (no longer defines them).

**`_ensure_memory_fresh(self, state: TripState) -> str`:** *(rewritten in v2)*

**Read-only.** Does NOT write to disk. The handler is the sole writer.

```python
def _ensure_memory_fresh(self, state: TripState) -> str:
    if self.agent_name not in MEMORY_AGENTS:
        return ""
    from src.tools.trip_memory import build_agent_memory_content
    trip_id = state.get("trip_id")
    if not trip_id:
        return ""
    return build_agent_memory_content(trip_id, self.agent_name, state) or ""
```

**`build_system_prompt` with token budgeting:** *(updated in v2)*

```python
if memory:
    base_tokens = _estimate_tokens(base_prompt)
    output_budget = AGENT_MAX_TOKENS.get(self.agent_name, 4096)
    available = _DEFAULT_INPUT_BUDGET - base_tokens - output_budget
    memory = _truncate_memory(memory, available)
    return base_prompt + f"--- {AGENT} MEMORY ---\n{memory}\n--- END ---"
```

**`_build_memory_sections` is deleted.** Agent subclasses no longer override this method. Memory building is fully centralized in `trip_memory.py` via `_ALL_BUILDERS`.

### 4.3 `src/tools/trip_memory.py`

**Public API:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `build_agent_memory_content` | `(trip_id, agent_name, state, base_dir) -> str \| None` | **Sync, read-only.** Build memory content without writing. Used by `_ensure_memory_fresh` (prompt) and `build_and_write_agent_memory` (persistence). |
| `build_and_write_agent_memory` | `async (trip_id, agent_name, state, base_dir) -> None` | Build content + write to disk. Called by handler only. |
| `refresh_agent_memory` | `async (trip_id, agent_name, content, base_dir) -> None` | Async locked, atomic write, idempotent (MD5 hash check). |
| `read_agent_notes` | `(trip_id, agent_name, base_dir) -> str \| None` | Sync. Supports old/new markers. Pinned/ephemeral split with independent caps. |
| `append_agent_notes` | `async (trip_id, agent_name, new_notes, base_dir) -> None` | Async locked, atomic. Sanitizes marker strings from input. |
| `read_shared_insights` | `(trip_id, exclude_agent, base_dir) -> str` | Read `[shared]`-tagged notes from other agents. Capped at 15 lines. |
| `get_memory_stats` | `(trip_id, base_dir) -> dict` | Size/token stats per agent. For observability. |
| `cleanup_trip_memory` | `async (trip_id, base_dir) -> None` | Remove all memory files + clean lock entries. |
| `cleanup_stale_trips` | `async (max_age_days, base_dir) -> list[str]` | Remove trip folders older than threshold. |

**Constants:**

```python
NOTES_MARKER = "## Agent Notes [accumulated] <!-- mem:notes -->"
_OLD_NOTES_MARKER = "## Agent Notes [accumulated]"  # backward compat
MAX_NOTES_LINES = 30
PINNED_TAG = "[pinned]"
SHARED_TAG = "[shared]"
MAX_PINNED_LINES = 10
MAX_EPHEMERAL_LINES = 25
SCHEDULER_FEEDBACK_ENTRIES = 3
MEMORY_FORMAT_VERSION = "1"
_MEMORY_SIZE_WARN_BYTES = 50 * 1024
```

**Write Safety Infrastructure:**

```python
_memory_locks: dict[tuple[str, str], asyncio.Lock] = {}  # per (trip_id, agent_name)
_get_lock(trip_id, agent_name) -> asyncio.Lock  # lazy creation, atomic in asyncio
_atomic_write(filepath, content) -> None  # tempfile.mkstemp + os.replace
```

**Per-Agent Builder Functions (private, dispatched via `_ALL_BUILDERS`):**

| Function | Agent | Sections Built |
|----------|-------|----------------|
| `_build_research_memory(state)` | research | slim context, destination intel, research status table, research findings |
| `_build_planner_memory(state)` | planner | extended context (interests/must-dos/budget), priorities, food highlights, itinerary |
| `_build_scheduler_memory(state)` | scheduler | slim context, itinerary, practical info, booking deadlines, recent feedback (last 3) |
| `_build_prioritizer_memory(state)` | prioritizer | extended context (interests/must-dos), research status, research findings |
| `_build_feedback_memory(state)` | feedback | slim context, today's plan excerpt (timezone-aware), feedback history table |
| `_build_cost_memory(state)` | cost | cost-extended context (budget/currency), cost knowledge, spending by category/city, benchmarks, tips |

**Unified Context Builder:**

```python
SLIM_FIELDS = {"destination", "dates", "cities", "travelers", "interests"}
EXTENDED_FIELDS = SLIM_FIELDS | {"must_dos", "budget"}
COST_FIELDS = EXTENDED_FIELDS | {"currency"}

_build_context(state, fields) -> str  # DRY replacement for 3 inline blocks
_build_slim_context(state) -> str     # backwards-compat wrapper
```

**Legacy API:** All removed. `generate_strategic_memory`, `generate_tactical_memory`, `write_trip_memory`, `read_trip_memory`, and all `_build_*` backward-compat aliases are deleted. Test `test_legacy_functions_removed` confirms they are gone.

### 4.4 Agent Subclasses *(simplified in v2)*

All 6 agent subclasses had their `_build_memory_sections` overrides **deleted**. Memory building is now fully centralized in `trip_memory.py` via the `_ALL_BUILDERS` module-level dispatch dict. Agents only define `agent_name` and `get_system_prompt()`.

**`src/agents/research.py`** -- `_parse_json` enhanced: logs `chars`, `est_tokens` (chars // 3), and `max_tokens` budget on every LLM synthesis completion. This provides calibration data for future automated thresholds.

**`src/agents/orchestrator.py`** -- `_get_state_summary` enriched with research depth (researched/total cities, item counts), plan status, feedback history with energy levels, and budget spent. Capped at 12 lines (`_MAX_ROUTING_CONTEXT_LINES`).

### 4.5 `src/agents/orchestrator.py`

**`_welcome_back_with_ideas` unchanged** — reads `generate_status(state)` directly (no file I/O).

**`_get_state_summary` enriched** *(new in v2)* — provides routing context:

```python
_MAX_ROUTING_CONTEXT_LINES = 12

def _get_state_summary(self, state):
    # Research: 2/3 cities, 45 items
    # Priorities: 3 cities prioritized
    # Plan: 14 days
    # Feedback: 3 entries, last energy=medium
    # Budget: $1,234 spent
```

### 4.6 `src/telegram/handlers.py` *(significantly refactored in v2)*

**Handler is the sole writer.** All memory writes happen here, not in agents.

**Write sites (all async):**

1. **After every agent invocation:** `await build_and_write_agent_memory(trip_id, responding_agent, state_to_save)`
2. **Onboarding completion:** `Path(...).mkdir(parents=True, exist_ok=True)` — directory only
3. **Auto-research after onboarding:** `await build_and_write_agent_memory(new_trip_id, "research", research_state)`

**Notes generation** *(redesigned in v2):*

Extracted into helper functions for clarity and robustness:

```python
_notes_failure_counts: dict[str, int] = {}
NOTES_MAX_TOKENS = 300  # up from 200
NOTES_MODEL = "claude-haiku-4-5-20251001"

def _should_generate_notes(result, responding_agent) -> bool
    # Uses NOTES_ELIGIBLE_AGENTS from constants.py (not inline tuple)
    # Checks state_updates for research/high_level_plan/feedback_log

def _extract_notes_context(response_text, max_chars=1500) -> str
    # First line (summary) + tail (conclusions), not blind head truncation

async def _generate_and_append_notes(trip_id, responding_agent, result, append_fn)
    # Structured bullet format: "Always format as `- `. If nothing new: `- NONE`"
    # Dedup: existing notes wrapped in <existing_notes> XML (prompt injection protection)
    # Tags: [pinned] for durable prefs, [shared] for cross-agent insights
    # Robust parsing: extract `- ` lines, filter "- NONE" sentinel

def _log_notes_failure(agent_name)
    # Log first 3 failures, then every 10th (with exc_info=True)
```

### 4.7 `tests/test_trip_memory.py` *(updated in v2)*

Test classes and counts:

| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| `TestAgentMemorySections` | 8 | Each agent's builder via `build_agent_memory_content` (no `_build_memory_sections`). Includes `test_orchestrator_returns_none`. |
| `TestPhaseDetection` | 9 | Phase detection, next actions, checklists (renamed from `TestTacticalMemory`) |
| `TestBuildSections` | 9 | Individual section builders (profile, intel, itinerary, learnings, etc.) |
| `TestFileIO` | 13 | Async file creation, overwrite, idempotency (`test_refresh_idempotent`), notes read/write, `build_and_write_agent_memory`, `test_legacy_functions_removed` |
| `TestNotesAccumulation` | 3 | Notes persistence across refreshes, pinned/ephemeral cap enforcement, no-op on missing file |
| `TestSystemPromptIntegration` | 7 | Prompt injection for memory agents, orchestrator exclusion, fallback behavior |
| `TestWelcomeBack` | 4 | Welcome-back uses `generate_status`, correct routing, title/flag display |
| `TestPrerequisiteGuards` | 4 | Soft guards with helpful messages and command suggestions |
| `TestPhasePrompt` | 5 | Phase-appropriate prompts for welcome-back at each trip stage |
| **Total** | **63** | |

Changes from v1:
- `TestStrategicMemory` removed (legacy functions deleted)
- `TestTacticalMemory` renamed to `TestPhaseDetection`
- File I/O tests converted to async (`@pytest.mark.asyncio`)
- Added `test_refresh_idempotent` and `test_orchestrator_returns_none`
- `test_legacy_write_and_read` replaced with `test_legacy_functions_removed`
- Notes cap test updated for pinned/ephemeral split

Full suite total: **289 tests passing**.

---

## 5. Retrospective

### 5.1 What Went Well (v1 + v2)

**Root cause identification was fast** (v1). The JSON truncation bug had clear symptoms: `_parse_json` logged warnings with char counts showing responses consistently hitting the 4,096-token ceiling. Correlating this with the hardcoded `max_tokens` in `BaseAgent.__init__` took minutes, not hours.

**The fix naturally led to a better architecture** (v1). Increasing `max_tokens` for the research agent would have been a one-line fix. But investigating the token budget revealed the second problem (monolithic shared memory consuming ~1,100--1,700 tokens of every agent's context window). Fixing both together produced a cleaner system than either fix alone.

**Centralizing builders eliminated the dual-path problem** (v2). Moving all memory building to `_ALL_BUILDERS` in `trip_memory.py` and deleting `_build_memory_sections` overrides from agent subclasses means there's now exactly one code path that builds memory content. `build_agent_memory_content` is the single function used by both the prompt builder (read-only) and the handler (write).

**Structured bullet format for notes was a reliability win** (v2). Switching from "detect empty string" to "require `- ` prefix, use `- NONE` sentinel" eliminated false positives from Haiku saying "No new patterns observed." or "All insights already captured." — those lines don't start with `- ` and are simply ignored.

**`constants.py` cleanly broke the circular import** (v2). The function-level imports between `base.py` and `trip_memory.py` were fragile. Extracting `MEMORY_AGENTS`, `AGENT_MAX_TOKENS`, and `NOTES_ELIGIBLE_AGENTS` to a dedicated module eliminated the risk entirely.

**Removing legacy functions was easier than maintaining them** (v2). Pre-flight grep confirmed zero production callers. Deleting the functions + aliases and adding a `test_legacy_functions_removed` test was simpler than maintaining deprecated wrappers.

### 5.2 What Was Tricky

**Async conversion of write functions** (v2). `build_agent_memory_content` needed to stay sync (called from `build_system_prompt` in an agent's hot path), while `refresh_agent_memory` and `append_agent_notes` needed to be async (for locking). The boundary is clear — sync reads state, async writes files — but required careful propagation of `await` through `handlers.py`.

**Idempotency and eventual consistency** (v2). The agent sees notes from the _previous_ invocation, not the current one (because the handler writes notes _after_ the agent responds). For rapid multi-message conversations, a note might take one extra round-trip to appear. This is acceptable but required documenting the design tradeoff.

**Pinned vs. ephemeral note caps** (v2). Separate caps (10 pinned, 25 ephemeral) mean the old test checking `MAX_NOTES_LINES = 30` broke. The total effective cap is now 35, but the behavioral contract changed. Tests needed updating to reflect the new split.

**Cross-agent shared insights ordering** (v2). Agent X's `[shared]` notes are written _after_ X responds. So if LangGraph routes through X then Y in a single pass, Y won't see X's shared notes until the next request. This is an eventual-consistency tradeoff documented in code comments.

### 5.3 Blind Spots Addressed

**Blind spot: Uniform token budgets** (v1). The original `max_tokens=4096` was set during initial development. The per-agent `AGENT_MAX_TOKENS` dict makes budget decisions explicit.

**Blind spot: No input-side token guard** (v2). Memory could grow unboundedly. A 6-city trip with extensive research could produce 50KB+ memory files. The conservative `len // 3` token estimation (safe for CJK/emoji) and `_truncate_memory` with notes preservation prevents context window overflow.

**Blind spot: Write corruption** (v2). Bare `write_text()` corrupts on interrupted writes (power loss, process kill). Atomic writes via `tempfile.mkstemp` + `os.replace` prevent partial files. Async locks prevent concurrent read-modify-write races during rapid-fire messages.

**Blind spot: Notes pollution** (v2). Raw append without dedup caused note duplication. The `<existing_notes>` XML injection into Haiku's prompt and structured bullet parsing fixed this. Marker sanitization in `append_agent_notes` prevents prompt injection via notes content.

**Blind spot: Agent silos** (v2). Agents never saw each other's insights. The `[shared]` tag with `read_shared_insights` provides opt-in cross-agent propagation without the noise of broadcasting everything.

**Blind spot: Orchestrator routing blind** (v2). The orchestrator routed on thin destination context without knowing research depth, plan status, or feedback patterns. Enriching `_get_state_summary` provides the routing context needed for smart delegation.

### 5.4 Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| New agent added without updating `MEMORY_AGENTS` and `_ALL_BUILDERS` | `_validate_agent_lists()` raises `RuntimeError` at first invocation if they mismatch |
| Notes marker in LLM-generated content | HTML comment suffix (`<!-- mem:notes -->`); `append_agent_notes` sanitizes incoming text to strip markers |
| File system I/O failures | Atomic writes + try/except in handler; agent falls back to destination context |
| Haiku notes LLM call adds latency | Post-response; `max_tokens=300`; `_log_notes_failure` throttles warnings (first 3, then every 10th) |
| Lock dict grows unboundedly | `cleanup_trip_memory` deletes lock entries; `cleanup_stale_trips` for old trips |
| Token budget too conservative | `_DEFAULT_INPUT_BUDGET = 160K` for 200K models; `// 3` estimation is safe but wasteful for ASCII. Can add `tiktoken` later |
| Rapid messages miss notes | Eventual consistency by design; note appears on next invocation (documented) |

---

## 6. Testing Strategy

### 6.1 Test Architecture

Tests are organized by functional concern, not by file. Each test class has a focused responsibility:

```
tests/test_trip_memory.py
|
+-- TestAgentMemorySections        (unit: each agent's builder via build_agent_memory_content)
+-- TestPhaseDetection             (unit: phase detection, actions, checklists)
+-- TestBuildSections              (unit: individual section builders)
+-- TestFileIO                     (integration: async file create/read/write/overwrite/idempotency)
+-- TestNotesAccumulation          (integration: notes persistence, pinned/ephemeral caps, edge cases)
+-- TestSystemPromptIntegration    (integration: prompt assembly with memory + token budget)
+-- TestWelcomeBack                (integration: orchestrator welcome-back flow)
+-- TestPrerequisiteGuards         (unit: routing guards)
+-- TestPhasePrompt                (unit: phase-appropriate prompt generation)
```

### 6.2 Fixture Hierarchy

Tests use a progressive fixture chain that mirrors real trip progression, plus edge-case fixtures:

```
japan_state (conftest.py)
    |-- Onboarding complete, destination set, no research
    |
    +-- researched_japan
    |       |-- Research data for Tokyo (19 items) and Kyoto (13 items)
    |       |
    |       +-- prioritized_japan
    |               |-- Priorities for Tokyo (8 items across tiers)
    |               |
    |               +-- planned_japan
    |                       |-- 5-day itinerary (3 Tokyo + 1 travel + 1 Kyoto)
    |                       |
    |                       +-- on_trip_japan
    |                               |-- Detailed agenda (Day 1)
    |                               |-- Feedback log (Day 1: ramen highlight)

morocco_state — Different continent, currency (MAD), climate (arid)
paris_state — Single-city trip (3 days). Tests no multi-city routing.
long_trip_state — 28-day 6-city European tour. Tests token budget pressure.
large_research_state — Japan with 60+ items per city. Tests memory size limits.
```

This chain allows testing each agent's memory builder at the exact state it would encounter in production.

### 6.3 Key Test Scenarios

**Agent Memory Sections (8 tests):**
- Each agent's builder is tested with the appropriate fixture state.
- Assertions check for presence of expected section headers AND specific data values.
- Example: research agent's memory must contain "Research Progress" header AND "Senso-ji" (a specific research item).

**Notes Accumulation (3 tests):**
- `test_notes_persist_across_refreshes`: Write memory -> append notes -> rebuild memory -> verify notes survived.
- `test_notes_cap_at_max_lines`: Write 50 lines of notes -> verify only last 30 are kept.
- `test_append_to_nonexistent_file_is_noop`: Append to a file that doesn't exist -> verify no crash, no file created.

**System Prompt Integration (7 tests):**
- Memory agents: verify prompt contains `--- {AGENT} MEMORY ---` wrapper and state-derived content.
- Notes injection: verify `## Agent Notes [accumulated]` appears in prompt when notes exist.
- Orchestrator: verify prompt contains `DESTINATION CONTEXT` (not `ORCHESTRATOR MEMORY`).
- No trip_id: verify graceful fallback to base prompt.
- Mode detection: verify orchestrator switches between TRAVEL STRATEGIST and EXECUTIVE ASSISTANT based on trip phase.

**File I/O (12 tests):**
- All tests use `tmp_path` fixture for filesystem isolation.
- Tests cover: file creation, directory creation, overwrite semantics, missing file handling, notes marker detection, notes accumulation, `build_and_write_agent_memory` dispatch, orchestrator exclusion, notes preservation across rebuilds, and legacy API compatibility.

**Welcome Back (4 tests):**
- Uses `AsyncMock` to patch `ChatAnthropic.ainvoke`.
- Verifies trip title and flag emoji appear in output.
- Captures the LLM prompt to verify it contains the correct phase-appropriate suggestion (e.g., `/research all` when no research exists).
- Verifies `generate_status(state)` is used as context (no file reads).

### 6.4 Test Execution

```bash
# Run full suite
pytest tests/ -v

# Run only per-agent memory tests
pytest tests/test_trip_memory.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### 6.5 What Is NOT Tested (and Why)

| Area | Reason |
|------|--------|
| Haiku notes LLM call in `handlers.py` | Requires live API key; wrapped in try/except; cost-prohibitive for CI |
| Actual file sizes of memory files | Depends on real trip state size; unit tests verify structure, not size |
| `_parse_json` truncation scenarios | Would require mocking a truncated LLM response; covered by integration testing in staging |
| Concurrent async lock contention | Would require multi-task test harness; locks are standard `asyncio.Lock` |
| Cross-agent `[shared]` tag end-to-end | Requires multi-agent invocation; unit tested via `read_shared_insights` |
| Token budget truncation in production | Requires 50KB+ memory files; logic unit tested; `large_research_state` fixture available |

---

## Appendix A: Token Budget Justification

| Agent | Typical Output | Previous Budget | New Budget | Headroom |
|-------|---------------|-----------------|------------|----------|
| Research (city) | 12,994--13,826 chars (~4,000--4,600 tokens) | 4,096 | 16,384 | ~3.5x |
| Research (destination intel) | ~3,000--5,000 chars (~1,000--1,700 tokens) | 4,096 | 16,384 | Shared budget |
| Planner | 5,000--10,000 chars (~1,700--3,300 tokens) | 4,096 | 16,384 | ~5x |
| Scheduler | 3,000--6,000 chars (~1,000--2,000 tokens) | 4,096 | 8,192 | ~4x |
| Prioritizer | 2,000--5,000 chars (~700--1,700 tokens) | 4,096 | 8,192 | ~5x |
| Feedback | 500--2,000 chars (~170--670 tokens) | 4,096 | 4,096 | No change |
| Cost | 500--2,000 chars (~170--670 tokens) | 4,096 | 4,096 | No change |
| Orchestrator | 200--1,000 chars (~70--330 tokens) | 4,096 | 4,096 | No change |

## Appendix B: Cost Impact of LLM Notes

| Parameter | Value |
|-----------|-------|
| Model | claude-haiku-4-5-20251001 |
| Input tokens per call | ~300--500 (increased: includes existing notes for dedup) |
| Output tokens per call | ~50--150 (`max_tokens=300`) |
| Cost per call (estimated) | ~$0.001 |
| Calls per trip (typical) | ~5--10 (research + plan + feedback sessions) |
| Cost per trip | ~$0.005--$0.010 |
| Failure mode | Warning logged (first 3, then every 10th; `exc_info=True`) |

## Appendix C: Migration Checklist for Adding a New Agent

1. Create the agent class in `src/agents/{name}.py` extending `BaseAgent`.
2. Set `agent_name = "{name}"` class attribute.
3. Define `get_system_prompt(self, state)` — no `_build_memory_sections` override needed.
4. Add `"{name}"` to `MEMORY_AGENTS` in `src/agents/constants.py`.
5. Add `_build_{name}_memory(state)` function in `src/tools/trip_memory.py`.
6. Add `"{name}": _build_{name}_memory` to `_ALL_BUILDERS` dict in `trip_memory.py`.
7. If the agent needs more than 4,096 output tokens, add it to `AGENT_MAX_TOKENS` in `constants.py`.
8. `_validate_agent_lists()` will **raise RuntimeError** at startup if `_ALL_BUILDERS` keys don't match `MEMORY_AGENTS`.
9. If the agent should generate LLM notes, add to `NOTES_ELIGIBLE_AGENTS` in `constants.py`.
10. Add tests in `tests/test_trip_memory.py`:
    - `TestAgentMemorySections`: verify `build_agent_memory_content` output.
    - `TestFileIO`: verify `build_and_write_agent_memory` creates the correct file.
    - `TestSystemPromptIntegration`: verify prompt injection.

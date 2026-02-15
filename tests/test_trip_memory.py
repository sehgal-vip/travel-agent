"""Tests for per-agent memory — agent sections, file I/O, notes, integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.constants import MEMORY_AGENTS
from src.agents.base import BaseAgent, get_destination_context
from src.agents.orchestrator import OrchestratorAgent, generate_status
from src.tools.trip_memory import (
    NOTES_MARKER,
    MAX_NOTES_LINES,
    _build_current_phase,
    _build_next_actions,
    _build_phase_checklist,
    _build_pretrip_checklist,
    _build_trip_learnings,
    _build_trip_profile,
    _detect_current_phase,
    append_agent_notes,
    build_agent_memory_content,
    build_and_write_agent_memory,
    build_booking_deadlines,
    build_cost_knowledge,
    build_destination_intel,
    build_itinerary_knowledge,
    build_priorities_knowledge,
    build_research_findings,
    read_agent_notes,
    refresh_agent_memory,
)


# ─── Fixtures ────────────────────────────────────────────


@pytest.fixture
def researched_japan(japan_state):
    """Japan state with research data populated."""
    japan_state["research"] = {
        "Tokyo": {
            "last_updated": "2026-02-01T12:00:00Z",
            "places": [
                {"name": "Senso-ji", "category": "place", "advance_booking": False},
                {"name": "Meiji Shrine", "category": "place", "advance_booking": False},
                {"name": "Shibuya Crossing", "category": "place", "advance_booking": False},
                {"name": "TeamLab Borderless", "category": "place", "advance_booking": True, "booking_lead_time": "Book 2 weeks ahead"},
                {"name": "Ueno Park", "category": "place", "advance_booking": False},
                {"name": "Shinjuku Gyoen", "category": "place", "advance_booking": False},
                {"name": "Imperial Palace", "category": "place", "advance_booking": False},
                {"name": "Akihabara", "category": "place", "advance_booking": False},
            ],
            "food": [
                {"name": "Tsukiji Outer Market", "category": "food", "advance_booking": False},
                {"name": "Ramen Alley", "category": "food", "advance_booking": False},
                {"name": "Tonkatsu Maisen", "category": "food", "advance_booking": False},
                {"name": "Sushi Dai", "category": "food", "advance_booking": True, "booking_lead_time": "Reserve 1 week ahead"},
            ],
            "activities": [
                {"name": "Tea Ceremony", "category": "activity", "advance_booking": True, "booking_lead_time": "Book 3 days ahead"},
                {"name": "Sumo Tournament", "category": "activity", "advance_booking": True, "booking_lead_time": "Tickets sell out fast"},
                {"name": "Cooking Class", "category": "activity", "advance_booking": False},
            ],
            "logistics": [{"name": "Suica Card", "category": "logistics"}],
            "tips": [{"name": "Best ramen spots", "category": "tip"}],
            "hidden_gems": [
                {"name": "Yanaka Ginza", "category": "hidden_gem"},
                {"name": "Shimokitazawa", "category": "hidden_gem"},
            ],
        },
        "Kyoto": {
            "last_updated": "2026-02-01T13:00:00Z",
            "places": [
                {"name": "Fushimi Inari", "category": "place", "advance_booking": False},
                {"name": "Kinkaku-ji", "category": "place", "advance_booking": False},
                {"name": "Arashiyama", "category": "place", "advance_booking": False},
                {"name": "Nijo Castle", "category": "place", "advance_booking": False},
                {"name": "Gion District", "category": "place", "advance_booking": False},
                {"name": "Philosopher's Path", "category": "place", "advance_booking": False},
                {"name": "Kiyomizu-dera", "category": "place", "advance_booking": False},
            ],
            "food": [
                {"name": "Nishiki Market", "category": "food", "advance_booking": False},
                {"name": "Kaiseki at Kikunoi", "category": "food", "advance_booking": True, "booking_lead_time": "Reserve 2 weeks ahead"},
            ],
            "activities": [
                {"name": "Geisha Show", "category": "activity", "advance_booking": True, "booking_lead_time": "Book 1 week ahead"},
            ],
            "logistics": [],
            "tips": [],
            "hidden_gems": [
                {"name": "Tofuku-ji", "category": "hidden_gem"},
                {"name": "Kurama Onsen", "category": "hidden_gem"},
                {"name": "Kifune Shrine", "category": "hidden_gem"},
            ],
        },
    }
    return japan_state


@pytest.fixture
def prioritized_japan(researched_japan):
    """Japan state with priorities set."""
    researched_japan["priorities"] = {
        "Tokyo": [
            {"item_id": "t1", "name": "Senso-ji", "category": "place", "tier": "must_do", "score": 95, "reason": "Iconic"},
            {"item_id": "t2", "name": "Tsukiji Outer Market", "category": "food", "tier": "must_do", "score": 93, "reason": "Food lover"},
            {"item_id": "t3", "name": "Meiji Shrine", "category": "place", "tier": "must_do", "score": 90, "reason": "Cultural"},
            {"item_id": "t4", "name": "Shibuya Crossing", "category": "place", "tier": "must_do", "score": 88, "reason": "Photography"},
            {"item_id": "t5", "name": "TeamLab Borderless", "category": "place", "tier": "must_do", "score": 85, "reason": "Unique"},
            {"item_id": "t6", "name": "Akihabara", "category": "place", "tier": "nice_to_have", "score": 70, "reason": "Fun"},
            {"item_id": "t7", "name": "Ueno Park", "category": "place", "tier": "nice_to_have", "score": 65, "reason": "Cherry blossoms"},
            {"item_id": "t8", "name": "Ramen Alley", "category": "food", "tier": "nice_to_have", "score": 60, "reason": "Casual"},
        ],
    }
    return researched_japan


@pytest.fixture
def planned_japan(prioritized_japan):
    """Japan state with itinerary."""
    prioritized_japan["high_level_plan"] = [
        {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Arrival & Asakusa"},
        {"day": 2, "date": "2026-04-02", "city": "Tokyo", "theme": "Tsukiji & Shibuya"},
        {"day": 3, "date": "2026-04-03", "city": "Tokyo", "theme": "Culture & Temples"},
        {"day": 4, "date": "2026-04-04", "city": "Tokyo", "theme": "TeamLab & Transfer"},
        {"day": 5, "date": "2026-04-05", "city": "Kyoto", "theme": "Fushimi Inari at Sunrise"},
    ]
    return prioritized_japan


@pytest.fixture
def on_trip_japan(planned_japan):
    """Japan state with agenda and feedback (on-trip)."""
    planned_japan["detailed_agenda"] = [
        {"day": 1, "date": "2026-04-01", "city": "Tokyo", "theme": "Arrival", "slots": []},
    ]
    planned_japan["feedback_log"] = [
        {
            "day": 1,
            "date": "2026-04-01",
            "city": "Tokyo",
            "highlight": "Amazing ramen near hotel",
            "discoveries": ["Local ramen alley not in guides"],
            "adjustments_made": ["Swapped Akihabara for extra time at Tsukiji"],
            "energy_level": "medium",
            "completed_items": [],
            "skipped_items": [],
        },
    ]
    return planned_japan


# ═══ TestAgentMemorySections ═══════════════════════════


class TestAgentMemorySections:
    """Test build_agent_memory_content returns relevant data for each agent."""

    def test_research_memory_has_context_and_status(self, japan_state, tmp_path):
        """Research memory includes trip context and research progress."""
        result = build_agent_memory_content("japan-2026", "research", japan_state, base_dir=str(tmp_path))
        assert "Trip Context" in result
        assert "Japan" in result
        assert "Research Progress" in result

    def test_research_memory_has_findings(self, researched_japan, tmp_path):
        """Research memory includes research findings when available."""
        result = build_agent_memory_content("japan-2026", "research", researched_japan, base_dir=str(tmp_path))
        assert "Research Findings" in result
        assert "Tokyo" in result
        assert "Senso-ji" in result

    def test_planner_memory_has_interests_and_budget(self, japan_state, tmp_path):
        """Planner memory includes interests, must-dos, budget."""
        result = build_agent_memory_content("japan-2026", "planner", japan_state, base_dir=str(tmp_path))
        assert "Trip Context" in result
        assert "food" in result
        assert "Fushimi Inari" in result  # must-do
        assert "midrange" in result  # budget style

    def test_planner_memory_has_food_highlights(self, researched_japan, tmp_path):
        """Planner memory includes food highlights per city."""
        result = build_agent_memory_content("japan-2026", "planner", researched_japan, base_dir=str(tmp_path))
        assert "Food Highlights" in result
        assert "Tsukiji" in result

    def test_scheduler_memory_has_practical_info(self, planned_japan, tmp_path):
        """Scheduler memory includes practical info (climate, transport, phrases)."""
        result = build_agent_memory_content("japan-2026", "scheduler", planned_japan, base_dir=str(tmp_path))
        assert "Practical Info" in result
        assert "temperate" in result
        assert "Itinerary" in result

    def test_prioritizer_memory_has_research(self, researched_japan, tmp_path):
        """Prioritizer memory includes research status and findings."""
        result = build_agent_memory_content("japan-2026", "prioritizer", researched_japan, base_dir=str(tmp_path))
        assert "Research Progress" in result
        assert "Research Findings" in result
        assert "food" in result.lower()  # interests

    def test_feedback_memory_has_history(self, on_trip_japan, tmp_path):
        """Feedback memory includes feedback history."""
        result = build_agent_memory_content("japan-2026", "feedback", on_trip_japan, base_dir=str(tmp_path))
        assert "Feedback History" in result
        assert "Amazing ramen" in result

    def test_cost_memory_has_budget_and_currency(self, japan_state, tmp_path):
        """Cost memory includes budget, currency info."""
        result = build_agent_memory_content("japan-2026", "cost", japan_state, base_dir=str(tmp_path))
        assert "Trip Context" in result
        assert "JPY" in result
        assert "4,000" in result  # budget total

    def test_orchestrator_returns_none(self, japan_state, tmp_path):
        """Orchestrator is not a memory agent — returns None."""
        result = build_agent_memory_content("japan-2026", "orchestrator", japan_state, base_dir=str(tmp_path))
        assert result is None


# ═══ TestPhaseDetection ════════════════════════════════


class TestPhaseDetection:
    def test_phase_detection_pre_research(self, japan_state):
        """Returns phase 3 Research."""
        phase_num, name, _ = _detect_current_phase(japan_state)
        assert phase_num == 3
        assert name == "Research"

    def test_phase_detection_post_research_no_priorities(self, researched_japan):
        """Returns phase 5 Structuring (skips phase 4 when no priorities)."""
        phase_num, name, _ = _detect_current_phase(researched_japan)
        assert phase_num == 5
        assert name == "Structuring"

    def test_phase_detection_on_trip(self, on_trip_japan):
        """Returns phase 9 based on feedback."""
        phase_num, name, _ = _detect_current_phase(on_trip_japan)
        assert phase_num == 9
        assert "Adaptation" in name

    def test_next_actions_match_phase(self, japan_state):
        """Correct commands suggested per phase."""
        result = _build_next_actions(japan_state)
        assert "/research all" in result

    def test_next_actions_priorities_phase(self, researched_japan):
        """After research, suggests /plan (with /priorities as optional)."""
        result = _build_next_actions(researched_japan)
        assert "/plan" in result

    def test_next_actions_plan_phase(self, prioritized_japan):
        """After priorities, suggests plan."""
        result = _build_next_actions(prioritized_japan)
        assert "/plan" in result

    def test_phase_checklist_marks_completed(self, researched_japan):
        """Completed phases have [x]."""
        result = _build_phase_checklist(researched_japan)
        assert "[x] Phase 1-2: Anchoring" in result
        assert "[x] Phase 3: Research" in result
        assert "[ ] Phase 4: Priorities (optional)" in result

    def test_pretrip_checklist_present(self, planned_japan):
        """Destination-specific items appear after planning."""
        result = _build_pretrip_checklist(planned_japan)
        assert "Book flights" in result
        assert "rail pass" in result.lower()

    def test_pretrip_checklist_absent_before_plan(self, japan_state):
        """No checklist before planning phase."""
        result = _build_pretrip_checklist(japan_state)
        assert result == ""


# ═══ TestBuildSections ═════════════════════════════════


class TestBuildSections:
    def test_trip_profile_table(self, japan_state):
        """Table has all expected field rows."""
        result = _build_trip_profile(japan_state)
        assert "| Destination |" in result
        assert "Japan" in result
        assert "| Route |" in result
        assert "Tokyo" in result
        assert "| Dates |" in result
        assert "| Travelers |" in result
        assert "| Budget |" in result
        assert "| Interests |" in result
        assert "food" in result

    def test_destination_intel(self, japan_state):
        """Currency, language, tipping present."""
        result = build_destination_intel(japan_state)
        assert "Japanese" in result
        assert "JPY" in result
        assert "Not customary" in result or "tipping" in result.lower()

    def test_itinerary_table(self, planned_japan):
        """Correct day/date/city/theme columns."""
        result = build_itinerary_knowledge(planned_japan)
        assert "| Day | Date | City | Theme |" in result
        assert "| 1 | 2026-04-01 | Tokyo | Arrival & Asakusa |" in result
        assert "| 5 | 2026-04-05 | Kyoto | Fushimi Inari at Sunrise |" in result

    def test_trip_learnings(self, on_trip_japan):
        """Highlights from feedback log."""
        result = _build_trip_learnings(on_trip_japan)
        assert "Amazing ramen near hotel" in result
        assert "Local ramen alley" in result

    def test_cost_knowledge(self, japan_state):
        """Budget/spent/status present."""
        result = build_cost_knowledge(japan_state)
        assert "$4,000" in result
        assert "on_track" in result

    def test_booking_deadlines(self, researched_japan):
        """Items with advance_booking=True are listed."""
        result = build_booking_deadlines(researched_japan)
        assert "TeamLab Borderless" in result
        assert "Sushi Dai" in result
        assert "Tea Ceremony" in result
        assert "Kaiseki at Kikunoi" in result

    def test_active_adjustments(self, on_trip_japan):
        """Adjustments from feedback appear."""
        from src.tools.trip_memory import _build_active_adjustments
        result = _build_active_adjustments(on_trip_japan)
        assert "Swapped Akihabara" in result
        assert "Day 1" in result

    def test_pending_decisions_with_must_dos(self, japan_state):
        """Must-dos without plan generate pending decisions."""
        from src.tools.trip_memory import _build_pending_decisions
        result = _build_pending_decisions(japan_state)
        assert "Fushimi Inari at sunrise" in result

    def test_research_findings_top_names(self, researched_japan):
        """Top 3 place/food names appear in research findings."""
        result = build_research_findings(researched_japan)
        assert "Senso-ji" in result
        assert "Meiji Shrine" in result
        assert "Fushimi Inari" in result


# ═══ TestFileIO ════════════════════════════════════════


class TestFileIO:
    @pytest.mark.asyncio
    async def test_refresh_creates_file(self, tmp_path):
        """refresh_agent_memory creates {trip_id}/{agent_name}.md."""
        await refresh_agent_memory("test-trip", "research", "# Research Memory", base_dir=str(tmp_path))
        assert (tmp_path / "test-trip" / "research.md").exists()
        content = (tmp_path / "test-trip" / "research.md").read_text()
        assert "Research Memory" in content

    @pytest.mark.asyncio
    async def test_refresh_creates_directory(self, tmp_path):
        """Creates trip folder if missing."""
        base = str(tmp_path / "new_base")
        await refresh_agent_memory("new-trip", "planner", "# Planner", base_dir=base)
        assert (Path(base) / "new-trip" / "planner.md").is_file()

    @pytest.mark.asyncio
    async def test_refresh_overwrites(self, tmp_path):
        """Writing again overwrites the file."""
        await refresh_agent_memory("trip-1", "research", "v1", base_dir=str(tmp_path))
        await refresh_agent_memory("trip-1", "research", "v2", base_dir=str(tmp_path))
        content = (tmp_path / "trip-1" / "research.md").read_text()
        assert content == "v2"

    @pytest.mark.asyncio
    async def test_refresh_idempotent(self, tmp_path):
        """Same content doesn't rewrite the file (idempotency check)."""
        await refresh_agent_memory("trip-1", "research", "same content", base_dir=str(tmp_path))
        target = tmp_path / "trip-1" / "research.md"
        mtime1 = target.stat().st_mtime
        import time
        time.sleep(0.01)
        await refresh_agent_memory("trip-1", "research", "same content", base_dir=str(tmp_path))
        mtime2 = target.stat().st_mtime
        assert mtime1 == mtime2  # file should not have been rewritten

    def test_read_notes_missing_file(self, tmp_path):
        """Returns None for nonexistent file."""
        result = read_agent_notes("nonexistent", "research", base_dir=str(tmp_path))
        assert result is None

    @pytest.mark.asyncio
    async def test_read_notes_no_marker(self, tmp_path):
        """Returns None when file has no notes section."""
        await refresh_agent_memory("trip-1", "research", "# Memory\n## Context", base_dir=str(tmp_path))
        result = read_agent_notes("trip-1", "research", base_dir=str(tmp_path))
        assert result is None

    @pytest.mark.asyncio
    async def test_read_notes_with_marker(self, tmp_path):
        """Returns notes content after marker."""
        content = f"# Memory\n## Context\nsome context\n\n{NOTES_MARKER}\n- Note one\n- Note two"
        await refresh_agent_memory("trip-1", "research", content, base_dir=str(tmp_path))
        result = read_agent_notes("trip-1", "research", base_dir=str(tmp_path))
        assert "Note one" in result
        assert "Note two" in result

    @pytest.mark.asyncio
    async def test_append_notes_creates_section(self, tmp_path):
        """append_agent_notes adds notes section when missing."""
        await refresh_agent_memory("trip-1", "research", "# Memory", base_dir=str(tmp_path))
        await append_agent_notes("trip-1", "research", "- First note", base_dir=str(tmp_path))
        content = (tmp_path / "trip-1" / "research.md").read_text()
        assert NOTES_MARKER in content
        assert "First note" in content

    @pytest.mark.asyncio
    async def test_append_notes_accumulates(self, tmp_path):
        """Multiple appends accumulate notes."""
        content = f"# Memory\n\n{NOTES_MARKER}\n- Note one"
        await refresh_agent_memory("trip-1", "research", content, base_dir=str(tmp_path))
        await append_agent_notes("trip-1", "research", "- Note two", base_dir=str(tmp_path))
        result = (tmp_path / "trip-1" / "research.md").read_text()
        assert "Note one" in result
        assert "Note two" in result

    @pytest.mark.asyncio
    async def test_build_and_write_agent_memory(self, tmp_path, japan_state):
        """build_and_write_agent_memory creates the right file for a memory agent."""
        await build_and_write_agent_memory("japan-2026", "research", japan_state, base_dir=str(tmp_path))
        filepath = tmp_path / "japan-2026" / "research.md"
        assert filepath.is_file()
        content = filepath.read_text()
        assert "Research Memory" in content
        assert "Japan" in content

    @pytest.mark.asyncio
    async def test_build_and_write_skips_orchestrator(self, tmp_path, japan_state):
        """Orchestrator should NOT get a memory file."""
        await build_and_write_agent_memory("japan-2026", "orchestrator", japan_state, base_dir=str(tmp_path))
        filepath = tmp_path / "japan-2026" / "orchestrator.md"
        assert not filepath.exists()

    @pytest.mark.asyncio
    async def test_build_and_write_preserves_notes(self, tmp_path, japan_state):
        """Notes survive a rebuild."""
        # First write
        await build_and_write_agent_memory("japan-2026", "research", japan_state, base_dir=str(tmp_path))
        # Append notes
        await append_agent_notes("japan-2026", "research", "- Important insight", base_dir=str(tmp_path))
        # Rebuild (should preserve notes)
        await build_and_write_agent_memory("japan-2026", "research", japan_state, base_dir=str(tmp_path))
        content = (tmp_path / "japan-2026" / "research.md").read_text()
        assert "Important insight" in content

    def test_legacy_functions_removed(self):
        """Legacy functions are gone — import should fail."""
        import importlib
        mod = importlib.import_module("src.tools.trip_memory")
        assert not hasattr(mod, "generate_strategic_memory")
        assert not hasattr(mod, "generate_tactical_memory")
        assert not hasattr(mod, "write_trip_memory")
        assert not hasattr(mod, "read_trip_memory")


# ═══ TestNotesAccumulation ════════════════════════════


class TestNotesAccumulation:
    @pytest.mark.asyncio
    async def test_notes_persist_across_refreshes(self, tmp_path, japan_state):
        """Agent notes survive memory refreshes."""
        # Initial write
        await build_and_write_agent_memory("trip-1", "research", japan_state, base_dir=str(tmp_path))
        # Add notes
        await append_agent_notes("trip-1", "research", "- User loves street food", base_dir=str(tmp_path))
        # Refresh (simulates next agent run)
        await build_and_write_agent_memory("trip-1", "research", japan_state, base_dir=str(tmp_path))
        # Read notes back
        notes = read_agent_notes("trip-1", "research", base_dir=str(tmp_path))
        assert "street food" in notes

    @pytest.mark.asyncio
    async def test_notes_cap_at_max_lines(self, tmp_path):
        """Ephemeral notes are capped at MAX_EPHEMERAL_LINES, pinned at MAX_PINNED_LINES."""
        from src.tools.trip_memory import MAX_EPHEMERAL_LINES, MAX_PINNED_LINES, PINNED_TAG

        # Mix of pinned and ephemeral — exceed both caps
        pinned_lines = [f"- {PINNED_TAG} Pinned note {i}" for i in range(MAX_PINNED_LINES + 5)]
        ephemeral_lines = [f"- Ephemeral note {i}" for i in range(MAX_EPHEMERAL_LINES + 10)]
        all_lines = pinned_lines + ephemeral_lines
        content = f"# Memory\n\n{NOTES_MARKER}\n" + "\n".join(all_lines)
        await refresh_agent_memory("trip-1", "research", content, base_dir=str(tmp_path))
        notes = read_agent_notes("trip-1", "research", base_dir=str(tmp_path))
        assert notes is not None
        note_lines = notes.splitlines()
        pinned_result = [l for l in note_lines if PINNED_TAG in l]
        ephemeral_result = [l for l in note_lines if PINNED_TAG not in l]
        assert len(pinned_result) == MAX_PINNED_LINES
        assert len(ephemeral_result) == MAX_EPHEMERAL_LINES
        # Should keep the LAST lines of each category
        assert f"Pinned note {MAX_PINNED_LINES + 4}" in pinned_result[-1]
        assert f"Ephemeral note {MAX_EPHEMERAL_LINES + 9}" in ephemeral_result[-1]

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_file_is_noop(self, tmp_path):
        """Appending to a file that doesn't exist does nothing."""
        await append_agent_notes("nonexistent", "research", "- Note", base_dir=str(tmp_path))
        assert not (tmp_path / "nonexistent" / "research.md").exists()


# ═══ TestSystemPromptIntegration ══════════════════════


class TestSystemPromptIntegration:
    def test_build_system_prompt_memory_agent_includes_memory(self, japan_state, tmp_path):
        """When a MEMORY_AGENTS agent builds its prompt, memory content is included (no disk write)."""
        from src.agents.research import ResearchAgent
        agent = ResearchAgent()

        with patch("src.tools.trip_memory.read_agent_notes", return_value=None):
            prompt = agent.build_system_prompt(japan_state)

        assert "RESEARCH MEMORY" in prompt
        assert "Japan" in prompt

    def test_build_system_prompt_includes_agent_notes(self, japan_state, tmp_path):
        """When notes exist, they appear in the system prompt."""
        from src.agents.research import ResearchAgent
        agent = ResearchAgent()

        with patch("src.tools.trip_memory.read_agent_notes", return_value="- User loves ramen"):
            prompt = agent.build_system_prompt(japan_state)

        assert "Agent Notes" in prompt
        assert "User loves ramen" in prompt

    def test_orchestrator_does_not_write_file(self, japan_state, tmp_path):
        """Orchestrator should NOT write a memory file."""
        orch = OrchestratorAgent()
        prompt = orch.build_system_prompt(japan_state)
        # Should use destination context fallback
        assert "DESTINATION CONTEXT" in prompt
        # Should NOT contain agent memory markers
        assert "ORCHESTRATOR MEMORY" not in prompt

    def test_build_system_prompt_fallback_no_trip_id(self, empty_state):
        """With no trip_id, falls back to destination context logic."""
        agent = BaseAgent()
        prompt = agent.build_system_prompt(empty_state)
        assert "helpful" in prompt.lower()
        assert "MEMORY" not in prompt

    def test_orchestrator_prompt_pre_trip_mode(self, japan_state):
        """System prompt contains TRAVEL STRATEGIST pre-trip."""
        orch = OrchestratorAgent()
        prompt = orch.get_system_prompt(state=japan_state)
        assert "TRAVEL STRATEGIST" in prompt
        assert "EXECUTIVE ASSISTANT" not in prompt

    def test_orchestrator_prompt_on_trip_mode(self, on_trip_japan):
        """With agenda + feedback, prompt contains EXECUTIVE ASSISTANT."""
        orch = OrchestratorAgent()
        prompt = orch.get_system_prompt(state=on_trip_japan)
        assert "EXECUTIVE ASSISTANT" in prompt
        assert "TRAVEL STRATEGIST" not in prompt

    def test_orchestrator_prompt_no_state(self):
        """Without state, defaults to pre-trip mode."""
        orch = OrchestratorAgent()
        prompt = orch.get_system_prompt()
        assert "TRAVEL STRATEGIST" in prompt


# ═══ TestWelcomeBack ══════════════════════════════════


class TestWelcomeBack:
    @pytest.mark.asyncio
    async def test_welcome_back_shows_title(self, japan_state):
        """Includes trip title and flag."""
        orch = OrchestratorAgent()
        mock_response = MagicMock()
        mock_response.content = "Welcome back! You have Tokyo, Kyoto, and Osaka to explore."

        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock, return_value=mock_response):
            result = await orch._welcome_back_with_ideas(japan_state)

        assert "Ramen & Temples Run" in result
        assert "\U0001f1ef\U0001f1f5" in result  # JP flag

    @pytest.mark.asyncio
    async def test_welcome_back_suggests_next_step(self, japan_state):
        """Correct next-step based on progress — no research yet, suggests /research."""
        prompt_capture = {}

        async def capture_ainvoke(messages, **kwargs):
            prompt_capture["human"] = messages[-1].content
            mock_resp = MagicMock()
            mock_resp.content = "Let's start exploring your cities!"
            return mock_resp

        orch = OrchestratorAgent()
        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock, side_effect=capture_ainvoke):
            await orch._welcome_back_with_ideas(japan_state)

        # The phase prompt should mention /research all
        assert "/research all" in prompt_capture["human"]

    @pytest.mark.asyncio
    async def test_welcome_back_uses_status(self, japan_state):
        """Uses generate_status() for context (no memory file reads)."""
        mock_response = MagicMock()
        mock_response.content = "Here's where we are."

        context_capture = {}

        async def capture_ainvoke(messages, **kwargs):
            context_capture["system"] = messages[0].content
            return mock_response

        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock, side_effect=capture_ainvoke):
            orch = OrchestratorAgent()
            result = await orch._welcome_back_with_ideas(japan_state)

        assert "Ramen & Temples Run" in result
        # Status should be used as context
        assert "Research" in context_capture["system"]

    @pytest.mark.asyncio
    async def test_start_routes_to_welcome_back(self, japan_state):
        """When /start is sent after onboarding, routes to welcome back."""
        mock_response = MagicMock()
        mock_response.content = "Welcome back to your Japan trip!"

        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock, return_value=mock_response):
            orch = OrchestratorAgent()
            result = await orch.route(japan_state, "/start")

        assert result["target_agent"] == "orchestrator"
        assert "Ramen & Temples Run" in result["response"]


# ═══ TestPrerequisiteGuards ═══════════════════════════


class TestPrerequisiteGuards:
    def test_guards_offer_help_not_block(self, japan_state):
        """Prerequisite messages contain 'Want me to' / offer to act."""
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "planner")
        assert result is not None
        assert "Want me to" in result or "want me" in result.lower()

    def test_guards_include_command(self, japan_state):
        """Messages include the relevant command."""
        orch = OrchestratorAgent()

        # Planner without research
        result = orch._check_prerequisites(japan_state, "planner")
        assert "/research all" in result

        # Scheduler without plan
        result = orch._check_prerequisites(japan_state, "scheduler")
        assert "/plan" in result

    def test_prioritizer_guard_offers_research(self, japan_state):
        """Prioritizer guard offers to research."""
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(japan_state, "prioritizer")
        assert result is not None
        assert "/research" in result

    def test_planner_allowed_after_research_without_priorities(self, researched_japan):
        """After research, planner is allowed without priorities."""
        orch = OrchestratorAgent()
        result = orch._check_prerequisites(researched_japan, "planner")
        assert result is None


# ═══ TestPhasePrompt ══════════════════════════════════


class TestPhasePrompt:
    def test_phase_prompt_pre_research(self, japan_state):
        orch = OrchestratorAgent()
        prompt = orch._get_phase_prompt(japan_state)
        assert "/research all" in prompt
        assert "Japan" in prompt

    def test_phase_prompt_pre_plan_no_priorities(self, researched_japan):
        """After research without priorities, suggests /plan."""
        orch = OrchestratorAgent()
        prompt = orch._get_phase_prompt(researched_japan)
        assert "/plan" in prompt

    def test_phase_prompt_pre_plan(self, prioritized_japan):
        orch = OrchestratorAgent()
        prompt = orch._get_phase_prompt(prioritized_japan)
        assert "/plan" in prompt

    def test_phase_prompt_pre_agenda(self, planned_japan):
        orch = OrchestratorAgent()
        prompt = orch._get_phase_prompt(planned_japan)
        assert "/agenda" in prompt

    def test_phase_prompt_fully_planned(self, on_trip_japan):
        orch = OrchestratorAgent()
        prompt = orch._get_phase_prompt(on_trip_japan)
        assert "Japan" in prompt
        assert "prep" in prompt.lower() or "download" in prompt.lower() or "book" in prompt.lower()

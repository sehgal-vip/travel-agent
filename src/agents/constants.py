"""Agent configuration constants. Imported by both base.py and trip_memory.py."""

MEMORY_AGENTS = {"research", "planner", "scheduler", "prioritizer", "feedback", "cost"}

AGENT_MAX_TOKENS: dict[str, int] = {
    "research": 16384,
    "planner": 16384,
    "scheduler": 8192,
    "prioritizer": 8192,
}

NOTES_ELIGIBLE_AGENTS = {"research", "planner", "feedback"}

"""Agent configuration constants. Imported by both base.py and trip_memory.py."""

MEMORY_AGENTS = {"research", "planner", "scheduler", "prioritizer", "feedback", "cost"}

AGENT_MAX_TOKENS: dict[str, int] = {
    "research": 75000,   # iterative per-category calls use less but ceiling is high
    "planner": 16384,
    "scheduler": 8192,
    "prioritizer": 8192,
}

# Per-agent LLM timeout overrides (seconds).  Agents not listed use Settings.LLM_TIMEOUT.
# Research and planner produce large structured JSON that can take 3-5+ minutes to generate,
# especially during peak API load (~30-50 tokens/sec × 10k+ tokens).
AGENT_TIMEOUTS: dict[str, float] = {
    "research": 420.0,  # 7 min — full-depth city research is the heaviest prompt
    "planner": 360.0,   # 6 min — multi-day itinerary generation
    "scheduler": 300.0,  # 5 min — detailed agenda with time blocks
}

# Per-agent max-retry overrides.  Agents not listed use Settings.LLM_MAX_RETRIES.
AGENT_MAX_RETRIES: dict[str, int] = {
    "research": 2,  # extra retry — research is the most timeout-prone call
}

NOTES_ELIGIBLE_AGENTS = {"research", "planner", "feedback"}

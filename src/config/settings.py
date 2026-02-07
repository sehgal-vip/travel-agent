"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — all values from .env or environment."""

    ANTHROPIC_API_KEY: str
    TELEGRAM_BOT_TOKEN: str
    TAVILY_API_KEY: str = ""
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/trip_state.db"
    LOG_LEVEL: str = "INFO"
    DEFAULT_MODEL: str = "claude-sonnet-4-5-20250929"
    PLANNER_MODEL: str = "claude-opus-4-5-20250929"

    LLM_TIMEOUT: float = 120.0
    LLM_MAX_RETRIES: int = 1

    AGENT_MODELS: dict[str, str] = {
        "orchestrator": "claude-sonnet-4-5-20250929",
        "onboarding": "claude-sonnet-4-5-20250929",
        "research": "claude-sonnet-4-5-20250929",
        "librarian": "claude-sonnet-4-5-20250929",
        "prioritizer": "claude-sonnet-4-5-20250929",
        "planner": "claude-opus-4-5-20250929",
        "scheduler": "claude-sonnet-4-5-20250929",
        "feedback": "claude-sonnet-4-5-20250929",
        "cost": "claude-sonnet-4-5-20250929",
    }

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

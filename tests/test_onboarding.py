"""Tests for the onboarding agent."""

from __future__ import annotations

import pytest

from src.agents.onboarding import OnboardingAgent


class TestOnboardingAgent:
    """Test onboarding agent initialisation and config extraction."""

    def test_agent_name(self):
        agent = OnboardingAgent()
        assert agent.agent_name == "onboarding"

    def test_system_prompt_destination_agnostic(self):
        agent = OnboardingAgent()
        prompt = agent.get_system_prompt()
        # Should mention ANY destination, not hardcode one
        assert "ANY destination" in prompt
        # Should list the 13 info categories
        assert "Destination" in prompt
        assert "Cities" in prompt
        assert "budget" in prompt.lower()

    def test_extract_config_valid_json(self):
        agent = OnboardingAgent()
        text = '''Here's your trip summary!

```json
{
    "confirmed": true,
    "destination": {"country": "Japan", "country_code": "JP", "region": "East Asia", "flag_emoji": "🇯🇵", "language": "Japanese", "currency_code": "JPY", "currency_symbol": "¥"},
    "cities": [{"name": "Tokyo", "country": "Japan", "days": 4, "order": 1}],
    "dates": {"start": "2026-04-01", "end": "2026-04-14", "total_days": 14},
    "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
    "budget": {"style": "midrange", "total_estimate_usd": null, "splurge_on": [], "save_on": []},
    "interests": ["food", "culture"],
    "must_dos": [],
    "deal_breakers": [],
    "accommodation_pref": "hotels",
    "transport_pref": ["train"]
}
```
'''
        config = agent._extract_config(text)
        assert config is not None
        assert config["confirmed"] is True
        assert config["destination"]["country"] == "Japan"
        assert len(config["cities"]) == 1

    def test_extract_config_invalid_json(self):
        agent = OnboardingAgent()
        text = "No JSON here, just a chat message."
        config = agent._extract_config(text)
        assert config is None

    def test_build_state_from_config(self):
        agent = OnboardingAgent()
        config = {
            "confirmed": True,
            "destination": {"country": "Morocco", "country_code": "MA", "flag_emoji": "🇲🇦", "currency_code": "MAD", "currency_symbol": "DH"},
            "cities": [{"name": "Marrakech", "country": "Morocco", "days": 3, "order": 1}],
            "dates": {"start": "2026-03-15", "end": "2026-03-20", "total_days": 5},
            "travelers": {"count": 2, "type": "couple", "dietary": [], "accessibility": []},
            "budget": {"style": "midrange"},
            "interests": ["food", "architecture"],
            "must_dos": ["Jardin Majorelle"],
            "deal_breakers": [],
            "accommodation_pref": "riad",
            "transport_pref": ["taxi"],
        }
        state: dict = {"trip_id": "test-123"}
        updates = agent._build_state_from_config(config, state)

        assert updates["onboarding_complete"] is True
        assert updates["destination"]["country"] == "Morocco"
        assert updates["cities"][0]["name"] == "Marrakech"
        assert updates["interests"] == ["food", "architecture"]
        assert "research" in updates  # initialised as empty
        assert "cost_tracker" in updates
        assert updates["cost_tracker"]["local_currency"] == "MAD"

    def test_step_estimation(self):
        agent = OnboardingAgent()
        state: dict = {"destination": {}, "cities": [], "dates": {}}
        assert agent._estimate_step(state, "hello", 0) == 0  # stays — destination unfilled

        state["destination"] = {"country": "Japan"}
        assert agent._estimate_step(state, "cities", 0) == 1  # jumps to cities

        state["cities"] = [{"name": "Tokyo"}]
        assert agent._estimate_step(state, "dates", 1) == 2  # jumps to dates

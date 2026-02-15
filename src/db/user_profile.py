"""User profile — cross-trip learning and preference persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class UserProfileModel(_Base):
    """SQLAlchemy model for user preferences across trips."""

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True)
    preferences_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# Default preferences schema
DEFAULT_PREFERENCES = {
    "dietary": [],
    "pace": None,  # "slow" | "moderate" | "fast"
    "accommodation_style": None,
    "budget_tendency": None,  # "frugal" | "moderate" | "splurge"
    "energy_pattern": None,  # "morning_person" | "night_owl" | "balanced"
    "interests_history": [],  # Accumulated interests across trips
    "food_preferences": [],
    "mobility_notes": [],
    "visited_countries": [],
    "travel_style": None,  # "solo" | "couple" | "family" | "friends"
}


class UserProfileRepository:
    """CRUD operations for user profiles."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url)
        self.session_factory = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init_table(self) -> None:
        """Create the user_profiles table if it doesn't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def get_or_create(self, user_id: str) -> dict:
        """Get user preferences, creating a new profile if needed."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if profile:
                try:
                    return json.loads(profile.preferences_json)
                except json.JSONDecodeError:
                    return dict(DEFAULT_PREFERENCES)

            # Create new profile
            new_profile = UserProfileModel(
                user_id=user_id,
                preferences_json=json.dumps(DEFAULT_PREFERENCES),
            )
            session.add(new_profile)
            await session.commit()
            return dict(DEFAULT_PREFERENCES)

    async def update_preferences(self, user_id: str, updates: dict) -> dict:
        """Merge updates into existing preferences."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if not profile:
                prefs = dict(DEFAULT_PREFERENCES)
                prefs.update(updates)
                profile = UserProfileModel(
                    user_id=user_id,
                    preferences_json=json.dumps(prefs),
                )
                session.add(profile)
            else:
                try:
                    prefs = json.loads(profile.preferences_json)
                except json.JSONDecodeError:
                    prefs = dict(DEFAULT_PREFERENCES)

                # Smart merge: lists get extended (deduplicated), scalars get replaced
                for key, value in updates.items():
                    if isinstance(value, list) and isinstance(prefs.get(key), list):
                        existing = set(prefs[key])
                        prefs[key] = list(existing | set(value))
                    else:
                        prefs[key] = value

                profile.preferences_json = json.dumps(prefs)
                profile.updated_at = datetime.now(timezone.utc)

            await session.commit()
            return prefs

    async def merge_from_trip(self, user_id: str, trip_state: dict) -> dict:
        """Extract durable preferences from a completed trip and merge into profile."""
        updates: dict[str, Any] = {}

        # Extract interests
        interests = trip_state.get("interests", [])
        if interests:
            updates["interests_history"] = interests

        # Extract dietary preferences
        dietary = trip_state.get("travelers", {}).get("dietary", [])
        if dietary:
            updates["dietary"] = dietary

        # Extract travel style
        travel_type = trip_state.get("travelers", {}).get("type")
        if travel_type:
            updates["travel_style"] = travel_type

        # Extract accommodation preference
        accom = trip_state.get("accommodation_pref")
        if accom:
            updates["accommodation_style"] = accom

        # Extract budget tendency from actual spending vs budget
        budget = trip_state.get("budget", {})
        if budget.get("style"):
            updates["budget_tendency"] = budget["style"]

        # Extract visited country
        dest = trip_state.get("destination", {})
        country = dest.get("country")
        if country:
            updates["visited_countries"] = [country]

        # Infer pace from feedback
        feedback = trip_state.get("feedback_log", [])
        if feedback:
            energy_levels = [f.get("energy_level", "medium") for f in feedback]
            low_count = sum(1 for e in energy_levels if e == "low")
            high_count = sum(1 for e in energy_levels if e == "high")
            if low_count > len(energy_levels) / 2:
                updates["pace"] = "slow"
            elif high_count > len(energy_levels) / 2:
                updates["pace"] = "fast"
            else:
                updates["pace"] = "moderate"

            # Infer energy pattern (not enough data usually, skip for now)

        # Extract food preferences from feedback
        food_prefs = []
        for f in feedback:
            if f.get("food_rating") == "amazing":
                food_prefs.append(f.get("city", ""))
        if food_prefs:
            updates["food_preferences"] = food_prefs

        # Extract mobility notes
        mobility = trip_state.get("travelers", {}).get("accessibility", [])
        if mobility:
            updates["mobility_notes"] = mobility

        if updates:
            return await self.update_preferences(user_id, updates)

        return await self.get_or_create(user_id)

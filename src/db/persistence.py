"""Async CRUD operations for trip state persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select, union_all
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base, Trip, TripMember

logger = logging.getLogger(__name__)


class TripRepository:
    """Async repository for Trip CRUD backed by SQLAlchemy."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialised.")

    async def create_trip(self, trip_id: str, user_id: str, state: dict) -> Trip:
        async with self.async_session() as session:
            country = state.get("destination", {}).get("country")
            trip = Trip(
                trip_id=trip_id,
                user_id=user_id,
                destination_country=country,
                state_json=json.dumps(state, default=str),
            )
            session.add(trip)
            await session.commit()
            await session.refresh(trip)
            logger.info("Created trip %s for user %s", trip_id, user_id)
            return trip

    async def get_trip(self, trip_id: str) -> Trip | None:
        async with self.async_session() as session:
            return await session.get(Trip, trip_id)

    async def get_active_trips(self, user_id: str) -> list[Trip]:
        async with self.async_session() as session:
            joined_ids = select(TripMember.trip_id).where(TripMember.user_id == user_id)
            result = await session.execute(
                select(Trip)
                .where(
                    (Trip.user_id == user_id) | (Trip.trip_id.in_(joined_ids)),
                    Trip.archived == False,  # noqa: E712
                )
                .order_by(Trip.updated_at.desc())
            )
            return list(result.scalars().all())

    async def list_trips(self, user_id: str) -> list[Trip]:
        async with self.async_session() as session:
            joined_ids = select(TripMember.trip_id).where(TripMember.user_id == user_id)
            result = await session.execute(
                select(Trip)
                .where((Trip.user_id == user_id) | (Trip.trip_id.in_(joined_ids)))
                .order_by(Trip.updated_at.desc())
            )
            return list(result.scalars().all())

    async def update_state(self, trip_id: str, state: dict) -> None:
        async with self.async_session() as session:
            trip = await session.get(Trip, trip_id)
            if trip is None:
                raise ValueError(f"Trip {trip_id} not found")
            trip.state_json = json.dumps(state, default=str)
            trip.destination_country = state.get("destination", {}).get("country")
            trip.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def merge_state(self, trip_id: str, updates: dict) -> None:
        """Merge updates into existing trip state instead of replacing it."""
        async with self.async_session() as session:
            trip = await session.get(Trip, trip_id)
            if trip is None:
                raise ValueError(f"Trip {trip_id} not found")
            existing = json.loads(trip.state_json) if trip.state_json else {}
            existing.update(updates)
            trip.state_json = json.dumps(existing, default=str)
            trip.destination_country = existing.get("destination", {}).get("country")
            trip.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def archive_trip(self, trip_id: str) -> None:
        async with self.async_session() as session:
            trip = await session.get(Trip, trip_id)
            if trip is None:
                raise ValueError(f"Trip {trip_id} not found")
            trip.archived = True
            trip.updated_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info("Archived trip %s", trip_id)

    async def add_member(self, trip_id: str, user_id: str) -> None:
        """Add a user as a member of a trip. Idempotent — no-op if already owner or member."""
        async with self.async_session() as session:
            trip = await session.get(Trip, trip_id)
            if trip is None:
                raise ValueError(f"Trip {trip_id} not found")
            if trip.user_id == user_id:
                return  # owner is implicitly a member
            existing = await session.execute(
                select(TripMember).where(TripMember.trip_id == trip_id, TripMember.user_id == user_id)
            )
            if existing.scalar_one_or_none():
                return  # already a member
            session.add(TripMember(trip_id=trip_id, user_id=user_id))
            await session.commit()
            logger.info("User %s joined trip %s", user_id, trip_id)

    async def is_member(self, trip_id: str, user_id: str) -> bool:
        """Return True if user_id is the owner or a member of the trip."""
        async with self.async_session() as session:
            trip = await session.get(Trip, trip_id)
            if trip is None:
                return False
            if trip.user_id == user_id:
                return True
            result = await session.execute(
                select(TripMember).where(TripMember.trip_id == trip_id, TripMember.user_id == user_id)
            )
            return result.scalar_one_or_none() is not None

    async def get_joined_trips(self, user_id: str) -> list[Trip]:
        """Return trips the user joined but does not own."""
        async with self.async_session() as session:
            joined_ids = select(TripMember.trip_id).where(TripMember.user_id == user_id)
            result = await session.execute(
                select(Trip)
                .where(Trip.trip_id.in_(joined_ids), Trip.user_id != user_id)
                .order_by(Trip.updated_at.desc())
            )
            return list(result.scalars().all())

    async def close(self) -> None:
        await self.engine.dispose()

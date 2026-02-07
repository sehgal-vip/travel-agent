"""Tests for database persistence layer."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_create_and_get_trip(async_db):
    """Test basic trip creation and retrieval."""
    state = {
        "destination": {"country": "Japan", "flag_emoji": "\U0001f1ef\U0001f1f5"},
        "cities": [{"name": "Tokyo"}],
    }
    trip = await async_db.create_trip("test-1", "user-123", state)
    assert trip.trip_id == "test-1"
    assert trip.user_id == "user-123"
    assert trip.destination_country == "Japan"

    retrieved = await async_db.get_trip("test-1")
    assert retrieved is not None
    assert retrieved.trip_id == "test-1"
    loaded_state = json.loads(retrieved.state_json)
    assert loaded_state["destination"]["country"] == "Japan"


@pytest.mark.asyncio
async def test_get_active_trips(async_db):
    """Test listing active trips for a user."""
    await async_db.create_trip("trip-1", "user-1", {"destination": {"country": "Japan"}})
    await async_db.create_trip("trip-2", "user-1", {"destination": {"country": "Morocco"}})
    await async_db.create_trip("trip-3", "user-2", {"destination": {"country": "Italy"}})

    trips = await async_db.get_active_trips("user-1")
    assert len(trips) == 2
    countries = {json.loads(t.state_json)["destination"]["country"] for t in trips}
    assert countries == {"Japan", "Morocco"}


@pytest.mark.asyncio
async def test_update_state(async_db):
    """Test state updates persist correctly."""
    await async_db.create_trip("trip-1", "user-1", {"destination": {"country": "Japan"}, "onboarding_complete": False})
    await async_db.update_state("trip-1", {"destination": {"country": "Japan"}, "onboarding_complete": True})

    trip = await async_db.get_trip("trip-1")
    state = json.loads(trip.state_json)
    assert state["onboarding_complete"] is True


@pytest.mark.asyncio
async def test_archive_trip(async_db):
    """Test trip archiving."""
    await async_db.create_trip("trip-1", "user-1", {"destination": {"country": "Japan"}})
    await async_db.archive_trip("trip-1")

    active = await async_db.get_active_trips("user-1")
    assert len(active) == 0

    all_trips = await async_db.list_trips("user-1")
    assert len(all_trips) == 1
    assert all_trips[0].archived is True


@pytest.mark.asyncio
async def test_list_trips(async_db):
    """Test listing all trips including archived."""
    await async_db.create_trip("trip-1", "user-1", {"destination": {"country": "Japan"}})
    await async_db.create_trip("trip-2", "user-1", {"destination": {"country": "Morocco"}})
    await async_db.archive_trip("trip-1")

    all_trips = await async_db.list_trips("user-1")
    assert len(all_trips) == 2


@pytest.mark.asyncio
async def test_nonexistent_trip(async_db):
    """Test getting a trip that doesn't exist."""
    trip = await async_db.get_trip("nonexistent")
    assert trip is None


@pytest.mark.asyncio
async def test_update_nonexistent_raises(async_db):
    """Test updating a nonexistent trip raises ValueError."""
    with pytest.raises(ValueError):
        await async_db.update_state("nonexistent", {})


@pytest.mark.asyncio
async def test_merge_state_preserves_existing_keys(async_db):
    """Test that merge_state adds new keys without removing existing ones."""
    initial = {
        "destination": {"country": "Japan", "flag_emoji": "\U0001f1ef\U0001f1f5"},
        "cities": [{"name": "Tokyo"}],
        "onboarding_complete": True,
    }
    await async_db.create_trip("trip-m1", "user-1", initial)

    # Merge only research data — destination and cities should survive
    await async_db.merge_state("trip-m1", {"research": {"Tokyo": {"places": ["Senso-ji"]}}})

    trip = await async_db.get_trip("trip-m1")
    state = json.loads(trip.state_json)
    assert state["destination"]["country"] == "Japan"
    assert state["cities"] == [{"name": "Tokyo"}]
    assert state["onboarding_complete"] is True
    assert state["research"]["Tokyo"]["places"] == ["Senso-ji"]


@pytest.mark.asyncio
async def test_merge_state_overwrites_updated_keys(async_db):
    """Test that merge_state updates keys that are present in both old and new."""
    await async_db.create_trip("trip-m2", "user-1", {"destination": {"country": "Japan"}, "plan_status": "not_started"})
    await async_db.merge_state("trip-m2", {"plan_status": "in_progress"})

    trip = await async_db.get_trip("trip-m2")
    state = json.loads(trip.state_json)
    assert state["plan_status"] == "in_progress"
    assert state["destination"]["country"] == "Japan"


@pytest.mark.asyncio
async def test_merge_state_nonexistent_raises(async_db):
    """Test that merge_state on a missing trip raises ValueError."""
    with pytest.raises(ValueError):
        await async_db.merge_state("nonexistent", {"foo": "bar"})


# ── Trip member tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_member(async_db):
    """Test adding a member and verifying is_member returns True for both owner and member."""
    await async_db.create_trip("trip-shared", "owner-1", {"destination": {"country": "Japan"}})
    await async_db.add_member("trip-shared", "member-1")

    assert await async_db.is_member("trip-shared", "owner-1") is True
    assert await async_db.is_member("trip-shared", "member-1") is True
    assert await async_db.is_member("trip-shared", "stranger") is False


@pytest.mark.asyncio
async def test_add_member_idempotent(async_db):
    """Test that adding the same member twice doesn't raise."""
    await async_db.create_trip("trip-idem", "owner-1", {"destination": {"country": "Italy"}})
    await async_db.add_member("trip-idem", "member-1")
    await async_db.add_member("trip-idem", "member-1")  # should not raise

    assert await async_db.is_member("trip-idem", "member-1") is True


@pytest.mark.asyncio
async def test_list_trips_includes_joined(async_db):
    """Test that list_trips returns both owned and joined trips."""
    await async_db.create_trip("trip-own", "user-A", {"destination": {"country": "Japan"}})
    await async_db.create_trip("trip-other", "user-B", {"destination": {"country": "Morocco"}})
    await async_db.add_member("trip-other", "user-A")

    trips = await async_db.list_trips("user-A")
    trip_ids = {t.trip_id for t in trips}
    assert trip_ids == {"trip-own", "trip-other"}

    # get_active_trips should also include joined trips
    active = await async_db.get_active_trips("user-A")
    active_ids = {t.trip_id for t in active}
    assert active_ids == {"trip-own", "trip-other"}


@pytest.mark.asyncio
async def test_add_member_nonexistent_trip(async_db):
    """Test that adding a member to a nonexistent trip raises ValueError."""
    with pytest.raises(ValueError):
        await async_db.add_member("nonexistent-trip", "user-1")


@pytest.mark.asyncio
async def test_owner_is_implicit_member(async_db):
    """Test that add_member with owner user_id is a no-op and is_member returns True."""
    await async_db.create_trip("trip-owner", "owner-1", {"destination": {"country": "France"}})
    await async_db.add_member("trip-owner", "owner-1")  # should be no-op

    assert await async_db.is_member("trip-owner", "owner-1") is True

    # get_joined_trips should NOT include owned trips
    joined = await async_db.get_joined_trips("owner-1")
    assert len(joined) == 0

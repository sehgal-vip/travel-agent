"""Database initialisation — creates tables at startup."""

from __future__ import annotations

import logging

from src.db.persistence import TripRepository
from src.db.user_profile import UserProfileRepository

logger = logging.getLogger(__name__)


async def init_db(database_url: str) -> TripRepository:
    """Create tables and return a ready-to-use repository."""
    repo = TripRepository(database_url)
    await repo.init_db()

    # Initialize user profiles table
    user_repo = UserProfileRepository(database_url)
    await user_repo.init_table()
    await user_repo.close()

    return repo

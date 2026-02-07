"""Database initialisation — creates tables at startup."""

from __future__ import annotations

import logging

from src.db.persistence import TripRepository

logger = logging.getLogger(__name__)


async def init_db(database_url: str) -> TripRepository:
    """Create tables and return a ready-to-use repository."""
    repo = TripRepository(database_url)
    await repo.init_db()
    return repo

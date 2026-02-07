"""SQLAlchemy models for multi-trip persistence."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Trip(Base):
    __tablename__ = "trips"

    trip_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_country: Mapped[str | None] = mapped_column(String(128))
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    archived: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("idx_trips_user", "user_id"),
        Index("idx_trips_active", "user_id", "archived"),
    )

    def __repr__(self) -> str:
        return f"<Trip {self.trip_id} user={self.user_id} country={self.destination_country}>"

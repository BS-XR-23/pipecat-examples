import uuid
from datetime import datetime, date
from sqlalchemy import (
    Column, String, Boolean, Numeric, Integer,
    DateTime, Date, Text, Enum as PgEnum
)
from sqlalchemy.dialects.postgresql import UUID
from db.database import Base
import enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class RoomType(str, enum.Enum):
    standard = "standard"
    deluxe   = "deluxe"
    suite    = "suite"

class RoomStatus(str, enum.Enum):
    available    = "available"
    booked       = "booked"
    maintenance  = "maintenance"
    out_of_order = "out_of_order"

class BookingStatus(str, enum.Enum):
    confirmed = "confirmed"
    cancelled = "cancelled"
    completed = "completed"

# ── Room ───────────────────────────────────────────────────────────────────────

class Room(Base):
    __tablename__ = "rooms"
    __table_args__ = {"extend_existing": True}

    room_id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_number     = Column(String(10),         nullable=False, unique=True, index=True)
    room_type       = Column(PgEnum(RoomType),   nullable=False)
    status          = Column(PgEnum(RoomStatus), nullable=False, default=RoomStatus.available)
    price_per_night = Column(Numeric(10, 2),     nullable=False)
    currency        = Column(String(5),          default="USD")
    capacity        = Column(Integer,            default=2)
    amenities       = Column(Text,               nullable=True)
    description     = Column(Text,               nullable=True)

    def __repr__(self):
        return f"<Room {self.room_number} — {self.room_type.value} — {self.status.value}>"


# ── Booking ────────────────────────────────────────────────────────────────────

class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = {"extend_existing": True}

    booking_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_number  = Column(String(10),         nullable=False, index=True)  # stored directly, no FK
    guest_name   = Column(String(120),        nullable=False)
    guest_phone  = Column(String(20),         nullable=False, index=True)
    check_in     = Column(Date,               nullable=False)
    check_out    = Column(Date,               nullable=False)
    guests_count = Column(Integer,            default=1)
    total_price  = Column(Numeric(10, 2),     nullable=True)
    status       = Column(PgEnum(BookingStatus), nullable=False, default=BookingStatus.confirmed)
    created_at   = Column(DateTime,           default=datetime.utcnow)

    def __repr__(self):
        return f"<Booking {self.booking_id} — Room {self.room_number} — {self.status.value}>"
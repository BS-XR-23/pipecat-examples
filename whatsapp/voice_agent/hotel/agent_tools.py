from sqlalchemy.orm import Session
from typing import Optional
from datetime import date as date_type
from db.models import Room, Booking, RoomType, RoomStatus, BookingStatus
import uuid
from utils.helpers import normalize_phone


def search_available_rooms(
    db: Session,
    check_in: str,
    check_out: str,
    guests: int = 1
) -> dict:
    """Find rooms available between check_in and check_out with enough capacity."""
    try:
        check_in_date  = date_type.fromisoformat(check_in)
        check_out_date = date_type.fromisoformat(check_out)
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    booked_room_numbers = (
        db.query(Booking.room_number)
        .filter(
            Booking.status == BookingStatus.confirmed,
            Booking.check_in  < check_out_date,
            Booking.check_out > check_in_date,
        )
        .subquery()
    )

    rooms = (
        db.query(Room)
        .filter(
            Room.capacity >= guests,
            Room.room_number.not_in(booked_room_numbers),
            Room.status != RoomStatus.maintenance,  # only exclude physically unusable rooms
        )
        .all()
    )

    if not rooms:
        return {"rooms": [], "available_count": 0}

    return {
        "rooms": [
            {
                "room_number": r.room_number,
                "type":        r.room_type.value.capitalize(),
                "price":       float(r.price_per_night),
                "currency":    r.currency,
                "capacity":    r.capacity,
                "amenities":   r.amenities,
            }
            for r in rooms
        ],
        "check_in":        check_in,
        "check_out":       check_out,
        "available_count": len(rooms),
    }


def get_room_prices(db: Session, room_type: Optional[str] = None) -> dict:
    """Get distinct room type prices."""
    query = db.query(Room)

    if room_type:
        try:
            rt = RoomType(room_type.lower())
            query = query.filter(Room.room_type == rt)
        except ValueError:
            return {"error": f"Unknown room type: {room_type}"}

    seen   = set()
    prices = []
    for r in query.all():
        if r.room_type not in seen:
            seen.add(r.room_type)
            prices.append({
                "type":      r.room_type.value.capitalize(),
                "price":     float(r.price_per_night),
                "currency":  r.currency,
                "capacity":  r.capacity,
                "amenities": r.amenities,
            })

    return {"prices": prices}


def make_booking(
    db: Session,
    guest_name: str,
    guest_phone: str,
    room_number: str,
    check_in: str,
    check_out: str,
    guests_count: int = 1,
) -> dict:
    """Book a room by room_number."""
    try:
        check_in_date  = date_type.fromisoformat(check_in)
        check_out_date = date_type.fromisoformat(check_out)
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    if check_out_date <= check_in_date:
        return {"error": "Check-out date must be after check-in date."}

    room = db.query(Room).filter(Room.room_number == room_number).first()
    if not room:
        return {"error": f"Room {room_number} does not exist."}

    if room.status == RoomStatus.maintenance:
        return {"error": f"Room {room_number} is currently under maintenance and cannot be booked."}

    conflict = db.query(Booking).filter(
        Booking.room_number == room_number,
        Booking.status      == BookingStatus.confirmed,
        Booking.check_in    < check_out_date,
        Booking.check_out   > check_in_date,
    ).first()

    if conflict:
        return {"error": f"Room {room_number} is already booked for the selected dates."}

    nights      = (check_out_date - check_in_date).days
    total_price = float(room.price_per_night) * nights

    booking = Booking(
        room_number  = room_number,
        guest_name   = guest_name,
        guest_phone  = normalize_phone(guest_phone),    # ← always store normalized
        check_in     = check_in_date,
        check_out    = check_out_date,
        guests_count = guests_count,
        total_price  = total_price,
        status       = BookingStatus.confirmed,
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)

    return {
        "booking_id":   str(booking.booking_id),
        "room_number":  booking.room_number,
        "room_type":    room.room_type.value.capitalize(),
        "guest_name":   booking.guest_name,
        "guest_phone":  booking.guest_phone,
        "check_in":     str(booking.check_in),
        "check_out":    str(booking.check_out),
        "guests_count": booking.guests_count,
        "nights":       nights,
        "total_price":  total_price,
        "currency":     room.currency,
        "status":       booking.status.value,
    }


def cancel_booking(db: Session, booking_id: str, guest_phone: str) -> dict:
    """Cancel a booking — deletes the record after verifying guest_phone."""
    try:
        bid = uuid.UUID(booking_id)
    except ValueError:
        return {"error": "Invalid booking ID format."}

    booking = db.query(Booking).filter(Booking.booking_id == bid).first()

    if not booking:
        return {"error": "Booking not found."}

    if normalize_phone(booking.guest_phone) != normalize_phone(guest_phone):
        return {"error": "Phone number does not match our records for this booking."}

    if booking.status == BookingStatus.completed:
        return {"error": "Completed bookings cannot be cancelled."}

    room_number = booking.room_number
    db.delete(booking)
    db.commit()

    return {
        "booking_id":  str(bid),
        "room_number": room_number,
        "status":      "cancelled",
        "message":     f"Booking {bid} has been successfully cancelled.",
    }


def check_booking_status(db: Session, booking_id: str) -> dict:
    """Look up a booking by its ID."""
    try:
        bid = uuid.UUID(booking_id)
    except ValueError:
        return {"error": "Invalid booking ID format."}

    booking = db.query(Booking).filter(Booking.booking_id == bid).first()

    if not booking:
        return {"error": "Booking not found."}

    room = db.query(Room).filter(Room.room_number == booking.room_number).first()

    return {
        "booking_id":   str(booking.booking_id),
        "room_number":  booking.room_number,
        "room_type":    room.room_type.value.capitalize() if room else "Unknown",
        "guest_name":   booking.guest_name,
        "guest_phone":  booking.guest_phone,
        "check_in":     str(booking.check_in),
        "check_out":    str(booking.check_out),
        "guests_count": booking.guests_count,
        "total_price":  float(booking.total_price) if booking.total_price else None,
        "currency":     room.currency if room else "USD",
        "status":       booking.status.value,
    }
"""
Run once to create all tables and seed rooms.
Usage: python -m db.init_db
"""
from db.database import engine, SessionLocal
from db.models import Base, Room, RoomType, RoomStatus
from loguru import logger


def create_tables():
    logger.info("Creating tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created.")


def seed_rooms():
    db = SessionLocal()
    try:
        if db.query(Room).count() > 0:
            logger.info("Rooms already seeded. Skipping.")
            return

        rooms = [
            Room(
                room_number="101", room_type=RoomType.standard,
                price_per_night=80, currency="USD", capacity=2,
                amenities="WiFi, TV, AC",
                description="Cozy standard room with garden view.",
                status=RoomStatus.available,
            ),
            Room(
                room_number="102", room_type=RoomType.standard,
                price_per_night=80, currency="USD", capacity=2,
                amenities="WiFi, TV, AC",
                description="Cozy standard room with garden view.",
                status=RoomStatus.available,
            ),
            Room(
                room_number="201", room_type=RoomType.deluxe,
                price_per_night=120, currency="USD", capacity=2,
                amenities="WiFi, TV, AC, Mini-bar, Sea View",
                description="Spacious deluxe room with ocean view.",
                status=RoomStatus.available,
            ),
            Room(
                room_number="202", room_type=RoomType.deluxe,
                price_per_night=120, currency="USD", capacity=2,
                amenities="WiFi, TV, AC, Mini-bar, Sea View",
                description="Spacious deluxe room with ocean view.",
                status=RoomStatus.available,
            ),
            Room(
                room_number="301", room_type=RoomType.suite,
                price_per_night=200, currency="USD", capacity=4,
                amenities="WiFi, TV, AC, Mini-bar, Jacuzzi, Living Room",
                description="Luxury suite with panoramic view and private jacuzzi.",
                status=RoomStatus.available,
            ),
            Room(
                room_number="302", room_type=RoomType.suite,
                price_per_night=200, currency="USD", capacity=4,
                amenities="WiFi, TV, AC, Mini-bar, Jacuzzi, Living Room",
                description="Luxury suite with panoramic view and private jacuzzi.",
                status=RoomStatus.available,
            ),
        ]

        db.add_all(rooms)
        db.commit()
        logger.info(f"Seeded {len(rooms)} rooms.")

    except Exception as e:
        db.rollback()
        logger.error(f"Seeding failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    create_tables()
    seed_rooms()
    logger.info("Database initialization complete.")


# sudo -u postgres psql -d hotel_db -c "ALTER TYPE roomstatus ADD VALUE 'booked';"
# psql -h 127.0.0.1 -U postgres -d whatsapp_chatbot
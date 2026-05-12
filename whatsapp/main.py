from db.database import create_tables
from db_bank.users.seed_data import seed_data


def main():
    print("🚀 Starting application...")

    # 1. Create tables ONLY ONCE
    create_tables()

    # 2. Seed database
    seed_data()

    print("✅ Database ready")

    # 3. Start your app here
    # from server import app
    # uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
from db.database import Base, engine
from db_bank.users import model  # IMPORTANT: only import models here

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")

if __name__ == "__main__":
    init_db()

# uv run python -m db.init_db
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# psql -h 127.0.0.1 -U postgres -d whatsapp_chatbot
# sudo -u postgres psql
# \c hotel_db

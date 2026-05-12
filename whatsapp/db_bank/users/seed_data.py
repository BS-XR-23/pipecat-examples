from db.database import SessionLocal
from db.users.model import User, Account, Nominee, Transaction
import uuid


def seed_data():
    db = SessionLocal()

    try:
        print("🔥 SEED SCRIPT STARTED")

        user = db.query(User).first()

        if not user:
            user = User(
                user_name="Test User",
                phone_number="01700000000",
                email="test@example.com",
                hashed_password="hashed_password",
                is_verified=True,
                is_active=True,
            )
            db.add(user)
            db.flush()   # 🔥 IMPORTANT
            db.commit()
            db.refresh(user)
            print("✅ User created")
        else:
            print("⚠️ User already exists")

        account = db.query(Account).first()

        if not account:
            account = Account(
                user_id=user.user_id,
                account_num=str(uuid.uuid4())[:12],
                account_type="savings",
                balance=1000,
                currency="BDT",
                is_active=True,
            )
            db.add(account)
            db.commit()
            db.refresh(account)
            print("✅ Account created")
        else:
            print("⚠️ Account already exists")

        nominee = db.query(Nominee).first()

        if not nominee:
            nominee = Nominee(
                user_id=user.user_id,
                nominee_name="Test Nominee",
                relation="Brother",
                phone_number="01800000000",
            )
            db.add(nominee)
            db.commit()
            print("✅ Nominee created")
        else:
            print("⚠️ Nominee already exists")

        tx = db.query(Transaction).first()

        if not tx:
            tx = Transaction(
                account_id=account.account_id,
                amount=500,
                transaction_type="credit",
                description="Initial deposit",
            )
            db.add(tx)
            db.commit()
            print("✅ Transaction created")
        else:
            print("⚠️ Transaction already exists")

        print("🎉 Seeding completed successfully!")

    except Exception as e:
        db.rollback()
        print("❌ Error:", e)

    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
from sqlalchemy.orm import Session
from typing import Optional
from datetime import date as date_type
from db_bank.users.model import User, Account, Ticket
import uuid

def get_user_info(db: Session, phone_number: str) -> dict:
    user = db.query(User).filter(User.phone_number == phone_number).first()

    if not user:
        return {"error": "User not found"}

    return {
        "user_name": user.user_name,
        "email": user.email,
        "is_verified": user.is_verified,
    }

def get_account_balance(db: Session, phone_number: str) -> dict:
    user = db.query(User).filter(User.phone_number == phone_number).first()

    if not user:
        return {"error": "User not found"}

    account = db.query(Account).filter(Account.user_id == user.user_id).first()

    if not account:
        return {"error": "Account not found"}

    return {
        "account_number": account.account_num,
        "balance": float(account.balance),
        "currency": account.currency,
    }

def create_ticket(db: Session, phone_number: str, category: str, short_description: str) -> dict:
    user = db.query(User).filter(User.phone_number == phone_number).first()

    ticket = Ticket(
        user_id=user.user_id if user else None,
        category=category,
        short_description=short_description,
        status="open"
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return {
        "ticket_id": str(ticket.ticket_id),
        "category": ticket.category,
        "status": ticket.status,
    }


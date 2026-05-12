from sqlalchemy import Column, String, Boolean, ForeignKey, Numeric, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from db.database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_name = Column(String(100), nullable=False)
    phone_number = Column(String(20), unique=True, nullable=False)
    email = Column(String(255), nullable=True)

    hashed_password = Column(String, nullable=False)

    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    accounts = relationship("Account", back_populates="user", cascade="all, delete")
    nominees = relationship("Nominee", back_populates="user", cascade="all, delete")

    # ✅ NEW
    tickets = relationship("Ticket", back_populates="user", cascade="all, delete")


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = {"extend_existing": True}

    account_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"))

    account_num = Column(String(30), unique=True, nullable=False)
    account_type = Column(String(20), default="savings")

    balance = Column(Numeric(15, 2), default=0.00)
    currency = Column(String(10), default="BDT")

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete")


class Nominee(Base):
    __tablename__ = "nominees"
    __table_args__ = {"extend_existing": True}

    nominee_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"))

    nominee_name = Column(String(100), nullable=False)
    relation = Column(String(50))
    phone_number = Column(String(20))

    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="nominees")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = {"extend_existing": True}

    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.account_id", ondelete="CASCADE"))

    amount = Column(Numeric(15, 2), nullable=False)
    transaction_type = Column(String(10))  # credit / debit

    description = Column(Text)

    created_at = Column(DateTime, server_default=func.now())

    account = relationship("Account", back_populates="transactions")


# ✅ NEW TABLE
class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = {"extend_existing": True}

    ticket_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True)

    category = Column(String(50), nullable=False)
    status = Column(String(20), default="open")

    short_description = Column(Text, nullable=False)

    date_created = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="tickets")
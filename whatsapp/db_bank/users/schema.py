from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from decimal import Decimal


class UserBase(BaseModel):
    user_name: str
    phone_number: str
    email: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    user_id: UUID
    is_verified: bool
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AccountBase(BaseModel):
    account_num: str
    account_type: str = "savings"


class AccountCreate(AccountBase):
    pass


class AccountResponse(AccountBase):
    account_id: UUID
    balance: Decimal
    currency: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class NomineeBase(BaseModel):
    nominee_name: str
    relation: Optional[str] = None
    phone_number: Optional[str] = None


class NomineeCreate(NomineeBase):
    pass


class NomineeResponse(NomineeBase):
    nominee_id: UUID

    model_config = ConfigDict(from_attributes=True)


class TransactionBase(BaseModel):
    amount: Decimal
    transaction_type: str
    description: Optional[str] = None


class TransactionCreate(TransactionBase):
    pass


class TransactionResponse(TransactionBase):
    transaction_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ✅ NEW TICKET SCHEMAS
class TicketBase(BaseModel):
    category: str
    status: str = "open"
    short_description: str


class TicketCreate(TicketBase):
    pass


class TicketResponse(TicketBase):
    ticket_id: UUID
    user_id: UUID
    date_created: datetime

    model_config = ConfigDict(from_attributes=True)


# ✅ UPDATED FULL USER RESPONSE
class UserFullResponse(UserResponse):
    accounts: List[AccountResponse] = []
    nominees: List[NomineeResponse] = []
    tickets: List[TicketResponse] = []

    model_config = ConfigDict(from_attributes=True)
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class WhatsAppTextMessage(BaseModel):
    body: str


class WhatsAppMessage(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[WhatsAppTextMessage] = None


class WhatsAppMessageValue(BaseModel):
    messaging_product: str
    metadata: Optional[dict[str, Any]] = None
    contacts: Optional[List[dict[str, Any]]] = None
    messages: Optional[List[WhatsAppMessage]] = None


class WhatsAppMessageChange(BaseModel):
    field: str
    value: WhatsAppMessageValue


class WhatsAppMessageEntry(BaseModel):
    id: str
    changes: List[WhatsAppMessageChange]


class WhatsAppMessageWebhookRequest(BaseModel):
    object: str
    entry: List[WhatsAppMessageEntry]

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict


class WhatsAppCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    session: Optional[Any] = None   # ← was str, but payload sends an SDP dict here
    event: Optional[str] = None
    status: Optional[str] = None


class WhatsAppError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: Optional[int] = None
    title: Optional[str] = None
    href: Optional[str] = None


class WhatsAppCallValue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messaging_product: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    contacts: Optional[List[Dict[str, Any]]] = None
    calls: Optional[List[WhatsAppCall]] = None
    errors: Optional[List[WhatsAppError]] = None


class WhatsAppCallChange(BaseModel):          # ← new
    model_config = ConfigDict(extra="ignore")

    field: Optional[str] = None
    value: Optional[WhatsAppCallValue] = None


class WhatsAppCallEntry(BaseModel):           # ← new
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    changes: Optional[List[WhatsAppCallChange]] = None  # ← now an attribute, not a dict key


class WhatsAppWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    object: Optional[str] = None
    entry: Optional[List[WhatsAppCallEntry]] = None
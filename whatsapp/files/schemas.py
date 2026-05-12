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
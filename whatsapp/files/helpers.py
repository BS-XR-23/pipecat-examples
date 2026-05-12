import re
import json
from loguru import logger

def safe_parse_json(text: str):
    try:
        text = text.strip()
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception as e:
        logger.warning(f"[JSON PARSE FAILED] {e}")
        return None

def extract_json(text: str):
    """
    Extract JSON safely from LLM output (handles messy outputs)
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None


def extract_phone_number(text: str):

    if not text:
        return None

    cleaned = re.sub(r"[^\d+]", "", text)

    # Match +880 format
    match = re.search(r"\+?8801[3-9]\d{8}", cleaned)
    if match:
        number = match.group(0)
        return "0" + number[-10:]  # normalize to 01XXXXXXXXX

    # Match local format
    match = re.search(r"01[3-9]\d{8}", cleaned)
    if match:
        return match.group(0)

    return None

def is_valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"01[3-9]\d{8}", phone))
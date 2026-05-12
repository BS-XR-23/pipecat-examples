import re
import json
from loguru import logger


# ── JSON helpers ───────────────────────────────────────────────────────────────

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
    """Extract JSON safely from LLM output (handles messy outputs)."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None


# ── Phone number helpers ───────────────────────────────────────────────────────

_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "oh": "0",  # common spoken alias for zero
}

_MULTIPLIER_WORDS = {
    "double": 2,
    "triple": 3,
    "quadruple": 4,
}


def _spoken_to_digits(text: str) -> str:
    """
    Convert spoken phone number words to a digit string.
    Handles:
      - Plain digit words:      "zero one seven..." → "017..."
      - Multiplier phrases:     "triple zero"       → "000"
                                "double five"       → "55"
    """
    tokens = re.split(r"[\s,\-]+", text.lower().strip())
    digits = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # "double/triple/quadruple <digit_word_or_single_digit>"
        # Handles both "triple zero" and "triple 0" (STT may output either)
        if token in _MULTIPLIER_WORDS and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            if next_token in _WORD_TO_DIGIT:
                # word form: "triple zero" → "000"
                digits.append(_WORD_TO_DIGIT[next_token] * _MULTIPLIER_WORDS[token])
                i += 2
                continue
            if re.fullmatch(r"\d", next_token):
                # digit char form: "triple 0" → "000"
                digits.append(next_token * _MULTIPLIER_WORDS[token])
                i += 2
                continue

        # Plain digit word
        if token in _WORD_TO_DIGIT:
            digits.append(_WORD_TO_DIGIT[token])
            i += 1
            continue

        # Already a numeric string
        if re.fullmatch(r"\d+", token):
            digits.append(token)
            i += 1
            continue

        i += 1

    return "".join(digits)


def extract_phone_number(text: str) -> str | None:
    """
    Extracts a Bangladeshi phone number from text.
    Handles:
      - Spoken digits:      "zero one seven zero zero..."
      - Multiplier phrases: "zero one seven triple zero double five..."
      - Spaced/dashed:      "01700 000 000", "017-0000-0000"
      - Country code:       +8801XXXXXXXXX, 008801XXXXXXXXX
      - Plain numeric:      01XXXXXXXXX
    Returns a normalized 11-digit string (01XXXXXXXXX) or None.
    """
    if not text:
        return None

    # 1. Try spoken-word conversion first (handles multipliers like "triple zero")
    converted = _spoken_to_digits(text)
    match = re.search(r"01[3-9]\d{8}", converted)
    if match:
        return match.group(0)

    # 2. Fall back to stripping all non-digits for numeric / country-code formats
    digits_only = re.sub(r"\D", "", text)

    # Normalize country code variants → local 11-digit format
    if digits_only.startswith("880") and len(digits_only) == 13:
        digits_only = "0" + digits_only[3:]       # 8801XXXXXXXXX → 01XXXXXXXXX
    elif digits_only.startswith("0088") and len(digits_only) == 14:
        digits_only = "0" + digits_only[4:]       # 00881XXXXXXXXX → 01XXXXXXXXX
    elif digits_only.startswith("+8801"):
        digits_only = "0" + digits_only[-10:]     # +8801XXXXXXXXX → 01XXXXXXXXX

    match = re.search(r"01[3-9]\d{8}", digits_only)
    if match:
        return match.group(0)

    return None


def is_valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"01[3-9]\d{8}", phone))


# ── Date / number helpers ──────────────────────────────────────────────────────

def extract_date(text: str):
    """Extract date as YYYY-MM-DD from natural text."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)
    match = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", text)
    if match:
        d, m, y = match.group(1), match.group(2), match.group(3)
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return None


def extract_number(text: str):
    # Strip dates first so numbers inside dates aren't matched
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}", "", text)
    match = re.search(r"\b([1-9][0-9]?)\b", text)
    return int(match.group(1)) if match else None


# ── Session helpers ────────────────────────────────────────────────────────────

def normalize_session_id(value: str) -> str:
    if not value:
        return value

    digits = re.sub(r"\D", "", value)

    if digits.startswith("01") and len(digits) == 11:
        digits = "88" + digits

    return digits

def normalize_phone(phone: str) -> str:
    """Normalize BD phone numbers to consistent 11-digit local format."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)           # strip +, spaces, dashes
    if digits.startswith("880") and len(digits) == 13:
        digits = "0" + digits[3:]               # +8801701001398 → 01701001398
    return digits
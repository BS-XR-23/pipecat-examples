from typing import TypedDict, Optional, Dict, Any
from random import choice
import json
import os
from dotenv import load_dotenv
import re
import copy
from loguru import logger
from ollama import AsyncClient
from db.database import SessionLocal
from utils.helpers import extract_phone_number, safe_parse_json, extract_json
from agents.bank.agent_tools import get_account_balance, get_user_info, create_ticket
from services.rag_services.rag_service import get_rag_context
from utils.agent_logger import log_tool_call
from response import (
    GREETING_RESPONSES,
    BALANCE_MISSING_PHONE,
    BALANCE_SUCCESS,
    USER_INFO_MISSING_PHONE,
    USER_INFO_SUCCESS,
    TICKET_CONFIRM_INTENT,
    TICKET_ASK_ISSUE,
    TICKET_ASK_PHONE,
    TICKET_FINAL,
    HUMAN_HANDOFF,
    THANK_YOU_RESPONSES
)


load_dotenv()
VENDOR_ID = int(os.getenv("VENDOR_ID", "0"))


# -------------------- STATE --------------------
class BotState(TypedDict):
    session_id: str
    message: str
    agent_name: str
    agent: Any
    intent: Optional[str]
    response: Optional[str]
    memory: Dict[str, Any]


# -------------------- OLLAMA CLIENT --------------------
ollama_client = AsyncClient()


def safe_state(state):
    state = copy.deepcopy(state)
    return state


def ensure_memory(state: BotState):
    if "memory" not in state:
        state["memory"] = {}

    memory = state["memory"]

    memory.setdefault("profile", {})
    memory.setdefault("history", [])
    memory.setdefault("ticket", {})
    memory.setdefault("flow", {})

    flow = memory["flow"]
    flow.setdefault("active", None)
    flow.setdefault("step", None)
    flow.setdefault("expandable", False)
    flow.setdefault("last_expand_offer", False)

    if not isinstance(state.get("intent"), str):
        state["intent"] = "RAG"

    return state


async def should_route_cancel_flow_to_rag_state(state: BotState) -> bool:
    try:
        flow = state["memory"]["flow"]
        active_flow = flow.get("active") or "None"
        history = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in history
        )

        prompt = CANCEL_QUESTION_PROMPT.format(
            active_flow=active_flow,
            history=history_text,
            message=state.get("message", "")
        )

        client = AsyncClient()
        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.9},
        )

        raw = res.response.strip()
        logger.info(f"[CANCEL QUESTION RAW] {raw}")
        data = safe_parse_json(raw)
        if not isinstance(data, dict):
            data = extract_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[CANCEL QUESTION PARSE] expected JSON object, got %s",
                type(data).__name__
            )
            return False

        return bool(data.get("route_to_rag", False))
    except Exception as e:
        logger.warning(f"[CANCEL QUESTION NODE ERROR] {e}")
        return False


CANCEL_QUESTION_PROMPT = """
You are an intent router helper for a banking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{"route_to_rag":<true|false>,"reason":"<short explanation>"}

The user message may contain both a cancellation signal and a factual question.

Instructions:
- If the user is only asking to cancel or stop the current flow, return false.
- If the user is also asking a factual, informational question about banking products,
  services, policies, or procedures, return true.
- Do not rely on keyword matching alone; use meaning and intent.
- If the message is ambiguous, prefer cancellation unless a clear informational
  request is present.

Active flow: {active_flow}

Conversation history:
{history}

User message:
{message}
"""


# -------------------- INTENT PROMPT --------------------
# Key improvements over the original:
#   1. Active flow + step injected so the LLM understands multi-turn context.
#   2. CANCEL_FLOW intent added so explicit negations ("no", "don't", "forget it",
#      "do not create a ticket") are recognised even mid-flow.
#   3. Negative examples are provided for TICKET to prevent false positives when
#      a user simply asks *about* complaints/processes.
INTENT_PROMPT = """
You are a routing system for a banking chatbot.

Return ONLY valid minified JSON. No explanation. No text. No markdown.

Format:
{{"intent":"<INTENT>","confidence":<0.0-1.0>}}

Allowed intents:
- GREETING
- BALANCE
- USER_INFO
- TICKET
- RAG
- EXPAND
- CONTROL
- SMALL_TALK
- CANCEL_FLOW

---------- ACTIVE CONTEXT ----------
Active flow  : {active_flow}
Flow step    : {flow_step}
------------------------------------

Routing Rules:

- CANCEL_FLOW:
  The user is EXPLICITLY rejecting, cancelling, or opting out of the current
  active flow or action. This takes highest priority over all other intents
  when an active flow exists.
  Examples: "no", "don't", "do not", "forget it", "cancel", "skip", "never mind",
            "I don't want a ticket", "do not create a ticket", "stop", "no thanks".
  Use CANCEL_FLOW whenever active_flow is not "None" and the user is pushing
  back, negating, or changing direction.

- GREETING:
  User is saying hello, hi, good morning, good evening, or starting conversation.

- SMALL_TALK:
  Casual non-banking chat, such as thanks, bye, okay, got it, casual replies,
  or emotional filler.

- CONTROL:
  Short acknowledgements like yes, ok, okay, sure (ONLY when no active flow
  and the user is not cancelling anything).

- EXPAND:
  User wants more detail or clarification of the previous assistant answer.

- BALANCE:
  Asking about account balance, available funds, or money inquiry.

- USER_INFO:
  Asking for profile, account details, email, or personal info.

- TICKET:
  User is PERSONALLY experiencing a problem RIGHT NOW and wants to report it.
  Examples: "my card is blocked", "I can't login", "my transaction failed".
  NOT TICKET: informational questions about complaint processes or requirements.

- RAG:
  Any informational or knowledge-based question about banking services, policies,
  procedures, or requirements.
  Examples: "what documents do I need", "how do I submit a complaint",
            "tell me about loans offered by ABB bank".
  ALWAYS use RAG when the user asks "what", "how", "tell me", "explain",
  "what are the requirements", "tell me about".

Critical Rules:
- If active_flow is not "None" and the message contains negation words
  ("no", "don't", "do not", "not", "never", "cancel", "stop", "forget",
  "never mind", "skip") → ALWAYS return CANCEL_FLOW.
- If the message includes a factual or informational question, route to RAG.
- If active_flow is set and the message appears to both cancel the flow and ask
  a question, use a secondary semantic check to decide whether the user is
  asking for information rather than only cancelling.
- Only use TICKET if the user is personally reporting an active issue,
  NOT asking about a process.

Conversation history (last 6 turns):
{history}

User message:
{message}
"""


async def detect_intent(state: BotState):
    state = ensure_memory(state)
    state["intent"] = "RAG"  # safe default

    try:
        client = AsyncClient()

        flow = state["memory"]["flow"]
        active_flow = flow.get("active") or "None"
        flow_step = flow.get("step") or "None"

        history = state["memory"]["history"][-6:]
        history_text = "\n".join(
            f"{h['role'].upper()}: {h['content']}"
            for h in history
        )

        prompt = INTENT_PROMPT.format(
            active_flow=active_flow,
            flow_step=flow_step,
            history=history_text,
            message=state["message"]
        )

        res = await client.generate(
            model="gemma4:e2b",
            prompt=prompt,
            options={"temperature": 0, "top_p": 0.9}
        )

        raw = res.response.strip()
        logger.info(f"[INTENT RAW] {raw}")

        data = safe_parse_json(raw)

        if not data:
            logger.warning(f"[INTENT FALLBACK - no data] raw: {raw}")
            state["intent"] = "RAG"
        else:
            allowed = {
                "GREETING", "BALANCE", "USER_INFO",
                "TICKET", "RAG", "EXPAND",
                "CONTROL", "SMALL_TALK", "CANCEL_FLOW"
            }

            raw_intent = str(data.get("intent", "RAG")).strip().upper()
            confidence = float(data.get("confidence", 1.0))

            # Low-confidence responses fall back to RAG
            if confidence < 0.5:
                logger.warning(
                    f"[INTENT LOW CONFIDENCE] {raw_intent} @ {confidence:.2f}, "
                    "falling back to RAG"
                )
                state["intent"] = "RAG"
            else:
                state["intent"] = raw_intent if raw_intent in allowed else "RAG"

            logger.info(
                f"[INTENT RESOLVED] {state['intent']} "
                f"(confidence={confidence:.2f})"
            )

            if state["intent"] == "CANCEL_FLOW" and active_flow != "None":
                if await should_route_cancel_flow_to_rag_state(state):
                    logger.info("[INTENT] CANCEL_FLOW message also contains an informational request; routing to RAG")
                    state["intent"] = "RAG"

    except Exception as e:
        logger.warning(f"[INTENT NODE ERROR] {e}, falling back to RAG")
        state["intent"] = "RAG"

    return state


# -------------------- ROUTER --------------------
# Key changes:
#   • CANCEL_FLOW always breaks out of any active flow and routes to control_node,
#     which handles the clean exit message.
#   • Active-flow locks are still in place for mid-flow continuation (e.g. collecting
#     a phone number), but they no longer override explicit cancellation signals.
def router(state: BotState):
    intent = state.get("intent", "RAG")

    if not isinstance(intent, str):
        intent = "RAG"

    intent = intent.strip().upper()

    # Cancellation always wins – route to control_node for a graceful exit
    if intent == "CANCEL_FLOW":
        return "CONTROL"

    flow = state.get("memory", {}).get("flow", {})
    active = flow.get("active")

    # Continue an in-progress flow only for neutral/continuation intents
    # (i.e. the user didn't cancel and just sent the next expected message)
    continuation_intents = {"CONTROL", "SMALL_TALK", "EXPAND"}
    if active and intent in continuation_intents:
        return active  # resume the active flow node

    # If there is an active flow AND the user sent something that looks like
    # a new question (RAG, BALANCE, USER_INFO) rather than flow input,
    # honour the new intent and let the active flow lapse gracefully.
    # The node itself is responsible for clearing flow state if interrupted.
    allowed = {
        "GREETING", "BALANCE", "USER_INFO",
        "TICKET", "RAG", "EXPAND",
        "CONTROL", "SMALL_TALK", "CANCEL_FLOW"
    }

    return intent if intent in allowed else "RAG"


# -------------------- CONTROL NODE --------------------
# Now handles CANCEL_FLOW exits in addition to yes/no acknowledgements.
async def control_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    flow = state["memory"]["flow"]
    intent = state.get("intent", "CONTROL").upper()

    # ── Cancellation exit ──────────────────────────────────────────────────
    if intent == "CANCEL_FLOW":
        active_flow = flow.get("active")
        flow["active"] = None
        flow["step"] = None
        flow["expandable"] = False
        flow["last_expand_offer"] = False

        if active_flow == "TICKET":
            state["response"] = (
                "No problem, I won't raise a ticket. "
                "Let me know if there's anything else I can help you with."
            )
        else:
            state["response"] = "Alright, I've cancelled that. How else can I help you?"
        return state

    # ── Standard yes/no handling ───────────────────────────────────────────
    text = state["message"].lower().strip()

    if text in ["no", "nope"]:
        flow["expandable"] = False
        flow["last_expand_offer"] = False
        state["response"] = "Alright 👍"
        return state

    if text in ["yes", "yeah", "sure", "ok", "okay"]:
        if flow.get("expandable") and flow.get("last_expand_offer"):
            state["intent"] = "EXPAND"
            return await expand_node(state)

        state["response"] = "Okay 👍 How can I help you?"
        return state

    state["response"] = "Alright 👍"
    state.setdefault("intent", "CONTROL")
    return state


# -------------------- TICKET NODE --------------------
# Key changes:
#   • Step "await_confirmation" now uses intent-based logic instead of hardcoded
#     keyword matching, so "do not create a ticket" is handled correctly.
#   • Any incoming CANCEL_FLOW intent at any step exits the flow cleanly.
async def ticket_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]
    ticket = state["memory"]["ticket"]
    intent = state.get("intent", "").upper()

    # ── Bail out immediately on any cancellation signal ────────────────────
    if intent == "CANCEL_FLOW":
        flow["active"] = None
        flow["step"] = None
        state["response"] = (
            "No problem, I won't raise a ticket. "
            "Let me know if there's anything else I can help you with."
        )
        return state

    step = flow.get("step") or "confirm"

    # ── Step: initial confirmation ask ─────────────────────────────────────
    if step == "confirm":
        state["response"] = choice(TICKET_CONFIRM_INTENT)
        flow["active"] = "TICKET"
        flow["step"] = "await_confirmation"
        return state

    # ── Step: interpret user's yes/no/human-agent response ────────────────
    # Uses intent instead of keyword matching to avoid false positives like
    # "do not create a ticket" being matched by the word "ticket".
    if step == "await_confirmation":
        # User explicitly said yes / ok / sure (CONTROL with positive tone)
        affirmative_intents = {"CONTROL", "SMALL_TALK"}
        text_lower = state["message"].lower()

        # Detect human-agent request regardless of intent
        if any(w in text_lower for w in ["human", "agent", "representative", "person"]):
            flow["active"] = None
            flow["step"] = None
            state["response"] = choice(HUMAN_HANDOFF)
            return state

        # Genuine affirmation: short message + affirmative intent
        is_affirmative = (
            intent in affirmative_intents
            and any(
                w in text_lower
                for w in ["yes", "yeah", "ok", "okay", "sure", "please", "go ahead", "create"]
            )
        )

        if is_affirmative:
            flow["step"] = "collect_issue"
            state["response"] = choice(TICKET_ASK_ISSUE)
        else:
            # Anything that isn't a clear yes → treat as cancellation
            flow["active"] = None
            flow["step"] = None
            state["response"] = "Alright, let me know how else I can help you."

        return state

    # ── Step: collect issue description ───────────────────────────────────
    if step == "collect_issue":
        ticket["issue"] = state["message"]
        flow["step"] = "collect_phone"
        state["response"] = choice(TICKET_ASK_PHONE)
        return state

    # ── Step: collect phone and create ticket ─────────────────────────────
    if step == "collect_phone":
        phone = extract_phone_number(state["message"])

        if not phone:
            state["response"] = "Please provide a valid registered phone number."
            return state

        ticket["phone"] = phone

        client = AsyncClient()
        prompt = f"""Convert issue into JSON:
{{
  "category": "card_issue | auth_issue | transaction_issue | account_issue | general",
  "short_description": "max 15 words"
}}

Issue:
{ticket['issue']}"""

        res = await client.generate(model="gemma4:e2b", prompt=prompt)

        try:
            data = json.loads(res.response)
        except Exception:
            match = re.search(r"\{.*\}", res.response, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        db = SessionLocal()
        try:
            result = create_ticket(
                db,
                phone,
                category=data.get("category", "general"),
                short_description=data.get("short_description", ticket["issue"][:80])
            )
            tool_status = "failed" if "error" in result else "success"
            short_desc = (
                f"Agent created support ticket for {phone}: "
                f"{data.get('short_description', '')}"
                if tool_status == "success"
                else f"Agent failed to create ticket for {phone}"
            )
        except Exception as e:
            result = {"error": str(e)}
            tool_status = "error"
            short_desc = f"Agent encountered an error creating ticket for {phone}"
        finally:
            db.close()

        await log_tool_call(
            vendor_id=1,
            agent_name=state["agent_name"],
            agent_type="bank",
            tool_name="create_ticket",
            tool_status=tool_status,
            short_description=short_desc,
            user_identifier=phone,
            raw_tool_input={
                "phone": phone,
                "category": data.get("category"),
                "short_description": data.get("short_description")
            },
            raw_tool_output=result,
        )

        ticket_id = result.get("ticket_id", "UNKNOWN")
        if tool_status == "success":
            state["response"] = choice(TICKET_FINAL) + f" Ticket ID: {ticket_id}"
        else:
            state["response"] = (
                "I'm sorry, I couldn't create a support ticket at this time. "
                "Please contact ABB support directly at 937."
            )
        flow["active"] = None
        flow["step"] = None
        return state

    # Fallback
    state["response"] = (
        "I understand you're facing an issue. "
        "Would you like me to raise a support ticket?"
    )
    flow["step"] = None
    return state


# -------------------- BALANCE NODE --------------------
async def balance_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "BALANCE"
        flow["step"] = "ask_phone"
        state["response"] = choice(BALANCE_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_account_balance(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc = (
            f"Agent checked account balance for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve balance for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = f"Agent encountered an error checking balance for {phone}"
    finally:
        db.close()

    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="bank",
        tool_name="get_account_balance",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=phone,
        raw_tool_input={"phone": phone},
        raw_tool_output=result,
    )

    flow["active"] = None
    flow["step"] = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(BALANCE_SUCCESS).format(
        balance=result["balance"], currency=result["currency"]
    )
    state.setdefault("intent", "BALANCE")
    return state


# -------------------- USER INFO NODE --------------------
async def user_info_node(state: BotState):
    state = ensure_memory(state)
    flow = state["memory"]["flow"]

    phone = state["memory"]["profile"].get("phone")
    new_phone = extract_phone_number(state["message"])
    if new_phone:
        phone = new_phone
        state["memory"]["profile"]["phone"] = phone

    if not phone:
        flow["active"] = "USER_INFO"
        flow["step"] = "ask_phone"
        state["response"] = choice(USER_INFO_MISSING_PHONE)
        return state

    db = SessionLocal()
    try:
        result = get_user_info(db, phone)
        tool_status = "failed" if "error" in result else "success"
        short_desc = (
            f"Agent retrieved profile info for {phone}"
            if tool_status == "success"
            else f"Agent failed to retrieve user info for {phone}: {result.get('error')}"
        )
    except Exception as e:
        result = {"error": str(e)}
        tool_status = "error"
        short_desc = f"Agent encountered an error fetching user info for {phone}"
    finally:
        db.close()

    await log_tool_call(
        vendor_id=VENDOR_ID,
        agent_name=state["agent_name"],
        agent_type="bank",
        tool_name="get_user_info",
        tool_status=tool_status,
        short_description=short_desc,
        user_identifier=phone,
        raw_tool_input={"phone": phone},
        raw_tool_output=result,
    )

    flow["active"] = None
    flow["step"] = None

    if "error" in result:
        state["response"] = result["error"]
        return state

    state["response"] = choice(USER_INFO_SUCCESS).format(
        name=result["user_name"], email=result["email"]
    )
    state.setdefault("intent", "USER_INFO")
    return state


# -------------------- RAG NODE --------------------
async def rag_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)
    agent = state["agent"]

    # If the user fires a RAG question while a flow is active, lapse the flow
    # gracefully so they get a useful answer instead of a confused continuation.
    flow = state["memory"]["flow"]
    if flow.get("active"):
        logger.info(
            f"[RAG] Interrupting active flow '{flow['active']}' "
            "to answer informational question."
        )
        flow["active"] = None
        flow["step"] = None

    try:
        vectorstore = agent.get_vectorstore()
    except Exception as e:
        logger.error(f"[VECTORSTORE ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_FAILED"
        return state

    if vectorstore is None:
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_FAILED"
        return state

    system_prompt = agent.get_system_prompt()
    docs, scores = get_rag_context(state["message"], vectorstore)

    if not docs:
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_EMPTY"
        return state

    context = "\n\n".join(docs)

    if not context.strip():
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "RAG_EMPTY"
        return state

    history = state["memory"]["history"][-8:]
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}" for h in history
    )

    prompt = f"""
            You are a human customer support representative for ABB.

            Answer naturally and conversationally like a real call center agent.

            IMPORTANT:
            - Use ONLY provided context for factual accuracy. However, you MAY generalize within the same product family (e.g., loan types, banking products) when terminology differs but meaning is equivalent
            - Do NOT invent information
            - Do NOT sound like a website, brochure, advertisement, or policy document
            - Speak in short, natural sentences
            - Answer the user's exact question first
            - If the context contains eligibility criteria and the user asks about documents, separate them clearly
            - Preserve all factual requirements and steps from the context
            - Do NOT copy large chunks verbatim unless necessary
            - Avoid marketing language
            - Avoid long paragraphs
            - Sound helpful and human

            CONTEXT:
            {context}

            CONVERSATION HISTORY:
            {history_text}

            USER QUESTION:
            {state['message']}

            ASSISTANT:
            """

    client = AsyncClient()
    try:
        res = await client.generate(
            model="gemma4:e4b",
            prompt=prompt,
            system=system_prompt,
            options={
                "temperature": 0.2,
                "top_p": 0.9
            }
        )
        answer = res.response.strip()

    except Exception as e:
        logger.error(f"[LLM ERROR] {e}")
        state["response"] = "Please contact ABB support at 937."
        state["intent"] = "LLM_FAILED"
        return state

    state["memory"]["history"].append(
        {"role": "user", "content": state["message"]}
    )
    state["memory"]["history"].append(
        {"role": "assistant", "content": answer}
    )
    state["memory"]["flow"]["expandable"] = True
    state["memory"]["flow"]["last_expand_offer"] = True
    state["response"] = answer
    state.setdefault("intent", "RAG")

    return state


# -------------------- EXPAND NODE --------------------
async def expand_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    flow = state["memory"]["flow"]
    agent = state["agent"]
    system_prompt = agent.get_system_prompt()

    last_answer = None
    for msg in reversed(state["memory"]["history"]):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    last_answer = last_answer or ""

    client = AsyncClient()

    res = await client.generate(
        model="gemma4:e4b",
        system=system_prompt,
        prompt=f"""
                You are a human ABB customer support agent.

                The user asked a follow-up question.

                Expand naturally based ONLY on the previous answer.

                IMPORTANT:
                - Keep the same factual information
                - Do NOT invent new information
                - Speak conversationally
                - Keep sentences short and natural
                - Do NOT sound promotional
                - Do NOT repeat unnecessary details
                - If information is unavailable, say:
                "Please contact ABB support at 937."

                PREVIOUS ANSWER:
                {last_answer}

                USER:
                {state['message']}

                ASSISTANT:
                """
    )
    expanded = res.response.strip()

    flow["expandable"] = False
    flow["last_expand_offer"] = False

    state["memory"]["history"].append({"role": "user", "content": state["message"]})
    state["memory"]["history"].append({"role": "assistant", "content": expanded})

    state["response"] = expanded
    state.setdefault("intent", "EXPAND")
    return state


# -------------------- GREETING NODE --------------------
async def greeting_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    state["response"] = choice(GREETING_RESPONSES)
    state.setdefault("intent", "GREETING")
    return state


# -------------------- SMALL TALK NODE --------------------
async def small_talk_node(state: BotState):
    state = ensure_memory(state)
    state = safe_state(state)

    state["response"] = choice(THANK_YOU_RESPONSES)
    state.setdefault("intent", "SMALL_TALK")
    return state


# -------------------- GRAPH --------------------
def build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(BotState)

    graph.add_node("detect", detect_intent)
    graph.add_node("control", control_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("balance", balance_node)
    graph.add_node("user_info", user_info_node)
    graph.add_node("ticket", ticket_node)
    graph.add_node("rag", rag_node)
    graph.add_node("expand", expand_node)

    graph.add_edge(START, "detect")
    graph.set_entry_point("detect")

    graph.add_conditional_edges(
        "detect",
        router,
        {
            "GREETING":   "greeting",
            "BALANCE":    "balance",
            "USER_INFO":  "user_info",
            "TICKET":     "ticket",
            "SMALL_TALK": "small_talk",
            "CONTROL":    "control",   # CANCEL_FLOW also routes here
            "RAG":        "rag",
            "EXPAND":     "expand",
        }
    )

    for node in [
        "greeting",
        "balance",
        "user_info",
        "ticket",
        "rag",
        "expand",
        "small_talk",
        "control",
    ]:
        graph.add_edge(node, END)

    return graph.compile()
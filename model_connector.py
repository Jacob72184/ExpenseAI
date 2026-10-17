"""
Step 2: connect the expense database (expense_db.py) to an AI model on Groq.

Setup:
    pip install openai
    export GROQ_API_KEY="your-key"        (Windows: set GROQ_API_KEY=your-key)
    # put expense_db.py in the same folder
Run a terminal chat to test:
    python expense_ai.py

How a request flows in the real app:
    OAuth login succeeds -> db.get_or_create_user(provider, sub, email, name)
    -> you get users.id -> chat(users.id, history, message)
The model never sees or chooses the user id.
"""
import json
import logging
import os
import time
from datetime import date

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

import expense_db as db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"   # check Groq's docs for current model names
MAX_STEPS = 5                   # safety limit for the tool loop
MAX_COMPLETION_TOKENS = 2000    # cap on the model's output; reasoning models spend
                                # part of this on thinking, so don't set it too low
MAX_HISTORY_MESSAGES = 10       # how many past messages we resend
MAX_TOOL_ROWS = 20              # cap on rows a tool may return (saves tokens)
RETRIES = 2                     # extra attempts for temporary API failures


class ModelUnavailable(Exception):
    """Raised when the AI service can't be used. The message is safe to show users."""


# ---------------------------------------------------------------
# 1. Tool descriptions: the only thing the model knows about your functions.
#    Note there is NO user_id here. The model never chooses who the user is.
# ---------------------------------------------------------------
TOOLS = [
    {"type": "function",
     "function":{"name":"delete_expense",
                 "description":"Delete the expense with the specified date",
                 "parameters":{"type":"object",
                               "properties":{"amount":{"type":"number","description":"positive amount spent."},
                                             "category": {"type": "string", "description": "e.g. food, transport, rent."},
                                             "spent_on": {"type": "string","description": "Date as YYYY-MM-DD. Omit for today."},
                                              "user_id": {"type": "integer", "description":"integer id of the user to uniquely identify the user"}}
                                              }
                            }
    },
    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": "Record a new expense for the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Positive amount spent."},
                    "category": {"type": "string", "description": "e.g. food, transport, rent."},
                    "description": {"type": "string", "description": "Short note, optional."},
                    "spent_on": {"type": "string",
                                 "description": "Date as YYYY-MM-DD. Omit for today."},
                },
                "required": ["amount", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_expenses",
            "description": "List the current user's expenses, newest first. "
                           "Use filters to narrow the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "limit": {"type": "integer",
                              "description": f"Max rows, up to {MAX_TOOL_ROWS}."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_by_category",
            "description": "Total spending per category for the current user. "
                           "Use this for any question about totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                },
            },
        },
    },
]


# ---------------------------------------------------------------
# 2. The dispatcher: turns the model's request into a real function call.
#    user_id is passed in by YOUR code (from the login), never by the model.
# ---------------------------------------------------------------
def run_tool(name, raw_arguments, user_id):
    # The model sends arguments as a JSON *string*, which may be malformed.
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "Tool arguments were not valid JSON."}
    if not isinstance(args, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object."}

    try:
        if name == "delete_expense":
            return db.delete_expense(
                user_id,
                args.get("amount"),
                args.get("category"),
                args.get("spent_on")
            )
        if name == "add_expense":
            return db.add_expense(
                user_id,
                args.get("amount"),
                args.get("category"),
                args.get("description", ""),
                args.get("spent_on"),
            )
        if name == "get_expenses":
            limit = args.get("limit", MAX_TOOL_ROWS)
            if isinstance(limit, int) and not isinstance(limit, bool):
                limit = min(limit, MAX_TOOL_ROWS)
            return db.get_expenses(
                user_id,
                category=args.get("category"),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
                limit=limit,
            )
        if name == "summarize_by_category":
            return db.summarize_by_category(
                user_id,
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
            )
    except Exception:  # last line of defense: a bug here must not crash the app
        logger.exception("Unexpected error while running tool %s", name)
        return {"ok": False, "error": "Unexpected error while running the tool."}

    return {"ok": False, "error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------
# 3. Talking to Groq (kept in one place so switching providers is easy)
# ---------------------------------------------------------------
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ModelUnavailable("The AI service isn't configured (missing GROQ_API_KEY).")
    # max_retries=0: we do our own retries below so we control the waiting.
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=0)


def call_model(client, messages):
    """One request to the model, with retries for temporary failures."""
    last_reason = "The AI service is unavailable right now. Please try again."
    for attempt in range(RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            if not response.choices:
                raise ModelUnavailable("The AI service returned an empty response.")
            return response.choices[0].message

        except RateLimitError:
            last_reason = "The AI service is busy (rate limit reached). Try again in a minute."
        except (APITimeoutError, APIConnectionError):
            last_reason = "Couldn't reach the AI service. Check your connection and try again."
        except APIStatusError as exc:
            logger.error("Groq API error: status=%s", exc.status_code)
            if exc.status_code in (401, 403):
                raise ModelUnavailable("The AI service rejected the API key.")
            if exc.status_code < 500:  # other 4xx errors: retrying won't help
                raise ModelUnavailable("The AI service couldn't process that request.")
            last_reason = "The AI service had an internal error. Please try again."

        logger.warning("Model call failed (attempt %d/%d)", attempt + 1, RETRIES + 1)
        if attempt < RETRIES:
            time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s

    raise ModelUnavailable(last_reason)


# ---------------------------------------------------------------
# 4. The chat function: the loop from the bookstore example, for expenses
# ---------------------------------------------------------------
def build_system_prompt():
    return (
        "You are an expense-tracking assistant inside an app. "
        f"Today's date is {date.today().isoformat()}. "
        "Use the tools to record or look up expenses; never guess amounts or totals. "
        "Convert relative dates ('yesterday', 'last month') into YYYY-MM-DD before calling tools. "
        "For totals, use summarize_by_category instead of adding numbers yourself. "
        "If a tool returns an error, explain it simply and ask the user to correct the input. "
        "If the request is unclear (for example no amount), ask a question instead of guessing. "
        "Do not assume a currency. Only discuss expenses. Be brief."
    )


def _assistant_message_to_dict(message):
    """Convert the model's reply object into a plain dict we can send back."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name,
                             "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    }


def chat(user_id, history, user_text, client=None):
    """
    user_id:   users.id (positive integer) from your OAuth login lookup,
               never from the model or the user's text
    history:   list of {"role": "user"/"assistant", "content": "..."} from earlier turns
    user_text: the new message
    Returns {"ok": True, "text": reply} or {"ok": False, "text": error message}.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return {"ok": False, "text": "You need to be signed in to use the assistant."}
    user_text = (user_text or "").strip()
    if not user_text:
        return {"ok": False, "text": "Please type a message."}
    if len(user_text) > 2000:
        return {"ok": False, "text": "That message is too long. Please shorten it."}

    # Make sure this id belongs to a real account before spending an API call.
    account = db.get_user(user_id)
    if not account["ok"]:
        return {"ok": False, "text": "Something went wrong. Please try again."}
    if account["data"] is None:
        return {"ok": False, "text": "We couldn't find your account. Please sign in again."}

    try:
        client = client or get_client()
        messages = (
            [{"role": "system", "content": build_system_prompt()}]
            + list(history)[-MAX_HISTORY_MESSAGES:]
            + [{"role": "user", "content": user_text}]
        )

        for _ in range(MAX_STEPS):
            message = call_model(client, messages)

            # No tool requested: this is the final answer.
            if not message.tool_calls:
                text = (message.content or "").strip()
                if not text:
                    return {"ok": False, "text": "I didn't get a response. Please try again."}
                return {"ok": True, "text": text}

            # Tool requested: keep the request in the history, run it, send results back.
            messages.append(_assistant_message_to_dict(message))
            for call in message.tool_calls:
                result = run_tool(call.function.name, call.function.arguments, user_id)
                logger.info("tool %s -> ok=%s", call.function.name, result.get("ok"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })

        return {"ok": False, "text": "That took too many steps. Please try a simpler request."}

    except ModelUnavailable as exc:
        return {"ok": False, "text": str(exc)}
    except Exception:
        logger.exception("Unexpected error in chat()")
        return {"ok": False, "text": "Something went wrong. Please try again."}


# ---------------------------------------------------------------
# 5. Terminal test. In the real app, Streamlit replaces this block.
# ---------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    setup = db.init_db()
    if not setup["ok"]:
        print(setup["error"])
        raise SystemExit(1)

    # TEMPORARY: simulates a finished OAuth login. In the real app these values
    # come from the verified login claims (st.user in Streamlit), never from typing.
    login = db.get_or_create_user("google", "demo-sub-123", "demo@example.com", "Demo User")
    if not login["ok"]:
        print(login["error"])
        raise SystemExit(1)
    user_id = login["data"]["id"]
    print(f"Signed in as {login['data']['name']} (internal id {user_id})")
    history = []
    print("Expense assistant. Type 'quit' to exit.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        result = chat(user_id, history, text)
        print("Assistant:", result["text"])
        if result["ok"]:  # only remember successful turns
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": result["text"]})
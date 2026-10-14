"""
Frontend for the expense assistant (Streamlit).

Files in the project folder:
    app.py                   <- this file: the UI and the login handling
    model_connector.py       <- chat(): talks to Groq and runs the tools
    expense_db.py            <- users + expenses in PostgreSQL
    .streamlit/secrets.toml  <- OAuth settings (NOT Python; never commit it)

Setup:
    pip install -r requirements.txt
Run:
    streamlit run app.py
"""
import logging
import os
import time

import streamlit as st

import expense_db as db
import model_connector as connector  # change this name if your file is named differently
import os
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"].strip()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROVIDER = "google"                       # label stored in users.provider
MAX_SESSION_AGE_SECONDS = 24 * 60 * 60    # force a fresh login after 24 hours
MAX_STORED_MESSAGES = 50                  # cap on chat history kept in memory

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Expense Assistant", page_icon="💸")

# expense_db.py reads its connection string from the DATABASE_URL environment
# variable, while Streamlit keeps secrets in st.secrets. This bridges the two,
# so expense_db.py needs no Streamlit code and stays easy to test on its own.
if not os.environ.get("DATABASE_URL"):
    try:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        logger.error("DATABASE_URL is not set (environment variable or secrets.toml).")


# ---------------------------------------------------------------
# 1. AUTHENTICATION: who is this person?
#    Streamlit does the OAuth/OIDC dance (redirects, code exchange, token
#    checks). We only read the result from st.user.
# ---------------------------------------------------------------
# If [auth] is missing from secrets.toml, st.user has no is_logged_in attribute.
is_logged_in = getattr(st.user, "is_logged_in", None)
if is_logged_in is None:
    st.error("Login is not configured. Add an [auth] section to .streamlit/secrets.toml.")
    st.stop()

if not is_logged_in:
    st.title("💸 Expense Assistant")
    st.write("Please sign in to see and manage your expenses.")
    if st.button("Log in with Google"):
        st.login()
    st.stop()  # nothing below this line runs for anonymous visitors

# Streamlit does not check the token's expiry for us (the docs say so), so we
# apply our own rule: logins older than MAX_SESSION_AGE_SECONDS are rejected.
issued_at = st.user.get("iat")  # "issued at": when the provider created the token
if isinstance(issued_at, (int, float)) and time.time() - issued_at > MAX_SESSION_AGE_SECONDS:
    st.warning("Your session expired. Please log in again.")
    st.logout()
    st.stop()

# "sub" is the provider's permanent id for this person. It is what we trust.
sub = st.user.get("sub")
if not isinstance(sub, str) or not sub.strip():
    st.error("Your login did not include an account identifier. Please log in again.")
    if st.button("Log out"):
        st.logout()
    st.stop()

# Only keep the email if the provider says it has verified it. We never use it
# to identify anyone; it is just for display.
email = st.user.get("email") if st.user.get("email_verified") is True else None
name = st.user.get("name")


# ---------------------------------------------------------------
# 2. MAP THE LOGIN TO OUR OWN USER ROW
# ---------------------------------------------------------------
if not st.session_state.get("db_ready"):
    setup = db.init_db()
    if not setup["ok"]:
        st.error(setup["error"])
        st.stop()
    st.session_state["db_ready"] = True

identity = (PROVIDER, sub)
if st.session_state.get("identity") != identity:
    # First run for this person in this browser session (or a different
    # account logged in): find or create their row.
    result = db.get_or_create_user(PROVIDER, sub, email, name)
    if not result["ok"]:
        st.error(result["error"])
        st.stop()
    st.session_state["identity"] = identity
    st.session_state["user_id"] = result["data"]["id"]   # our integer users.id
    st.session_state["display_name"] = result["data"]["name"] or result["data"]["email"] or "there"
    st.session_state["history"] = []  # a different person must never inherit old chat

user_id = st.session_state["user_id"]


# ---------------------------------------------------------------
# 3. THE CHAT SCREEN
# ---------------------------------------------------------------
st.title("💸 Expense Assistant")
st.caption("Try: “I spent 200 on lunch yesterday” or “How much did I spend on food this month?”")

for message in st.session_state["history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tell me about an expense, or ask a question")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # user_id comes from OUR session, never from the text the person typed.
                outcome = connector.chat(user_id, st.session_state["history"], prompt)
        except Exception:
            logger.exception("Unexpected error while calling chat()")
            outcome = {"ok": False, "text": "Something went wrong. Please try again."}

        if outcome["ok"]:
            st.markdown(outcome["text"])
        else:
            st.error(outcome["text"])

    if outcome["ok"]:  # only remember turns that worked
        history = st.session_state["history"]
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": outcome["text"]})
        st.session_state["history"] = history[-MAX_STORED_MESSAGES:]


# ---------------------------------------------------------------
# 4. SIDEBAR (drawn last so it shows data from the message above)
# ---------------------------------------------------------------
with st.sidebar:
    st.subheader(f"Hello, {st.session_state['display_name']}")
    if st.button("Log out"):
        st.logout()
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()

    st.divider()
    st.subheader("Spending by category")
    totals = db.summarize_by_category(user_id)
    if not totals["ok"]:
        st.caption("Couldn't load totals right now.")
    elif not totals["data"]:
        st.caption("No expenses yet.")
    else:
        st.dataframe(totals["data"], hide_index=True)

    st.subheader("Recent expenses")
    recent = db.get_expenses(user_id, limit=10)
    if not recent["ok"]:
        st.caption("Couldn't load expenses right now.")
    elif not recent["data"]:
        st.caption("Nothing recorded yet.")
    else:
        st.dataframe(recent["data"], hide_index=True)
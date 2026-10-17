"""
Database layer, PostgreSQL version. A drop-in replacement for the SQLite file:
same function names, same {"ok": True, "data": ...} / {"ok": False, "error": ...} results.

Setup:
    pip install "psycopg[binary,pool]"
    export DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"
    (use the connection string from your hosted database; never put it in code)

Smoke test (creates and then deletes its own test rows):
    python expense_db.py
"""
import logging
import math
import os
import secrets
import threading
from datetime import date
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

MAX_AMOUNT = 1_000_000_000

logger = logging.getLogger(__name__)


class DatabaseNotConfigured(Exception):
    """DATABASE_URL is missing."""


# Everything that can go wrong when talking to the database. psycopg_pool's
# timeout error is a subclass of psycopg.Error, so it is covered too.
_DB_ERRORS = (psycopg.Error, DatabaseNotConfigured)


# ---------------------------------------------------------------
# Connection pool
# Opening a connection to a remote database is slow (network + TLS), so we keep
# a few open and lend them out. One pool per process, created on first use.
# ---------------------------------------------------------------
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            url = os.environ.get("DATABASE_URL", "").strip()
            if not url:
                raise DatabaseNotConfigured("DATABASE_URL is not set.")
            pool = ConnectionPool(
                conninfo=url,
                min_size=1,
                max_size=5,
                timeout=15,            # max seconds to wait for a free connection
                open=False,
                # test a connection before lending it out: serverless databases
                # close idle connections when they suspend
                check=ConnectionPool.check_connection,
                kwargs={
                    "row_factory": dict_row,   # rows come back as dicts
                    "prepare_threshold": None, # no server-side prepared statements:
                                               # they break behind PgBouncer-style poolers
                    "connect_timeout": 10,
                },
            )
            pool.open()
            _pool = pool
    return _pool


def _conn():
    """Usage: `with _conn() as conn:` commits on success, rolls back on error."""
    return _get_pool().connection()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _ok(data):
    return {"ok": True, "data": data}


def _fail(message):
    return {"ok": False, "error": message}


def _plain(row):
    """Postgres returns Decimal and date objects; convert to plain floats/strings."""
    out = {}
    for key, value in dict(row).items():
        if isinstance(value, Decimal):
            value = float(value)
        elif isinstance(value, date):   # datetime is a subclass of date
            value = value.isoformat()
        out[key] = value
    return out


def _check_user(user_id):
    # user_id is users.id: a positive integer.
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return "A valid user_id is required."
    return None


def _check_date(value, field):
    """Return an error message if value isn't a valid YYYY-MM-DD date."""
    if value is None:
        return None
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return f"{field} must be a valid date in YYYY-MM-DD format, got {value!r}."
    return None


def _check_range(start_date, end_date):
    for value, name in ((start_date, "start_date"), (end_date, "end_date")):
        error = _check_date(value, name)
        if error:
            return error
    if start_date and end_date and start_date > end_date:
        return "start_date must not be after end_date."
    return None


def _database_error(action, exc):
    # Log technical details for you; give the caller a short, safe message.
    # Never log DATABASE_URL: it contains the password.
    if isinstance(exc, DatabaseNotConfigured):
        logger.error("Database is not configured (DATABASE_URL missing).")
        return _fail("The database is not configured.")
    logger.exception("Database error while %s", action)
    return _fail(f"Database error while {action}. Please try again.")


# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
_schema_ready = False
_schema_lock = threading.Lock()

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        name         TEXT,
        email        TEXT,
        provider     TEXT NOT NULL,
        provider_sub TEXT NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (provider, provider_sub)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS expenses (
        id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id     BIGINT NOT NULL REFERENCES users(id),
        amount      NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
        category    TEXT NOT NULL,
        description TEXT,
        spent_on    DATE NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS expenses_user_date_idx ON expenses (user_id, spent_on)",
]


def init_db():
    """Create the tables if they don't exist. Runs once per process."""
    global _schema_ready
    if _schema_ready:
        return _ok("Database ready.")
    try:
        with _schema_lock:
            if not _schema_ready:
                with _conn() as conn:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                _schema_ready = True
        return _ok("Database ready.")
    except _DB_ERRORS as exc:
        return _database_error("setting up the database", exc)


# ---------------------------------------------------------------
# Users
# ---------------------------------------------------------------
_USER_COLUMNS = "id, name, email, provider"


def _clean_user_fields(provider, provider_sub, email, name):
    """Validate and normalize login details. Returns (cleaned_tuple, error)."""
    if not isinstance(provider, str) or not provider.strip():
        return None, "provider is required (for example 'google')."
    if not isinstance(provider_sub, str) or not provider_sub.strip():
        return None, "provider_sub must be non-empty text."
    if len(provider_sub) > 255:
        return None, "provider_sub is too long."

    if email is not None:
        if not isinstance(email, str):
            return None, "email must be text."
        email = email.strip().lower() or None
        if email and ("@" not in email or len(email) > 254):
            return None, "email does not look valid."

    if name is not None:
        if not isinstance(name, str):
            return None, "name must be text."
        name = name.strip()[:200] or None

    return (provider.strip().lower(), provider_sub, email, name), None


def add_user(provider, provider_sub, email=None, name=None):
    """Create a user. Fails if this provider account already exists.
    For logins, use get_or_create_user() instead."""
    cleaned, error = _clean_user_fields(provider, provider_sub, email, name)
    if error:
        return _fail(error)
    provider, provider_sub, email, name = cleaned
    try:
        with _conn() as conn:
            row = conn.execute(
                "INSERT INTO users (name, email, provider, provider_sub) "
                f"VALUES (%s, %s, %s, %s) RETURNING {_USER_COLUMNS}",
                (name, email, provider, provider_sub),
            ).fetchone()
        return _ok(_plain(row))
    except psycopg.errors.UniqueViolation:
        return _fail("A user with this login already exists.")
    except _DB_ERRORS as exc:
        return _database_error("creating the user", exc)


def get_or_create_user(provider, provider_sub, email=None, name=None):
    """
    Call this right after a successful OAuth login.
    Finds the user by (provider, sub); creates them on first login.
    Returns {"ok": True, "data": {"id", "name", "email", "provider", "created"}}
    where "created" is True on the first-ever login.
    """
    cleaned, error = _clean_user_fields(provider, provider_sub, email, name)
    if error:
        return _fail(error)
    provider, provider_sub, email, name = cleaned

    find_sql = f"SELECT {_USER_COLUMNS} FROM users WHERE provider = %s AND provider_sub = %s"
    try:
        with _conn() as conn:
            created = False
            row = conn.execute(find_sql, (provider, provider_sub)).fetchone()

            if row is None:
                # ON CONFLICT DO NOTHING: if another request inserts the same user
                # at the same moment, we get no row back instead of an error.
                # (A failed statement would abort the whole Postgres transaction.)
                row = conn.execute(
                    "INSERT INTO users (name, email, provider, provider_sub) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (provider, provider_sub) DO NOTHING "
                    f"RETURNING {_USER_COLUMNS}",
                    (name, email, provider, provider_sub),
                ).fetchone()
                created = row is not None
                if row is None:  # lost the race: the other request's row now exists
                    row = conn.execute(find_sql, (provider, provider_sub)).fetchone()
                    if row is None:
                        return _fail("Could not sign you in. Please try again.")

            if not created and ((email and email != row["email"])
                                or (name and name != row["name"])):
                # Returning user whose name/email changed at the provider.
                # COALESCE keeps the old value when the new one is missing.
                row = conn.execute(
                    "UPDATE users SET email = COALESCE(%s, email), name = COALESCE(%s, name) "
                    f"WHERE id = %s RETURNING {_USER_COLUMNS}",
                    (email, name, row["id"]),
                ).fetchone()

            data = _plain(row)
            data["created"] = created
        return _ok(data)
    except _DB_ERRORS as exc:
        return _database_error("signing the user in", exc)


def get_user(user_id):
    """Look up a user by internal id. data is None if no such user exists."""
    error = _check_user(user_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            row = conn.execute(
                f"SELECT {_USER_COLUMNS} FROM users WHERE id = %s", (user_id,)
            ).fetchone()
        return _ok(_plain(row) if row else None)
    except _DB_ERRORS as exc:
        return _database_error("looking up the user", exc)


# ---------------------------------------------------------------
# Expenses
# ---------------------------------------------------------------
def add_expense(user_id, amount, category, description="", spent_on=None):
    """Validate and save one expense. Returns the saved record."""
    error = _check_user(user_id)
    if error:
        return _fail(error)

    # bool is a subclass of int in Python, so reject it explicitly
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return _fail("Amount must be a number.")
    if not math.isfinite(amount) or amount <= 0:
        return _fail("Amount must be a positive number.")
    if amount > MAX_AMOUNT:
        return _fail("Amount is unrealistically large; please check it.")
    if round(amount, 2) <= 0:  # the column stores 2 decimal places
        return _fail("Amount must be at least 0.01.")

    if not isinstance(category, str) or not category.strip():
        return _fail("Category is required.")
    category = category.lower().strip()

    description = (description or "").strip()[:500]

    spent_on = spent_on or date.today().isoformat()
    error = _check_date(spent_on, "spent_on")
    if error:
        return _fail(error)

    try:
        with _conn() as conn:
            row = conn.execute(
                "INSERT INTO expenses (user_id, amount, category, description, spent_on) "
                "VALUES (%s, %s, %s, %s, %s) "
                "RETURNING id, amount, category, description, spent_on",
                (user_id, amount, category, description, date.fromisoformat(spent_on)),
            ).fetchone()
        return _ok(_plain(row))
    except psycopg.errors.ForeignKeyViolation:
        # user_id isn't in the users table
        logger.warning("Rejected expense for unknown user_id=%s", user_id)
        return _fail("Unknown user. Please sign in again.")
    except _DB_ERRORS as exc:
        return _database_error("saving the expense", exc)
def delete_expense(user_id, amount,category,spent_on=None):
    error = _check_user(user_id)
    if error:
        return _fail(error)
    
        # bool is a subclass of int in Python, so reject it explicitly
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return _fail("Amount must be a number.")
    if not math.isfinite(amount) or amount <= 0:
            return _fail("Amount must be a positive number.")
    if round(amount, 2) <= 0:  # the column stores 2 decimal places
            return _fail("Amount must be at least 0.01.")
    
    if not isinstance(category, str) or not category.strip():
            return _fail("Category is required.")
    category = category.lower().strip()
    spent_on = spent_on or date.today().isoformat()
    error = _check_date(spent_on, "spent_on")
    if error:
        return _fail(error)
    
    try:
        with _conn() as conn:
            row = conn.execute("DELETE FROM expenses WHERE user_id = %s AND amount=%s AND category=%s AND spent_on=%s ",
                    (user_id,amount,category, spent_on))
            return _ok("Succesfully deleted expense!")
    except psycopg.errors.ForeignKeyViolation:
                # user_id isn't in the users table
                logger.warning("Rejected expense for unknown user_id=%s", user_id)      
                return _fail("Unknown user. Please sign in again.")

        
#def rename_category():


def get_expenses(user_id, category=None, start_date=None, end_date=None, limit=100):
    """List a user's expenses, optionally filtered. Newest first, capped by limit."""
    error = _check_user(user_id) or _check_range(start_date, end_date)
    if error:
        return _fail(error)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        return _fail("limit must be a whole number between 1 and 500.")

    # user_id is deliberately left out of the columns we return.
    query = ("SELECT id, amount, category, description, spent_on "
             "FROM expenses WHERE user_id = %s")
    params = [user_id]
    if category:
        query += " AND category = %s"
        params.append(category.lower().strip())
    if start_date:
        query += " AND spent_on >= %s"
        params.append(date.fromisoformat(start_date))
    if end_date:
        query += " AND spent_on <= %s"
        params.append(date.fromisoformat(end_date))
    query += " ORDER BY spent_on DESC, id DESC LIMIT %s"
    params.append(limit)

    try:
        with _conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading expenses", exc)


def summarize_by_category(user_id, start_date=None, end_date=None):
    """Total spending per category, so the model doesn't have to do the math."""
    error = _check_user(user_id) or _check_range(start_date, end_date)
    if error:
        return _fail(error)

    query = "SELECT category, SUM(amount) AS total FROM expenses WHERE user_id = %s"
    params = [user_id]
    if start_date:
        query += " AND spent_on >= %s"
        params.append(date.fromisoformat(start_date))
    if end_date:
        query += " AND spent_on <= %s"
        params.append(date.fromisoformat(end_date))
    query += " GROUP BY category ORDER BY total DESC"

    try:
        with _conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("summarizing expenses", exc)


# ---------------------------------------------------------------
# Smoke test: python expense_db.py   (needs DATABASE_URL)
# Creates its own test users and deletes them at the end.
# ---------------------------------------------------------------
def _delete_test_users(user_ids):
    with _conn() as conn:
        for uid in user_ids:
            conn.execute("DELETE FROM expenses WHERE user_id = %s", (uid,))
            conn.execute("DELETE FROM users WHERE id = %s", (uid,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    print(init_db())
    tag = secrets.token_hex(4)
    created_ids = []
    try:
        print("\n--- users ---")
        first = get_or_create_user("google", f"smoke-{tag}", "Demo@Example.com", "Demo User")
        print("first login:   ", first)
        if not first["ok"]:
            raise SystemExit(1)
        uid = first["data"]["id"]
        created_ids.append(uid)
        again = get_or_create_user("google", f"smoke-{tag}", "demo@example.com", "Demo User")
        print("second login:  ", again, "| same id:", again["data"]["id"] == uid)
        print("add duplicate: ", add_user("google", f"smoke-{tag}"))
        other = get_or_create_user("github", f"smoke-{tag}", None, "Demo (GitHub)")
        print("other provider:", other)
        if other["ok"]:
            created_ids.append(other["data"]["id"])
        print("get_user:      ", get_user(uid))
        print("get_user(0):   ", get_user(0))

        print("\n--- expenses ---")
        print(add_expense(uid, 200, "Food", "lunch"))
        print(add_expense(uid, 19.99, "Transport", "taxi", "2026-10-01"))
        print(get_expenses(uid))
        print(summarize_by_category(uid))

        print("\n--- bad input should return errors, not crash ---")
        print(add_expense(uid, -5, "Food"))
        print(add_expense(uid, "ten", "Food"))
        print(add_expense(uid, 0.001, "Food"))
        print(add_expense(uid, 10, "  "))
        print(add_expense(uid, 10, "Food", spent_on="03/10/2026"))
        print(add_expense(0, 10, "Food"))
        print(add_expense(2**62, 10, "Food"))   # valid shape, but no such user
        print(get_expenses(uid, start_date="2026-12-01", end_date="2026-01-01"))
        print(get_or_create_user("", "x"))
        print(get_or_create_user("google", 12345))
        print(get_or_create_user("google", "s", email="not-an-email"))
    finally:
        try:
            _delete_test_users(created_ids)
            print("\n(test rows deleted)")
        except _DB_ERRORS:
            print("\nCould not delete the test rows; remove them by hand.")
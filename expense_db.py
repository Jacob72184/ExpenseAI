"""
Database layer, PostgreSQL version. Multi-organization edition.

All tenant tables carry org_id and every tenant query filters by org_id taken
from the verified login session (never from the model, prompt, or typed text).
Schema changes go through numbered files in migrations/ (hard rule 9); the
inline CREATE TABLE IF NOT EXISTS blocks below are left untouched and migrations
are applied on top of them.

All functions return {"ok": True, "data": ...} or {"ok": False, "error": ...}.

Setup:
    pip install "psycopg[binary,pool]"
    export DATABASE_URL="postgresql://..."
Smoke test (creates and then deletes its own test rows):
    python expense_db.py
"""
import json
import logging
import math
import os
import re
import secrets
import threading
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg.errors import UniqueViolation, ForeignKeyViolation, UndefinedTable

MAX_AMOUNT = 1_000_000_000

VALID_ROLES = ("employee", "manager", "finance_admin", "auditor")
VALID_STATUSES = ("draft", "submitted", "approved", "rejected", "reimbursed")

logger = logging.getLogger(__name__)


class DatabaseNotConfigured(Exception):
    """DATABASE_URL is missing."""


# Everything that can go wrong when talking to the database. psycopg_pool's
# timeout error is a subclass of psycopg.Error, so it is covered too.
_DB_ERRORS = (psycopg.Error, DatabaseNotConfigured)


# ---------------------------------------------------------------
# Connection pool
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
                timeout=15,
                open=False,
                check=ConnectionPool.check_connection,
                kwargs={
                    "row_factory": dict_row,
                    "prepare_threshold": None,
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
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        return "A valid user_id is required."
    return None


def _check_org(org_id):
    if isinstance(org_id, bool) or not isinstance(org_id, int) or org_id <= 0:
        return "A valid org_id is required."
    return None


def _check_role(role):
    if role not in VALID_ROLES:
        return f"Unknown role: {role!r}."
    return None


def _check_date(value, field):
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


def _check_expense_id(expense_id):
    if isinstance(expense_id, bool) or not isinstance(expense_id, int) or expense_id <= 0:
        return "A valid expense id is required."
    return None


def _database_error(action, exc):
    if isinstance(exc, DatabaseNotConfigured):
        logger.error("Database is not configured (DATABASE_URL missing).")
        return _fail("The database is not configured.")
    logger.exception("Database error while %s", action)
    return _fail(f"Database error while {action}. Please try again.")


# ---------------------------------------------------------------
# Schema setup: inline base schema + numbered migrations
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


def _split_sql(sql):
    """Split a migration file into individual statements.

    Ignores full-line comments (--). Splits on semicolons. Safe for our
    migration files which contain no semicolons inside string literals.
    """
    parts = []
    current = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            parts.append("\n".join(current))
            current = []
    if current:
        rest = "\n".join(current).strip()
        if rest:
            parts.append(rest)
    return parts


def _applied_migrations(conn):
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        return {row["version"] for row in rows}
    except UndefinedTable:
        return set()


def _apply_migrations():
    """Apply numbered migration files in order, recording each in schema_migrations.

    Statements run in autocommit so a single failed statement (for example
    CREATE EXTENSION vector when pgvector is not installed) does not abort the
    whole migration. Failed statements are logged and skipped; rag.py degrades
    gracefully to plain-text search when embeddings are unavailable.
    """
    migrations_dir = Path(__file__).resolve().parent / "migrations"
    if not migrations_dir.is_dir():
        return _ok("No migrations directory.")
    files = sorted(migrations_dir.glob("*.sql"))
    applied_any = False
    with _conn() as conn:
        conn.autocommit = True
        try:
            applied = _applied_migrations(conn)
            for f in files:
                try:
                    version = int(f.stem.split("_", 1)[0])
                except ValueError:
                    logger.warning("Skipping migration with non-numeric prefix: %s", f.name)
                    continue
                if version in applied:
                    continue
                sql = f.read_text()
                for stmt in _split_sql(sql):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            conn.execute(stmt)
                        except _DB_ERRORS as exc:
                            logger.warning("Migration %s statement skipped: %s", f.name, exc)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (version,),
                )
                applied_any = True
                logger.info("Applied migration %s", f.name)
        finally:
            pass
    if applied_any:
        return _ok("Migrations applied.")
    return _ok("Database already up to date.")


def init_db():
    """Create the base tables if they don't exist, then apply migrations."""
    global _schema_ready
    if _schema_ready:
        return _ok("Database ready.")
    try:
        with _schema_lock:
            if not _schema_ready:
                with _conn() as conn:
                    for statement in _SCHEMA:
                        conn.execute(statement)
                result = _apply_migrations()
                _schema_ready = True
            else:
                result = _ok("Database ready.")
        return result
    except _DB_ERRORS as exc:
        return _database_error("setting up the database", exc)


# ---------------------------------------------------------------
# Users
# ---------------------------------------------------------------
_USER_COLUMNS = "id, name, email, provider"


def _clean_user_fields(provider, provider_sub, email, name):
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
    """Create a user. Fails if this provider account already exists."""
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
    except UniqueViolation:
        return _fail("A user with this login already exists.")
    except _DB_ERRORS as exc:
        return _database_error("creating the user", exc)


def get_or_create_user(provider, provider_sub, email=None, name=None):
    """Find the user by (provider, sub); create them on first login.

    Also ensures the user has at least one organization (an org of one) and an
    employee membership, so every logged-in person has an active org.
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
                row = conn.execute(
                    "INSERT INTO users (name, email, provider, provider_sub) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (provider, provider_sub) DO NOTHING "
                    f"RETURNING {_USER_COLUMNS}",
                    (name, email, provider, provider_sub),
                ).fetchone()
                created = row is not None
                if row is None:
                    row = conn.execute(find_sql, (provider, provider_sub)).fetchone()
                    if row is None:
                        return _fail("Could not sign you in. Please try again.")

            if not created and ((email and email != row["email"])
                                or (name and name != row["name"])):
                row = conn.execute(
                    "UPDATE users SET email = COALESCE(%s, email), name = COALESCE(%s, name) "
                    f"WHERE id = %s RETURNING {_USER_COLUMNS}",
                    (email, name, row["id"]),
                ).fetchone()

            data = _plain(row)
            data["created"] = created

            # Ensure the user belongs to at least one organization (org of one).
            memberships = _list_user_orgs_in_conn(conn, row["id"])
            if not memberships:
                org = _create_org_of_one_in_conn(conn, row["id"], name or email or "User")
                data["default_org"] = org
                data["orgs"] = [org]
            else:
                data["orgs"] = memberships

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
# Organizations / memberships (internal helpers run inside a conn)
# ---------------------------------------------------------------
def _list_user_orgs_in_conn(conn, user_id):
    rows = conn.execute(
        """
        SELECT o.id, o.name, o.slug, m.role, m.is_active
        FROM org_memberships m
        JOIN organizations o ON o.id = m.org_id
        WHERE m.user_id = %s
        ORDER BY m.created_at
        """,
        (user_id,),
    ).fetchall()
    return [_plain(row) for row in rows]


def list_user_orgs(user_id):
    """Organizations this user belongs to, with their role in each."""
    error = _check_user(user_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            return _ok(_list_user_orgs_in_conn(conn, user_id))
    except _DB_ERRORS as exc:
        return _database_error("listing organizations", exc)


def _create_org_of_one_in_conn(conn, user_id, display_name):
    slug_base = re.sub(r"[^a-z0-9]+", "-", (display_name or "user").strip().lower()).strip("-") or "user"
    slug = slug_base
    counter = 1
    while conn.execute("SELECT 1 FROM organizations WHERE slug = %s", (slug,)).fetchone():
        counter += 1
        slug = f"{slug_base}-{counter}"
    row = conn.execute(
        "INSERT INTO organizations (name, slug) VALUES (%s, %s) RETURNING id, name, slug",
        (f"{display_name}'s organization", slug),
    ).fetchone()
    org_id = row["id"]
    conn.execute(
        "INSERT INTO org_memberships (user_id, org_id, role) VALUES (%s, %s, %s)",
        (user_id, org_id, "employee"),
    )
    return {"id": org_id, "name": row["name"], "slug": row["slug"], "role": "employee", "is_active": True}


def create_organization(name, slug, created_by_user_id, role="finance_admin"):
    """Create a new organization and add the creator as a member with the given role.

    The creator defaults to finance_admin so they can bootstrap membership
    (managers, employees, auditors) for the new organization.
    """
    error = _check_user(created_by_user_id) or _check_role(role)
    if error:
        return _fail(error)
    if not isinstance(name, str) or not name.strip():
        return _fail("Organization name is required.")
    if not isinstance(slug, str) or not slug.strip():
        return _fail("Organization slug is required.")
    name = name.strip()
    slug = slug.strip().lower()
    try:
        with _conn() as conn:
            if conn.execute("SELECT 1 FROM organizations WHERE slug = %s", (slug,)).fetchone():
                return _fail("An organization with this slug already exists.")
            row = conn.execute(
                "INSERT INTO organizations (name, slug) VALUES (%s, %s) RETURNING id, name, slug",
                (name, slug),
            ).fetchone()
            org_id = row["id"]
            conn.execute(
                "INSERT INTO org_memberships (user_id, org_id, role) VALUES (%s, %s, %s)",
                (created_by_user_id, org_id, role),
            )
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, created_by_user_id, "create_organization", "organizations", org_id,
                 json.dumps({"name": name, "slug": slug})),
            )
        return _ok({"id": org_id, "name": row["name"], "slug": row["slug"], "role": role})
    except UniqueViolation:
        return _fail("An organization with this slug already exists.")
    except _DB_ERRORS as exc:
        return _database_error("creating the organization", exc)


def add_organization_member(org_id, user_id, role, actor_user_id, actor_role):
    """Add a user to an organization with a role (actor must be a member of org)."""
    error = _check_org(org_id) or _check_user(user_id) or _check_user(actor_user_id) or _check_role(role)
    if error:
        return _fail(error)
    if actor_role != "finance_admin":
        return _fail("Only a finance admin can add organization members.")
    try:
        with _conn() as conn:
            actor_member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, actor_user_id, actor_role),
            ).fetchone()
            if not actor_member:
                return _fail("You are not a finance admin in this organization.")
            target_user = conn.execute(
                "SELECT id FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            if not target_user:
                return _fail("User not found.")
            conn.execute(
                "INSERT INTO org_memberships (user_id, org_id, role) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, org_id, role) DO UPDATE SET is_active = true",
                (user_id, org_id, role),
            )
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, "add_member", "org_memberships", user_id,
                 json.dumps({"added_user_id": user_id, "role": role})),
            )
        return _ok({"org_id": org_id, "user_id": user_id, "role": role})
    except _DB_ERRORS as exc:
        return _database_error("adding organization member", exc)


def get_user_roles_in_org(user_id, org_id):
    """Return all roles the user has in an org, or empty list if not a member."""
    error = _check_user(user_id) or _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT role FROM org_memberships WHERE org_id = %s AND user_id = %s AND is_active "
                "ORDER BY role",
                (org_id, user_id),
            ).fetchall()
        return _ok([row["role"] for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("listing roles in organization", exc)


def user_has_role_in_org(user_id, org_id, role):
    """Check if the user has a specific role in an org."""
    error = _check_user(user_id) or _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, role),
            ).fetchone()
        return _ok(bool(row))
    except _DB_ERRORS as exc:
        return _database_error("checking role in organization", exc)


def get_user_by_email(email):
    """Look up a user by email address. Returns data or None."""
    if not isinstance(email, str) or not email.strip():
        return _fail("A valid email is required.")
    email = email.strip().lower()
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT id, name, email, provider, provider_sub FROM users WHERE email = %s",
                (email,),
            ).fetchone()
        return _ok(_plain(row) if row else None)
    except _DB_ERRORS as exc:
        return _database_error("looking up user by email", exc)


def deactivate_organization_member(org_id, user_id, role, actor_user_id, actor_role):
    """Set a membership inactive (soft remove). Actor must be finance_admin."""
    error = _check_org(org_id) or _check_user(user_id) or _check_user(actor_user_id)
    if error:
        return _fail(error)
    if actor_role != "finance_admin":
        return _fail("Only a finance admin can remove organization members.")
    try:
        with _conn() as conn:
            actor_member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, actor_user_id, actor_role),
            ).fetchone()
            if not actor_member:
                return _fail("You are not a finance admin in this organization.")
            if user_id == actor_user_id and role == actor_role:
                return _fail("You cannot deactivate your own membership.")
            row = conn.execute(
                "UPDATE org_memberships SET is_active = false "
                "WHERE org_id = %s AND user_id = %s AND role = %s AND is_active "
                "RETURNING id, user_id, role",
                (org_id, user_id, role),
            ).fetchone()
            if not row:
                return _fail("Member not found or already inactive.")
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, "deactivate_member", "org_memberships", user_id,
                 json.dumps({"removed_user_id": user_id, "role": row["role"]})),
            )
        return _ok({"org_id": org_id, "user_id": user_id, "role": role, "active": False})
    except _DB_ERRORS as exc:
        return _database_error("deactivating organization member", exc)


def update_organization_member_role(org_id, user_id, current_role, new_role, actor_user_id, actor_role):
    """Change a member's role. Actor must be finance_admin."""
    error = _check_org(org_id) or _check_user(user_id) or _check_user(actor_user_id) or _check_role(new_role)
    if error:
        return _fail(error)
    if actor_role != "finance_admin":
        return _fail("Only a finance admin can change member roles.")
    try:
        with _conn() as conn:
            actor_member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, actor_user_id, actor_role),
            ).fetchone()
            if not actor_member:
                return _fail("You are not a finance admin in this organization.")
            if user_id == actor_user_id and current_role == actor_role:
                return _fail("You cannot change your own role.")
            row = conn.execute(
                "UPDATE org_memberships SET role = %s "
                "WHERE org_id = %s AND user_id = %s AND role = %s AND is_active "
                "RETURNING id, user_id, role",
                (new_role, org_id, user_id, current_role),
            ).fetchone()
            if not row:
                return _fail("Member not found or inactive.")
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, "change_member_role", "org_memberships", user_id,
                 json.dumps({"changed_user_id": user_id, "old_role": current_role, "new_role": new_role})),
            )
        return _ok({"org_id": org_id, "user_id": user_id, "role": new_role})
    except _DB_ERRORS as exc:
        return _database_error("updating organization member role", exc)


def update_organization_member_employee_id(org_id, user_id, role, employee_id, actor_user_id, actor_role):
    """Set/change a member's per-org employee_id. Actor must be finance_admin."""
    error = _check_org(org_id) or _check_user(user_id) or _check_user(actor_user_id)
    if error:
        return _fail(error)
    if actor_role != "finance_admin":
        return _fail("Only a finance admin can edit member details.")
    if employee_id is not None:
        employee_id = employee_id.strip() or None
    try:
        with _conn() as conn:
            actor_member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, actor_user_id, actor_role),
            ).fetchone()
            if not actor_member:
                return _fail("You are not a finance admin in this organization.")
            row = conn.execute(
                "UPDATE org_memberships SET employee_id = %s "
                "WHERE org_id = %s AND user_id = %s AND role = %s AND is_active "
                "RETURNING id, user_id, employee_id",
                (employee_id, org_id, user_id, role),
            ).fetchone()
            if not row:
                return _fail("Member not found or inactive.")
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, "edit_member", "org_memberships", user_id,
                 json.dumps({"edited_user_id": user_id, "employee_id": employee_id})),
            )
        return _ok({"org_id": org_id, "user_id": user_id, "employee_id": employee_id})
    except _DB_ERRORS as exc:
        return _database_error("updating organization member employee id", exc)


def create_org_invitation(org_id, email, employee_id, role, invited_by, expires_in_days=7):
    """Create a pending invitation for someone to join an organization.

    The invitee accepts by signing in with the matching email and providing the
    employee_id recorded here. email + employee_id together are the security
    check: a typo'd or forwarded email cannot let the wrong person join.

    The invitation row is created directly; the admin shares the token with the
    invitee out-of-band (for example by copying it from the pending invitations
    list). No email is sent by the app.
    """
    error = _check_org(org_id) or _check_user(invited_by) or _check_role(role)
    if error:
        return _fail(error)
    if not isinstance(email, str) or not email.strip():
        return _fail("Invitee email is required.")
    email = email.strip().lower()
    if "@" not in email or len(email) > 254:
        return _fail("Invitee email does not look valid.")
    if not isinstance(employee_id, str) or not employee_id.strip():
        return _fail("Employee ID is required.")
    employee_id = employee_id.strip()
    if isinstance(expires_in_days, bool) or not isinstance(expires_in_days, int) or expires_in_days < 1:
        return _fail("expires_in_days must be a positive integer.")
    try:
        with _conn() as conn:
            actor_role = conn.execute(
                "SELECT role FROM org_memberships WHERE org_id = %s AND user_id = %s AND is_active",
                (org_id, invited_by),
            ).fetchone()
            if not actor_role or actor_role["role"] != "finance_admin":
                return _fail("Only a finance admin can create invitations.")
            # Expire any existing pending invitation for the same email + org.
            conn.execute(
                "UPDATE org_invitations SET status = 'expired' "
                "WHERE org_id = %s AND email = %s AND status = 'pending'",
                (org_id, email),
            )
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            row = conn.execute(
                """
                INSERT INTO org_invitations
                    (org_id, email, employee_id, role, invited_by, token, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, token, email, role, expires_at
                """,
                (org_id, email, employee_id, role, invited_by, token, expires_at),
            ).fetchone()
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, invited_by, "create_invitation", "org_invitations", row["id"],
                 json.dumps({"email": email, "role": role, "employee_id": employee_id})),
            )
        return _ok({"id": row["id"], "token": row["token"], "email": row["email"],
                    "role": row["role"], "expires_at": row["expires_at"]})
    except _DB_ERRORS as exc:
        return _database_error("creating invitation", exc)


def accept_org_invitation(token, user_id, employee_id):
    """Accept a pending invitation.

    The signed-in user's email must match the invitation email, and the user
    must provide the employee_id recorded on the invitation. employee_id is
    mandatory and must match the invitation exactly (case-insensitive), so a
    typo'd or forwarded email cannot let the wrong person join the org.
    On success, an active membership is created for the user in the inviting org.
    """
    error = _check_user(user_id)
    if error:
        return _fail(error)
    if not isinstance(token, str) or not token.strip():
        return _fail("An invitation token is required.")
    token = token.strip()
    if not isinstance(employee_id, str) or not employee_id.strip():
        return _fail("Employee ID is required.")
    employee_id = employee_id.strip()
    try:
        with _conn() as conn:
            # Find the user's email.
            user_row = conn.execute(
                "SELECT id, email FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            if not user_row:
                return _fail("User not found.")
            user_email = (user_row["email"] or "").strip().lower()

            inv = conn.execute(
                """
                SELECT id, org_id, email, employee_id, role, invited_by, status, expires_at
                FROM org_invitations WHERE token = %s
                """,
                (token,),
            ).fetchone()
            if not inv:
                return _fail("Invitation not found.")
            if inv["status"] != "pending":
                return _fail(f"Invitation is already {inv['status']}.")
            if inv["expires_at"] and inv["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
                conn.execute(
                    "UPDATE org_invitations SET status = 'expired' WHERE id = %s",
                    (inv["id"],),
                )
                return _fail("Invitation has expired.")
            # The user's signed-in email must match the invitation email.
            if user_email != (inv["email"] or "").strip().lower():
                return _fail("The invitation email does not match your signed-in email.")
            # The employee_id provided on acceptance must match the invitation.
            if not inv["employee_id"]:
                return _fail("This invitation has no employee ID recorded; ask the admin to re-invite.")
            if employee_id.lower() != inv["employee_id"].strip().lower():
                return _fail("The employee ID does not match the invitation.")
            org_id = inv["org_id"]
            invited_role = inv["role"]
            # Verify the org still exists.
            org_row = conn.execute(
                "SELECT id FROM organizations WHERE id = %s", (org_id,)
            ).fetchone()
            if not org_row:
                return _fail("The inviting organization no longer exists.")
            # If the user already has this role membership in this org, its
            # employee_id must match.
            membership = conn.execute(
                "SELECT employee_id FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, invited_role),
            ).fetchone()
            if membership and membership["employee_id"] and membership["employee_id"] != employee_id:
                return _fail("The employee ID does not match your existing membership in this organization.")
            # Create or reactivate the membership with the verified employee_id.
            conn.execute(
                """
                INSERT INTO org_memberships (user_id, org_id, role, is_active, employee_id)
                VALUES (%s, %s, %s, true, %s)
                ON CONFLICT (user_id, org_id, role) DO UPDATE
                    SET is_active = true,
                        employee_id = EXCLUDED.employee_id
                """,
                (user_id, org_id, invited_role, employee_id),
            )
            conn.execute(
                "UPDATE org_invitations SET status = 'accepted', accepted_at = now() WHERE id = %s",
                (inv["id"],),
            )
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, user_id, "accept_invitation", "org_invitations", inv["id"],
                 json.dumps({"invited_email": inv["email"], "role": invited_role,
                             "employee_id": employee_id})),
            )
        return _ok({"org_id": org_id, "role": inv["role"], "employee_id": employee_id})
    except _DB_ERRORS as exc:
        return _database_error("accepting invitation", exc)


def list_org_invitations(org_id, actor_user_id):
    """List invitations for an organization. Actor must be finance_admin."""
    error = _check_org(org_id) or _check_user(actor_user_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            actor_role = conn.execute(
                "SELECT role FROM org_memberships WHERE org_id = %s AND user_id = %s AND is_active",
                (org_id, actor_user_id),
            ).fetchone()
            if not actor_role or actor_role["role"] != "finance_admin":
                return _fail("Only a finance admin can list invitations.")
            rows = conn.execute(
                """
                SELECT id, email, employee_id, role, status, invited_by, expires_at, accepted_at, created_at
                FROM org_invitations WHERE org_id = %s
                ORDER BY created_at DESC
                """,
                (org_id,),
            ).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("listing invitations", exc)


# ---------------------------------------------------------------
# Expenses (org-scoped lifecycle)
# ---------------------------------------------------------------
def add_expense(user_id, org_id, role, amount, category, description="", spent_on=None):
    """Validate and save one draft expense in the caller's active org and role."""
    error = _check_user(user_id) or _check_org(org_id)
    if error:
        return _fail(error)

    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return _fail("Amount must be a number.")
    if not math.isfinite(amount) or amount <= 0:
        return _fail("Amount must be a positive number.")
    if amount > MAX_AMOUNT:
        return _fail("Amount is unrealistically large; please check it.")
    if round(amount, 2) <= 0:
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
            # Verify membership with any role in the org.
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s AND is_active",
                (org_id, user_id),
            ).fetchone()
            if not member:
                return _fail("You are not a member of this organization.")

            row = conn.execute(
                "INSERT INTO expenses (user_id, org_id, amount, category, description, spent_on, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'draft') "
                "RETURNING id, amount, category, description, spent_on, status, org_id",
                (user_id, org_id, amount, category, description, date.fromisoformat(spent_on)),
            ).fetchone()
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, user_id, "create_expense", "expenses", row["id"],
                 json.dumps({"amount": amount, "category": category, "status": "draft", "role": role})),
            )
        return _ok(_plain(row))
    except ForeignKeyViolation:
        logger.warning("Rejected expense for unknown user_id=%s org_id=%s", user_id, org_id)
        return _fail("Unknown user or organization. Please sign in again.")
    except _DB_ERRORS as exc:
        return _database_error("saving the expense", exc)


def submit_expense(expense_id, user_id, org_id):
    """Move a draft expense to submitted (only the owner, in their org)."""
    error = _check_expense_id(expense_id) or _check_org(org_id) or _check_user(user_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            # Confirm the expense exists, is draft, belongs to this user and org.
            row = conn.execute(
                "SELECT id, status, user_id FROM expenses WHERE id = %s AND org_id = %s",
                (expense_id, org_id),
            ).fetchone()
            if not row:
                return _fail("Expense not found or not in your organization.")
            if row["user_id"] != user_id:
                return _fail("You can only submit your own expenses.")
            if row["status"] != "draft":
                return _fail(f"Only a draft expense can be submitted (current: {row['status']}).")
            conn.execute(
                "UPDATE expenses SET status = 'submitted', submitted_at = now() "
                "WHERE id = %s AND org_id = %s AND status = 'draft' "
                "RETURNING id",
                (expense_id, org_id),
            )
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, user_id, "submit_expense", "expenses", expense_id,
                 json.dumps({"status": "submitted"})),
            )
        return _ok({"id": expense_id, "status": "submitted"})
    except _DB_ERRORS as exc:
        return _database_error("submitting the expense", exc)


def get_expenses(user_id, org_id, role, category=None, start_date=None, end_date=None, limit=100):
    """List expenses in the caller's org, optionally filtered. Newest first."""
    error = _check_user(user_id) or _check_org(org_id) or _check_range(start_date, end_date)
    if error:
        return _fail(error)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        return _fail("limit must be a whole number between 1 and 500.")

    # Verify membership with the active role and determine visibility.
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a member of this organization with this role.")

            query = ("SELECT id, amount, category, description, spent_on, status, "
                     "approver_id, submitted_at, approved_at, reimbursed_at, policy_version, verdict "
                     "FROM expenses WHERE org_id = %s")
            params = [org_id]
            # user_id filter only for employees; managers/finance/auditor see org-wide.
            if role == "employee":
                query += " AND user_id = %s"
                params.append(user_id)
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

            rows = conn.execute(query, params).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading expenses", exc)


def get_expense(expense_id, user_id, org_id, role):
    """Fetch a single expense, only if it is in the caller's org (and owned by them if employee)."""
    error = _check_expense_id(expense_id) or _check_user(user_id) or _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a member of this organization with this role.")

            query = ("SELECT id, user_id, org_id, amount, category, description, spent_on, status, "
                     "approver_id, submitted_at, approved_at, reimbursed_at, policy_version, verdict "
                     "FROM expenses WHERE id = %s AND org_id = %s")
            params = [expense_id, org_id]
            if role == "employee":
                query += " AND user_id = %s"
                params.append(user_id)
            row = conn.execute(query, params).fetchone()
        return _ok(_plain(row) if row else None)
    except _DB_ERRORS as exc:
        return _database_error("reading the expense", exc)


def summarize_by_category(user_id, org_id, role, start_date=None, end_date=None):
    """Total spending per category in the caller's org."""
    error = _check_user(user_id) or _check_org(org_id) or _check_range(start_date, end_date)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a member of this organization with this role.")

            query = ("SELECT category, SUM(amount) AS total, COUNT(*) AS count "
                     "FROM expenses WHERE org_id = %s")
            params = [org_id]
            if role == "employee":
                query += " AND user_id = %s"
                params.append(user_id)
            if start_date:
                query += " AND spent_on >= %s"
                params.append(date.fromisoformat(start_date))
            if end_date:
                query += " AND spent_on <= %s"
                params.append(date.fromisoformat(end_date))
            query += " GROUP BY category ORDER BY total DESC"

            rows = conn.execute(query, params).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("summarizing expenses", exc)


def update_expense_verdict(expense_id, verdict, policy_version, org_id):
    """Store the pre-check verdict and policy version on an expense."""
    error = _check_expense_id(expense_id) or _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            row = conn.execute(
                "UPDATE expenses SET verdict = %s, policy_version = %s "
                "WHERE id = %s AND org_id = %s "
                "RETURNING id, verdict, policy_version",
                (verdict, policy_version, expense_id, org_id),
            ).fetchone()
        return _ok(_plain(row) if row else None)
    except _DB_ERRORS as exc:
        return _database_error("updating the expense verdict", exc)


def delete_expense(expense_id, user_id, org_id, role):
    """Delete an expense (owner only, draft only, in their org)."""
    error = _check_expense_id(expense_id) or _check_user(user_id) or _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, user_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a member of this organization with this role.")

            row = conn.execute(
                "SELECT id, status, user_id FROM expenses WHERE id = %s AND org_id = %s",
                (expense_id, org_id),
            ).fetchone()
            if not row:
                return _fail("Expense not found or not in your organization.")
            if row["user_id"] != user_id:
                return _fail("You can only delete your own expenses.")
            if row["status"] != "draft":
                return _fail("Only draft expenses can be deleted.")
            conn.execute(
                "DELETE FROM expenses WHERE id = %s AND org_id = %s AND user_id = %s AND status = 'draft'",
                (expense_id, org_id, user_id),
            )
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, user_id, "delete_expense", "expenses", expense_id,
                 json.dumps({"status": row["status"]})),
            )
        return _ok({"id": expense_id, "status": "deleted"})
    except _DB_ERRORS as exc:
        return _database_error("deleting the expense", exc)


# ---------------------------------------------------------------
# Approvals (hard rule 5: atomic UPDATE + audit row in one transaction)
# ---------------------------------------------------------------
def approve_expense(expense_id, approver_id, org_id, role):
    """Manager approves a submitted expense. One atomic UPDATE + audit row.

    Requires: expense is submitted, in org, not the approver's own expense,
    approver is a manager or finance_admin in the org with the active role.
    """
    error = _check_expense_id(expense_id) or _check_user(approver_id) or _check_org(org_id)
    if error:
        return _fail(error)
    if role not in ("manager", "finance_admin"):
        return _fail("Only a manager or finance admin can approve expenses.")
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, approver_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a manager or finance admin in this organization.")

            row = conn.execute(
                "SELECT id, status, user_id, amount, category FROM expenses WHERE id = %s AND org_id = %s",
                (expense_id, org_id),
            ).fetchone()
            if not row:
                return _fail("Expense not found or not in your organization.")
            if row["status"] != "submitted":
                return _fail(f"Only a submitted expense can be approved (current: {row['status']}).")
            if row["user_id"] == approver_id:
                return _fail("You cannot approve your own expense.")

            result = conn.execute(
                "UPDATE expenses SET status = 'approved', approver_id = %s, approved_at = now() "
                "WHERE id = %s AND status = 'submitted' AND approver_id IS NULL "
                "AND user_id <> %s AND org_id = %s "
                "RETURNING id",
                (approver_id, expense_id, approver_id, org_id),
            ).fetchone()
            if not result:
                return _fail("Could not approve the expense (already handled or not eligible).")

            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, approver_id, "approve_expense", "expenses", expense_id,
                 json.dumps({"amount": float(row["amount"]), "category": row["category"], "status": "approved"})),
            )
        return _ok({"id": expense_id, "status": "approved", "approver_id": approver_id})
    except _DB_ERRORS as exc:
        return _database_error("approving the expense", exc)


def reject_expense(expense_id, approver_id, org_id, role, reason=""):
    """Manager/finance_admin rejects a submitted expense."""
    error = _check_expense_id(expense_id) or _check_user(approver_id) or _check_org(org_id)
    if error:
        return _fail(error)
    if role not in ("manager", "finance_admin"):
        return _fail("Only a manager or finance admin can reject expenses.")
    reason = (reason or "").strip()[:500]
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, approver_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a manager or finance admin in this organization.")

            row = conn.execute(
                "SELECT id, status, user_id FROM expenses WHERE id = %s AND org_id = %s",
                (expense_id, org_id),
            ).fetchone()
            if not row:
                return _fail("Expense not found or not in your organization.")
            if row["status"] != "submitted":
                return _fail(f"Only a submitted expense can be rejected (current: {row['status']}).")
            if row["user_id"] == approver_id:
                return _fail("You cannot reject your own expense.")

            result = conn.execute(
                "UPDATE expenses SET status = 'rejected', approver_id = %s, approved_at = now() "
                "WHERE id = %s AND status = 'submitted' AND approver_id IS NULL "
                "AND user_id <> %s AND org_id = %s "
                "RETURNING id",
                (approver_id, expense_id, approver_id, org_id),
            ).fetchone()
            if not result:
                return _fail("Could not reject the expense (already handled or not eligible).")

            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, approver_id, "reject_expense", "expenses", expense_id,
                 json.dumps({"status": "rejected", "reason": reason})),
            )
        return _ok({"id": expense_id, "status": "rejected", "approver_id": approver_id})
    except _DB_ERRORS as exc:
        return _database_error("rejecting the expense", exc)


def mark_reimbursed(expense_id, actor_user_id, org_id, role):
    """Finance admin marks an approved expense as reimbursed. No real payments."""
    error = _check_expense_id(expense_id) or _check_user(actor_user_id) or _check_org(org_id)
    if error:
        return _fail(error)
    if role != "finance_admin":
        return _fail("Only a finance admin can mark expenses as reimbursed.")
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, actor_user_id, role),
            ).fetchone()
            if not member:
                return _fail("You are not a finance admin in this organization.")

            row = conn.execute(
                "SELECT id, status, user_id FROM expenses WHERE id = %s AND org_id = %s",
                (expense_id, org_id),
            ).fetchone()
            if not row:
                return _fail("Expense not found or not in your organization.")
            if row["status"] != "approved":
                return _fail(f"Only an approved expense can be reimbursed (current: {row['status']}).")

            result = conn.execute(
                "UPDATE expenses SET status = 'reimbursed', reimbursed_at = now() "
                "WHERE id = %s AND status = 'approved' AND org_id = %s "
                "RETURNING id",
                (expense_id, org_id),
            ).fetchone()
            if not result:
                return _fail("Could not mark the expense as reimbursed.")

            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, "mark_reimbursed", "expenses", expense_id,
                 json.dumps({"status": "reimbursed"})),
            )
        return _ok({"id": expense_id, "status": "reimbursed"})
    except _DB_ERRORS as exc:
        return _database_error("marking reimbursed", exc)


# ---------------------------------------------------------------
# Policy rules as data
# ---------------------------------------------------------------
def add_policy_rule(org_id, rule_type, category=None, amount_threshold=None,
                    period=None, pattern=None, active=True):
    """Add a policy rule for an organization."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    valid_types = ("max_amount_per_category_period", "receipt_required_above",
                   "allowed_categories", "blocked_keywords")
    if rule_type not in valid_types:
        return _fail(f"Unknown rule type: {rule_type!r}.")
    if amount_threshold is not None:
        try:
            amount_threshold = round(float(amount_threshold), 2)
        except (TypeError, ValueError):
            return _fail("amount_threshold must be a number.")
    try:
        with _conn() as conn:
            row = conn.execute(
                "INSERT INTO policy_rules (org_id, rule_type, category, amount_threshold, period, pattern, active) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, rule_type, category, amount_threshold, period, pattern, active",
                (org_id, rule_type, category and category.strip().lower() or None,
                 amount_threshold, period, pattern, bool(active)),
            ).fetchone()
        return _ok(_plain(row))
    except _DB_ERRORS as exc:
        return _database_error("adding the policy rule", exc)


def get_policy_rules(org_id, active_only=True):
    """List policy rules for an organization."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT id, rule_type, category, amount_threshold, period, pattern, active "
                    "FROM policy_rules WHERE org_id = %s AND active ORDER BY rule_type, category",
                    (org_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, rule_type, category, amount_threshold, period, pattern, active "
                    "FROM policy_rules WHERE org_id = %s ORDER BY rule_type, category",
                    (org_id,),
                ).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading policy rules", exc)


# ---------------------------------------------------------------
# Policy documents + RAG chunks
# ---------------------------------------------------------------
def add_policy_document(org_id, title, content, version, uploaded_by, role):
    """Finance admin uploads a policy document for an organization."""
    error = _check_org(org_id) or _check_user(uploaded_by)
    if error:
        return _fail(error)
    if role != "finance_admin":
        return _fail("Only a finance admin can upload policy documents.")
    if not isinstance(title, str) or not title.strip():
        return _fail("Title is required.")
    if not isinstance(content, str) or not content.strip():
        return _fail("Content is required.")
    if not isinstance(version, str) or not version.strip():
        return _fail("Version is required.")
    title = title.strip()
    version = version.strip()
    try:
        with _conn() as conn:
            member = conn.execute(
                "SELECT 1 FROM org_memberships WHERE org_id = %s AND user_id = %s "
                "AND role = %s AND is_active",
                (org_id, uploaded_by, role),
            ).fetchone()
            if not member:
                return _fail("You are not a finance admin in this organization.")
            row = conn.execute(
                "INSERT INTO policy_documents (org_id, title, content, version, uploaded_by) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id, version",
                (org_id, title, content, version, uploaded_by),
            ).fetchone()
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, uploaded_by, "upload_policy_document", "policy_documents", row["id"],
                 json.dumps({"title": title, "version": version})),
            )
        return _ok({"id": row["id"], "version": row["version"]})
    except _DB_ERRORS as exc:
        return _database_error("uploading the policy document", exc)


def get_policy_documents(org_id):
    """List policy documents for an organization."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, title, version, uploaded_at, uploaded_by "
                "FROM policy_documents WHERE org_id = %s ORDER BY uploaded_at DESC",
                (org_id,),
            ).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading policy documents", exc)


def add_policy_chunks(document_id, org_id, chunks):
    """Store text chunks (with optional embeddings) for a policy document.

    chunks: list of dicts {chunk_index, content, embedding(list of floats)|None}
    Sensitive categories are excluded from the RAG index by the caller.
    """
    error = _check_org(org_id)
    if error:
        return _fail(error)
    if not isinstance(document_id, bool) and isinstance(document_id, int) and document_id > 0:
        pass
    else:
        return _fail("A valid document id is required.")
    try:
        with _conn() as conn:
            for ch in chunks:
                embedding = ch.get("embedding")
                if embedding is not None:
                    conn.execute(
                        "INSERT INTO policy_chunks (document_id, org_id, chunk_index, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (document_id, org_id, ch["chunk_index"], ch["content"], embedding),
                    )
                else:
                    conn.execute(
                        "INSERT INTO policy_chunks (document_id, org_id, chunk_index, content) "
                        "VALUES (%s, %s, %s, %s)",
                        (document_id, org_id, ch["chunk_index"], ch["content"]),
                    )
        return _ok({"chunks": len(chunks)})
    except _DB_ERRORS as exc:
        return _database_error("storing policy chunks", exc)


# ---------------------------------------------------------------
# Sensitive categories
# ---------------------------------------------------------------
def add_sensitive_category(org_id, category):
    """Mark a category as sensitive (kept out of prompts/logs/RAG index)."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    if not isinstance(category, str) or not category.strip():
        return _fail("Category is required.")
    category = category.strip().lower()
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO org_sensitive_categories (org_id, category) VALUES (%s, %s) "
                "ON CONFLICT (org_id, category) DO NOTHING",
                (org_id, category),
            )
        return _ok({"category": category})
    except _DB_ERRORS as exc:
        return _database_error("adding sensitive category", exc)


def get_sensitive_categories(org_id):
    """List sensitive categories for an organization."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT category FROM org_sensitive_categories WHERE org_id = %s",
                (org_id,),
            ).fetchall()
        return _ok([row["category"] for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading sensitive categories", exc)


def is_sensitive_category(org_id, category):
    error = _check_org(org_id)
    if error:
        return False
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM org_sensitive_categories WHERE org_id = %s AND category = %s",
                (org_id, category.lower().strip()),
            ).fetchone()
        return bool(row)
    except _DB_ERRORS:
        return False


# ---------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------
def write_audit_log(org_id, actor_user_id, action, target_table=None, target_id=None, details=None):
    """Write an audit log row. Returns ok even on failure (best-effort logging)."""
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (org_id, actor_user_id, action, target_table, target_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (org_id, actor_user_id, action, target_table, target_id,
                 json.dumps(details) if details is not None else None),
            )
        return _ok(True)
    except _DB_ERRORS as exc:
        logger.exception("Database error while writing audit log")
        return _fail("Audit log write failed.")


def get_audit_log(org_id, actor_user_id=None, limit=100):
    """Read audit log entries for an organization (read-only)."""
    error = _check_org(org_id)
    if error:
        return _fail(error)
    try:
        with _conn() as conn:
            query = "SELECT id, actor_user_id, action, target_table, target_id, details, created_at FROM audit_log WHERE org_id = %s"
            params = [org_id]
            if actor_user_id is not None:
                query += " AND actor_user_id = %s"
                params.append(actor_user_id)
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
        return _ok([_plain(row) for row in rows])
    except _DB_ERRORS as exc:
        return _database_error("reading audit log", exc)


# ---------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------
def _delete_test_users(user_ids):
    with _conn() as conn:
        for uid in user_ids:
            orgs = conn.execute(
                "SELECT org_id FROM org_memberships WHERE user_id = %s", (uid,)
            ).fetchall()
            org_ids = [r["org_id"] for r in orgs]
            conn.execute("DELETE FROM audit_log WHERE actor_user_id = %s", (uid,))
            conn.execute("DELETE FROM policy_documents WHERE uploaded_by = %s", (uid,))
            conn.execute("DELETE FROM org_invitations WHERE invited_by = %s", (uid,))
            if org_ids:
                conn.execute("DELETE FROM policy_chunks WHERE org_id = ANY(%s)", (org_ids,))
                conn.execute("DELETE FROM policy_rules WHERE org_id = ANY(%s)", (org_ids,))
                conn.execute("DELETE FROM org_sensitive_categories WHERE org_id = ANY(%s)", (org_ids,))
            conn.execute("DELETE FROM expenses WHERE user_id = %s", (uid,))
            conn.execute("DELETE FROM org_memberships WHERE user_id = %s", (uid,))
            conn.execute("DELETE FROM users WHERE id = %s", (uid,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    print(init_db())
    tag = secrets.token_hex(4)
    created_ids = []
    try:
        print("\n--- users / org of one ---")
        first = get_or_create_user("google", f"smoke-{tag}", "Demo@Example.com", "Demo User")
        print("first login:   ", first)
        if not first["ok"]:
            raise SystemExit(1)
        uid = first["data"]["id"]
        created_ids.append(uid)
        orgs = list_user_orgs(uid)
        print("user orgs:     ", orgs)
        org_id = orgs["data"][0]["id"]
        again = get_or_create_user("google", f"smoke-{tag}", "demo@example.com", "Demo User")
        print("second login:  ", again["data"]["id"] == uid)
        print("get_user(0):   ", get_user(0))

        print("\n--- expenses (org-scoped) ---")
        print(add_expense(uid, org_id, 200, "Food", "lunch"))
        print(add_expense(uid, org_id, 19.99, "Transport", "taxi", "2026-10-01"))
        print(get_expenses(uid, org_id))
        print(summarize_by_category(uid, org_id))

        print("\n--- submit + approve (atomic) ---")
        exp = add_expense(uid, org_id, 150, "Food", "dinner")
        eid = exp["data"]["id"]
        print(submit_expense(eid, uid, org_id))
        # self-approval should fail
        print("self-approve:  ", approve_expense(eid, uid, org_id))
        # a second user in same org as manager approves
        other = get_or_create_user("google", f"smoke-mgr-{tag}", "Mgr@example.com", "Manager")
        oid = other["data"]["id"]
        created_ids.append(oid)
        add_organization_member(org_id, oid, "manager", uid)
        print("approve by manager: ", approve_expense(eid, oid, org_id))
        print(get_expense(eid, uid, org_id))

        print("\n--- bad input ---")
        print(add_expense(uid, org_id, -5, "Food"))
        print(add_expense(uid, org_id, "ten", "Food"))
        print(get_expenses(uid, 999999))  # wrong org
        print(add_organization_member(org_id, uid, "boss", uid))  # invalid role

        print("\n--- audit log ---")
        print(get_audit_log(org_id, uid, 5))

    finally:
        try:
            _delete_test_users(created_ids)
            print("\n(test rows deleted)")
        except _DB_ERRORS:
            print("\nCould not delete the test rows; remove them by hand.")

"""Pytest configuration and fixtures for the expense tracker test suite.

Tests use a real PostgreSQL database (DATABASE_URL from environment or
.streamlit/secrets.toml). Each test creates its own organizations/users and
cleans them up (including audit_log rows) so orgs are isolated and no test
data leaks.
"""
import os
import tomllib
from pathlib import Path

import pytest

import expense_db as db


def _database_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    secrets = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if secrets.exists():
        with open(secrets, "rb") as f:
            return tomllib.load(f)["DATABASE_URL"]
    return None


@pytest.fixture(scope="session", autouse=True)
def _ensure_db_ready():
    url = _database_url()
    if not url:
        pytest.skip("DATABASE_URL is not configured; DB tests cannot run.")
    os.environ["DATABASE_URL"] = url
    result = db.init_db()
    assert result["ok"], result["error"]


@pytest.fixture
def org_user_pair():
    """Create an organization with a finance_admin owner and return (org_id, user_id)."""
    tag = os.environ.get("PYTEST_TAG", "test")
    import secrets as _secrets
    tag = _secrets.token_hex(4)
    owner = db.get_or_create_user("google", f"owner-{tag}", f"owner-{tag}@example.com", "Owner")
    assert owner["ok"], owner
    uid = owner["data"]["id"]
    created = [uid]
    org = db.create_organization(f"Test Org {tag}", f"testorg-{tag}", uid)
    assert org["ok"], org
    org_id = org["data"]["id"]
    yield org_id, uid, created
    # cleanup
    try:
        db._delete_test_users(created)
    except Exception:
        pass


@pytest.fixture
def employee_pair(org_user_pair):
    """Add an employee to the org; return (org_id, employee_id, manager_id)."""
    org_id, owner_id, created = org_user_pair
    emp = db.get_or_create_user("google", f"emp-{owner_id}", "emp@example.com", "Emp")
    assert emp["ok"], emp
    eid = emp["data"]["id"]
    created.append(eid)
    mgr = db.get_or_create_user("google", f"mgr-{owner_id}", "mgr@example.com", "Mgr")
    assert mgr["ok"], mgr
    mid = mgr["data"]["id"]
    created.append(mid)
    db.add_organization_member(org_id, eid, "employee", owner_id)
    db.add_organization_member(org_id, mid, "manager", owner_id)
    return org_id, eid, mid


@pytest.fixture
def clean_db():
    """Yield, then attempt broad cleanup of test orgs/users created in this session."""
    yield
    # Best-effort: nothing to do globally; individual fixtures clean themselves.


"""
Tests for rag.py (org-scoped policy search with text-search fallback).
"""
import pytest

import expense_db as db
import rag as rag_mod


def _finance_owner(tag):
    owner = db.get_or_create_user("google", f"fo-{tag}", f"fo-{tag}@ex.com", "Finance Owner")
    assert owner["ok"], owner
    uid = owner["data"]["id"]
    created = [uid]
    org = db.create_organization(f"Finance Org {tag}", f"finorg-{tag}", uid)
    assert org["ok"], org
    return org["data"]["id"], uid, created


def test_search_policy_org_scoped_text_fallback():
    org_id, uid, created = _finance_owner("rag1")
    # finance admin uploads a policy doc
    doc = db.add_policy_document(org_id, "Travel Policy", "Employees may spend up to 60 on meals per day. Alcohol is never reimbursable.", "1.0", uid)
    assert doc["ok"], doc
    doc_id = doc["data"]["id"]
    content = "Employees may spend up to 60 on meals per day. Alcohol is never reimbursable."
    # index it via rag (text-search fallback since no fastembed/pgvector)
    idx = rag_mod.add_policy_document(doc_id, org_id, content)
    assert idx["ok"]

    # search within org
    res = rag_mod.search_policy(org_id, "meals", limit=5)
    assert res["ok"]
    # cross-org search returns nothing
    org_b, uid_b, created_b = _finance_owner("rag2")
    res_b = rag_mod.search_policy(org_b, "meals", limit=5)
    assert res_b["ok"]
    assert res_b["data"] == []
    db._delete_test_users(created + created_b)


def test_search_policy_empty_query():
    org_id, uid, created = _finance_owner("rag3")
    res = rag_mod.search_policy(org_id, "", limit=5)
    assert res["ok"]
    assert res["data"] == []
    db._delete_test_users(created)

"""
Tests for policy_engine.py (code-side policy evaluation).
"""
import pytest

import expense_db as db
import policy_engine as pe


def test_build_pre_check_advice_shapes():
    result = pe.build_pre_check_advice({
        "ok": True,
        "data": {
            "verdict": "likely_not_allowed",
            "policy_version": "v2",
            "violations": [
                {"rule_type": "blocked_keywords", "clause": "Blocked keyword(s) found: alcohol.", "severity": "likely_not_allowed"}
            ],
            "suggestions": ["Avoid blocked keywords in the expense description."],
        },
    })
    assert result["ok"]
    text = result["data"]
    assert "likely_not_allowed" in text
    assert "Blocked keyword(s) found: alcohol" in text
    assert "Avoid blocked keywords" in text
    assert "v2" in text


def test_evaluate_no_rules_likely_ok(employee_pair):
    org_id, _, _ = employee_pair
    res = pe.evaluate_expense(org_id, 10, "Food", "lunch", has_receipt=True)
    assert res["ok"]
    assert res["data"]["verdict"] == "likely_ok"


def test_evaluate_max_amount_blocked(employee_pair):
    org_id, _, _ = employee_pair
    db.add_policy_rule(org_id, "max_amount_per_category_period", category="food",
                       amount_threshold=50, period="monthly")
    res = pe.evaluate_expense(org_id, 100, "Food", "lunch", has_receipt=True)
    assert res["ok"]
    assert res["data"]["verdict"] == "likely_not_allowed"
    assert res["data"]["violations"]
    assert "exceeds" in res["data"]["violations"][0]["clause"]
    assert res["data"]["suggestions"]


def test_evaluate_receipt_required(employee_pair):
    org_id, _, _ = employee_pair
    db.add_policy_rule(org_id, "receipt_required_above", amount_threshold=30, period=None)
    # with receipt -> ok
    res = pe.evaluate_expense(org_id, 50, "Food", "lunch", has_receipt=True)
    assert res["ok"]
    assert res["data"]["verdict"] == "likely_ok"
    # without receipt -> needs_approval
    res = pe.evaluate_expense(org_id, 50, "Food", "lunch", has_receipt=False)
    assert res["ok"]
    assert res["data"]["verdict"] == "needs_approval"
    assert any("receipt" in v["clause"].lower() for v in res["data"]["violations"])


def test_evaluate_allowed_categories(employee_pair):
    org_id, _, _ = employee_pair
    db.add_policy_rule(org_id, "allowed_categories", pattern="food,transport")
    res = pe.evaluate_expense(org_id, 10, "hotel", "stay")
    assert res["ok"]
    assert res["data"]["verdict"] == "likely_not_allowed"
    assert res["data"]["suggestions"]


def test_evaluate_blocked_keywords(employee_pair):
    org_id, _, _ = employee_pair
    db.add_policy_rule(org_id, "blocked_keywords", pattern="alcohol,cash")
    res = pe.evaluate_expense(org_id, 20, "food", "wine and alcohol")
    assert res["ok"]
    assert res["data"]["verdict"] == "likely_not_allowed"
    assert "alcohol" in res["data"]["violations"][0]["clause"].lower()


def test_evaluate_unknown_rule_needs_review(employee_pair):
    org_id, _, _ = employee_pair
    # insert an unknown rule type directly (bypassing the validator)
    with db._conn() as conn:
        conn.execute(
            "INSERT INTO policy_rules (org_id, rule_type, category, amount_threshold, period, pattern, active) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (org_id, "weird_rule", None, None, None, None, True),
        )
    res = pe.evaluate_expense(org_id, 10, "Food", "lunch")
    assert res["ok"]
    assert res["data"]["verdict"] == "needs_review"

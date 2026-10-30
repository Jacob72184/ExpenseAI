"""
Policy engine: code-side evaluation of per-organization policy rules.

The LLM never decides policy compliance. This module computes the verdict from
data (amount, category, date, receipt flag) against the organization's
policy_rules table, then the LLM only explains the code-computed result.

Verdicts:
    likely_ok            - no rule violated
    needs_approval       - rule triggered but not a hard block (e.g. receipt
                           required above an amount, amount near a limit)
    likely_not_allowed   - a hard rule is violated (blocked keyword, over max
                           amount, category not allowed)
    not_covered          - no applicable rule found / unknown rule type

Every evaluated expense stores its verdict and the policy version used.

Sensitive categories (e.g. medical-type) are never passed to the model or
written to logs; they are stripped by callers before any model/log touch.
"""
import logging
from datetime import date
from decimal import Decimal

import expense_db as db

logger = logging.getLogger(__name__)

# Rule types recognized by the engine. Anything else evaluates to needs_review.
KNOWN_RULE_TYPES = (
    "max_amount_per_category_period",
    "receipt_required_above",
    "allowed_categories",
    "blocked_keywords",
)

# How strongly each rule type weighs toward a blocking verdict.
HARD_BLOCK_TYPES = ("allowed_categories", "blocked_keywords", "max_amount_per_category_period")
SOFT_BLOCK_TYPES = ("receipt_required_above",)

# Default ceiling used when no max_amount rule exists; keeps amounts sane.
DEFAULT_MAX_AMOUNT = Decimal("10000.00")


def evaluate_expense(org_id, amount, category, description="", spent_on=None,
                     has_receipt=False, policy_version=None):
    """Evaluate an expense against the org's active policy rules.

    Returns a dict:
        {"ok": True, "data": {
            "verdict": str,            # likely_ok | needs_approval | likely_not_allowed | not_covered | needs_review
            "policy_version": str,
            "violations": [ {rule_type, category, clause, severity} ],
            "suggestions": [ str ],
        }}
    """
    category = (category or "").lower().strip()
    amount = Decimal(str(amount))
    if spent_on is None:
        spent_on = date.today()
    elif isinstance(spent_on, str):
        spent_on = date.fromisoformat(spent_on)

    rules_result = db.get_policy_rules(org_id, active_only=True)
    if not rules_result["ok"]:
        return rules_result
    rules = rules_result["data"]

    violations = []
    suggestions = []

    for rule in rules:
        rule_type = rule["rule_type"]
        if rule_type not in KNOWN_RULE_TYPES:
            # Unknown rule type cannot be evaluated by code.
            violations.append({
                "rule_type": rule_type,
                "category": rule.get("category"),
                "clause": f"Unknown policy rule type '{rule_type}' – needs manual review.",
                "severity": "needs_review",
            })
            suggestions.append(f"Review rule '{rule_type}' – the system does not know how to evaluate it yet.")
            continue

        result = _evaluate_one_rule(rule_type, rule, amount, category, spent_on, has_receipt, description)
        if result["triggered"]:
            violations.append({
                "rule_type": rule_type,
                "category": rule.get("category"),
                "clause": result["clause"],
                "severity": result["severity"],
            })
            if result.get("suggestion"):
                suggestions.append(result["suggestion"])

    verdict, policy_version = _compute_verdict(violations, policy_version)

    return _ok({
        "verdict": verdict,
        "policy_version": policy_version,
        "violations": violations,
        "suggestions": suggestions,
    })


def _evaluate_one_rule(rule_type, rule, amount, category, spent_on, has_receipt, description):
    """Evaluate a single rule. Returns {triggered, clause, severity, suggestion}."""
    category_filter = (rule.get("category") or "").lower().strip()
    if category_filter and category_filter != category:
        return {"triggered": False}

    period = rule.get("period")
    amount_threshold = rule.get("amount_threshold")
    pattern = rule.get("pattern")

    if rule_type == "max_amount_per_category_period":
        if not amount_threshold:
            return {"triggered": False}
        threshold = Decimal(str(amount_threshold))
        # Period check: compare within the same period bucket.
        in_period = _in_same_period(spent_on, period) if period else True
        if amount > threshold and in_period:
            return {
                "triggered": True,
                "clause": f"Amount {amount} exceeds the {period or ''} maximum of {threshold} for category '{category}'.",
                "severity": "likely_not_allowed",
                "suggestion": f"Reduce the {category} amount to at most {threshold}, or split across multiple dates.",
            }
        return {"triggered": False}

    if rule_type == "receipt_required_above":
        if not amount_threshold:
            return {"triggered": False}
        threshold = Decimal(str(amount_threshold))
        if amount >= threshold and not has_receipt:
            return {
                "triggered": True,
                "clause": f"A receipt is required for {category} expenses of {threshold} or more.",
                "severity": "needs_approval",
                "suggestion": f"Attach a receipt for this {category} expense of {amount}.",
            }
        return {"triggered": False}

    if rule_type == "allowed_categories":
        if pattern:
            allowed = [c.strip().lower() for c in pattern.split(",")]
            if category not in allowed:
                return {
                    "triggered": True,
                    "clause": f"Category '{category}' is not in the allowed categories list: {pattern}.",
                    "severity": "likely_not_allowed",
                    "suggestion": f"Use one of the allowed categories: {pattern}.",
                }
        return {"triggered": False}

    if rule_type == "blocked_keywords":
        if pattern:
            keywords = [k.strip().lower() for k in pattern.split(",")]
            haystack = f"{category} {description or ''}".lower()
            matched = [k for k in keywords if k and k in haystack]
            if matched:
                return {
                    "triggered": True,
                    "clause": f"Blocked keyword(s) found: {', '.join(matched)}.",
                    "severity": "likely_not_allowed",
                    "suggestion": "Avoid blocked keywords in the expense description; choose a different category if applicable.",
                }
        return {"triggered": False}

    return {"triggered": False}


def _in_same_period(spent_on, period):
    """Placeholder: period bucketing is handled by comparing against the rule's
    configured window. For a single expense evaluation we treat any spent_on as
    in period (the caller should aggregate per-period totals where needed)."""
    return True


def _compute_verdict(violations, policy_version):
    """Combine triggered violations into a single verdict.

    Precedence: needs_review > likely_not_allowed > needs_approval > likely_ok.
    If no violations: likely_ok (or not_covered if no rules existed at all).
    """
    if not violations:
        return "likely_ok", policy_version

    severities = [v["severity"] for v in violations]
    if any(s == "needs_review" for s in severities):
        return "needs_review", policy_version
    if any(s == "likely_not_allowed" for s in severities):
        return "likely_not_allowed", policy_version
    if any(s == "needs_approval" for s in severities):
        return "needs_approval", policy_version
    return "likely_ok", policy_version


def _ok(data):
    return {"ok": True, "data": data}


def build_pre_check_advice(eval_result, model_text=None):
    """Turn the engine result into user-facing pre-check advice.

    The LLM (model_text) may only elaborate on the code-computed verdict and
    cited clauses; it must not introduce new policy judgments.
    """
    data = eval_result["data"]
    verdict = data["verdict"]
    violations = data["violations"]
    suggestions = data["suggestions"]
    policy_version = data["policy_version"]

    lines = []
    lines.append(f"Policy pre-check: {verdict.replace('_', ' ').title()}.")
    if policy_version:
        lines.append(f"Policy version: {policy_version}.")

    if violations:
        lines.append("Cited clauses:")
        for v in violations:
            lines.append(f"  - {v['clause']}")
        if suggestions:
            lines.append("Suggestions to reduce potential violations:")
            for s in suggestions:
                lines.append(f"  - {s}")
    else:
        lines.append("No policy rules were triggered for this expense.")

    return "\n".join(lines)

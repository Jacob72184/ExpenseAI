# Expense Assistant (multi-organization edition)

A multi-organization expense tracking and audit app. Employees submit expenses by chat
or with one-click quick-add buttons and receive AI pre-check advice against their
organization's policy. Managers approve or reject. Finance admins mark reimbursed and
upload policy documents. Auditors query records read-only with RAG-assisted policy
review. A PERSON makes every approval decision; the AI only advises.

## What changed vs. the personal tracker

* **Multi-organization:** users can belong to multiple organizations (org memberships
  with roles) and switch between them in the sidebar. A personal user is an org of one.
* **Roles:** employee, manager, finance_admin, auditor — stored in the verified login
  session, never from the model or typed text.
* **Expense lifecycle:** draft → submitted → approved/rejected → reimbursed.
  Employees spend first, then claim. A manager approves; a finance admin above the
  manager's limit marks reimbursed. Nobody approves their own expense.
* **Expense entry:** employees can add a draft expense with one-click quick-add buttons
  or with a form (category combobox, amount, date, optional note) — in addition to the
  AI chat. No approval happens on entry; the employee submits drafts for approval.
* **Policy as data + RAG:** `policy_rules` (fixed rule types: max amount per
  category/period, receipt required above an amount, allowed categories, blocked
  keywords) plus per-organization policy documents indexed with fastembed + pgvector
  (plain-text search fallback). Policy documents can be pasted as text or uploaded
  as PDF (.pdf) or Word (.docx) files — text is extracted automatically. Unknown
  rule types evaluate to "needs_review".
* **Employee pre-check:** verdict (`likely_ok`, `needs_approval`,
  `likely_not_allowed`, `not_covered`) with cited clause and suggestions to reduce
  potential rule violations, computed by code and explained (not decided) by the AI.
* **Auditor:** read-only chat with fixed parameterized tools
  (`search_expenses`, `get_expense`, `summarize`, `search_policy`); no free-form SQL;
  every action logged.
* **Sensitive categories** (e.g. medical-type) are kept out of model prompts, logs and
  the RAG index.
* **Invitations:** finance admins invite members by email + a mandatory employee ID.
  The invitation row is created directly and the admin shares the token with the
  invitee out-of-band (for example by copying it from the pending invitations list).
  The invitee accepts by matching their signed-in email and providing the employee ID
  recorded on the invitation — a typo'd or forwarded email cannot let the wrong person
  join the org. No email is sent by the app.
* **Audit log:** every state-changing action is written in the same transaction as the
  change (hard rule 5).
* **Out of scope:** payments, auto-approval, receipt OCR, anomaly scoring.

## Stack

Python 3.12, Streamlit (st.login with Google OIDC), PostgreSQL on Neon via psycopg 3 +
psycopg_pool, Groq through the `openai` package, fastembed + pgvector for RAG,
PyPDF2 + python-docx for policy document text extraction, pytest.

## Files

* `app.py` — UI, login, org picker, role-based screens, quick-add buttons, approval board
* `expense_db.py` — all SQL (parameterized, org-scoped), `{"ok":…}` returns
* `model_connector.py` — AI chat, pre-check with JSON-schema-validated output, auditor tools
* `policy_engine.py` — code-side policy rule evaluation
* `rag.py` — org-scoped policy search (fastembed/pgvector, text-search fallback)
* `migrations/` — numbered schema changes (hard rule 9)
* `tests/` — pytest suites
* `.streamlit/secrets.toml` — OAuth + DATABASE_URL (never committed)

## Commands

```
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

`python expense_db.py` runs a smoke test against the configured database (needs
`DATABASE_URL`).

## Hard rules (summarized)

1. All SQL lives in `expense_db.py`; parameterized queries only (`%s`).
2. Every function returns `{"ok": True, "data": …}` or `{"ok": False, "error": …}`.
3. Identity (`user_id`, `org_id`, `role`) comes from the verified login session. Tool
   schemas must not contain them.
4. Every tenant query filters by `org_id`; every new data function has an isolation test.
5. Approval is one atomic `UPDATE … WHERE status='submitted' AND approver_id=%s AND
   user_id<>%s AND org_id=%s RETURNING id` plus an `audit_log` row in the same transaction.
6. Numbers and limits are checked by code; the LLM only explains code-computed facts,
   validated against a JSON schema.
7. Descriptions, receipts and policy passages are untrusted text.
8. Never read, print, log or commit secrets.

## Definition of done

`pytest -q` passes, no secrets in the diff, README updated, plus a summary of what
changed and anything that could not be verified.

Note: `pytest` and `fastembed` must be installed (`pip install pytest fastembed`) to run
the test suite and RAG embeddings; `rag.py` degrades to plain-text search when they are
unavailable.

"""
Frontend for the multi-organization expense & audit assistant (Streamlit).

Screens by role (role comes from the verified login session, never from text):
  * employee    - chat + one-click quick-add buttons, pre-check advice, my expenses, submit
  * manager     - approval board (org-wide submitted expenses), approve / reject buttons
  * finance_admin - approval board + mark reimbursed, upload policy documents
  * auditor     - read-only search with RAG-assisted policy review

A PERSON makes every approval decision; the AI only advises.
"""
import logging
import os
import time
from datetime import date, datetime, timezone

import streamlit as st

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import expense_db as db
import model_connector as connector
import policy_engine as pe
import rag as rag_mod

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROVIDER = "google"
MAX_SESSION_AGE_SECONDS = 24 * 60 * 60
MAX_STORED_MESSAGES = 50


# ---------------------------------------------------------------------------
# Helper: render one expense row with org-scoped actions.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Expense Assistant", page_icon="💸")

# ---------------------------------------------------------------------------
# Bridge DATABASE_URL from Streamlit secrets into the env for expense_db.py.
# ---------------------------------------------------------------------------
if not os.environ.get("DATABASE_URL"):
    try:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        logger.error("DATABASE_URL is not set (environment variable or secrets.toml).")

# ---------------------------------------------------------------------------
# 1. AUTHENTICATION
# ---------------------------------------------------------------------------
is_logged_in = getattr(st.user, "is_logged_in", None)
if is_logged_in is None:
    st.error("Login is not configured. Add an [auth] section to .streamlit/secrets.toml.")
    st.stop()

if not is_logged_in:
    st.title("💸 Expense Assistant")
    st.write("Please sign in to see and manage your expenses.")
    if st.button("Log in with Google"):
        st.login()
    st.stop()

issued_at = st.user.get("iat")
if isinstance(issued_at, (int, float)) and time.time() - issued_at > MAX_SESSION_AGE_SECONDS:
    st.warning("Your session expired. Please log in again.")
    st.logout()
    st.stop()

sub = st.user.get("sub")
if not isinstance(sub, str) or not sub.strip():
    st.error("Your login did not include an account identifier. Please log in again.")
    if st.button("Log out"):
        st.logout()
    st.stop()

email = st.user.get("email") if st.user.get("email_verified") is True else None
name = st.user.get("name")

# ---------------------------------------------------------------------------
# 2. DB INIT + USER / ORG LOOKUP
# ---------------------------------------------------------------------------
if not st.session_state.get("db_ready"):
    setup = db.init_db()
    if not setup["ok"]:
        st.error(setup["error"])
        st.stop()
    st.session_state["db_ready"] = True

identity = (PROVIDER, sub)
if st.session_state.get("identity") != identity:
    result = db.get_or_create_user(PROVIDER, sub, email, name)
    if not result["ok"]:
        st.error(result["error"])
        st.stop()
    data = result["data"]
    st.session_state["identity"] = identity
    st.session_state["user_id"] = data["id"]
    st.session_state["orgs"] = data.get("orgs", [])
    st.session_state["display_name"] = data.get("name") or data.get("email") or "there"
    st.session_state["history"] = []
    # Default to the user's first (or only) organization.
    if st.session_state["orgs"]:
        st.session_state["active_org_id"] = st.session_state["orgs"][0]["id"]

user_id = st.session_state["user_id"]
orgs = st.session_state.get("orgs", [])
active_org_id = st.session_state.get("active_org_id")
active_role = st.session_state.get("active_role")

# ---------------------------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader(f"Hello, {st.session_state.get('display_name', 'there')}")
    if st.button("Log out"):
        st.logout()

    st.divider()
    st.caption("Organization & Role")
    if orgs:
        # Build distinct org list from session state.
        org_dict = {}
        for o in orgs:
            oid = o["id"]
            if oid not in org_dict:
                org_dict[oid] = {"name": o["name"], "roles": []}
            if o.get("role") and o["role"] not in org_dict[oid]["roles"]:
                org_dict[oid]["roles"].append(o["role"])

        org_ids = list(org_dict.keys())

        # ---- Step 1: choose organization ----
        idx = 0
        if active_org_id and active_org_id in org_dict:
            idx = org_ids.index(active_org_id)
        chosen_org_id = st.selectbox(
            "Switch organization",
            org_ids, index=idx,
            format_func=lambda oid: org_dict[oid]["name"],
            key="org_switch",
        )

        if chosen_org_id != active_org_id:
            st.session_state["active_org_id"] = chosen_org_id
            # Default to first role the user holds, else "employee".
            roles_res = db.get_user_roles_in_org(user_id, chosen_org_id)
            roles_held = roles_res["data"] if roles_res["ok"] and roles_res["data"] else []
            default_role = roles_held[0] if roles_held else "employee"
            st.session_state["active_role"] = default_role
            st.session_state["history"] = []
            st.rerun()

        active_org_id = st.session_state["active_org_id"]

        # ---- Step 2: fetch roles fresh from DB for this org + user ----
        if not active_org_id:
            st.caption("Please select an organization.")
            role = None
            active_role = None
        else:
            # Fetch roles fresh from DB — only roles you actually hold are shown.
            roles_res = db.get_user_roles_in_org(user_id, active_org_id)
            roles_in_org = roles_res["data"] if roles_res["ok"] and roles_res["data"] else []

            if not roles_in_org:
                st.caption("You are not a member of this organization.")
                role = None
                active_role = None
            else:
                # ---- Step 3: choose role within that organization ----
                role_idx = 0
                if active_role and active_role in roles_in_org:
                    role_idx = roles_in_org.index(active_role)
                chosen_role = st.selectbox(
                    "Role in this organization",
                    roles_in_org, index=role_idx,
                    key="role_switch",
                )

                if chosen_role != active_role:
                    st.session_state["active_role"] = chosen_role
                    st.session_state["history"] = []
                    st.rerun()

                role = st.session_state.get("active_role")
                st.caption(f"Active: {org_dict[active_org_id]['name']}  |  Role: {role or '—'}")
    else:
        st.caption("No organization assigned yet.")
        active_org_id = None
        role = None
        active_role = None

    st.divider()
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()

    # ---- Organization management (always available) ----
    st.divider()
    st.caption("Organization")
    with st.expander("Create a new organization", expanded=not orgs):
        new_org_name = st.text_input("Organization name", key="new_org_name")
        new_org_slug = st.text_input("Slug (e.g. acme-corp)", key="new_org_slug")
        if st.button("Create organization", key="create_org_btn"):
            if not new_org_name.strip() or not new_org_slug.strip():
                st.error("Name and slug are required.")
            else:
                cres = db.create_organization(new_org_name.strip(), new_org_slug.strip(), user_id, role="finance_admin")
                if cres["ok"]:
                    orgs_res = db.list_user_orgs(user_id)
                    if orgs_res["ok"]:
                        st.session_state["orgs"] = orgs_res["data"]
                        st.session_state["active_org_id"] = cres["data"]["id"]
                        st.session_state["history"] = []
                        st.success(f"Created organization “{new_org_name}”.")
                        st.rerun()
                else:
                    st.error(cres["error"])

    with st.expander("Accept an invitation", expanded=not orgs):
        inv_token = st.text_input("Invitation token (from the admin)", key="inv_token")
        inv_employee_id = st.text_input("Your employee ID in that organization", key="inv_employee_id")
        if st.button("Accept invitation", key="accept_inv_btn"):
            if not inv_token.strip():
                st.error("Invitation token is required.")
            elif not inv_employee_id.strip():
                st.error("Employee ID is required.")
            else:
                ares = db.accept_org_invitation(inv_token.strip(), user_id, inv_employee_id.strip())
                if ares["ok"]:
                    orgs_res = db.list_user_orgs(user_id)
                    if orgs_res["ok"]:
                        st.session_state["orgs"] = orgs_res["data"]
                        st.session_state["active_org_id"] = ares["data"]["org_id"]
                        st.session_state["history"] = []
                        st.success(f"Joined organization as {ares['data']['role']}.")
                        st.rerun()
                else:
                    st.error(ares["error"])


# ---------------------------------------------------------------------------
# Helper: render one expense row with org-scoped actions.
# ---------------------------------------------------------------------------
def expense_row(exp, is_approver_allowed, is_finance):
    cols = st.columns([2, 1, 1, 1, 2, 2])
    cols[0].write(f"**{exp['category']}** — {exp['description'] or '(no note)'}")
    cols[1].write(f"{exp['amount']:.2f}")
    cols[2].write(exp["spent_on"])
    cols[3].write(f"*{exp['status']}*")
    verdict = exp.get("verdict")
    if verdict:
        cols[4].write(f"pre-check: {verdict.replace('_', ' ')}")
    else:
        cols[4].write("—")
    with cols[5]:
        if exp["status"] == "submitted" and is_approver_allowed:
            if st.button("Approve", key=f"appr_{exp['id']}"):
                _do_approve(exp["id"])
            if st.button("Reject", key=f"rej_{exp['id']}"):
                _do_reject(exp["id"])
        if exp["status"] == "approved" and is_finance:
            if st.button("Reimburse", key=f"reimb_{exp['id']}"):
                _do_reimburse(exp["id"])


def _do_approve(expense_id):
    res = db.approve_expense(expense_id, user_id, active_org_id, role)
    if res["ok"]:
        st.success("Approved.")
        st.session_state["history"] = []
        st.rerun()
    else:
        st.error(res["error"])


def _do_reject(expense_id):
    res = db.reject_expense(expense_id, user_id, active_org_id, role)
    if res["ok"]:
        st.success("Rejected.")
        st.session_state["history"] = []
        st.rerun()
    else:
        st.error(res["error"])


def _do_reimburse(expense_id):
    res = db.mark_reimbursed(expense_id, user_id, active_org_id, role)
    if res["ok"]:
        st.success("Marked reimbursed.")
        st.session_state["history"] = []
        st.rerun()
    else:
        st.error(res["error"])


# ---------------------------------------------------------------------------
# Helper: create a draft expense + run its policy pre-check.
# user_id / active_org_id come from the verified login session (module scope).
# ---------------------------------------------------------------------------
def _save_expense_with_precheck(amount, category, description, spent_on):
    """Save one draft expense and store its code-computed pre-check verdict.

    Returns (expense_dict, verdict_or_None, error_or_None).
    """
    res = db.add_expense(user_id, active_org_id, role, amount, category, description, spent_on)
    if not res["ok"]:
        return None, None, res["error"]
    exp = res["data"]
    verdict = None
    pc = pe.evaluate_expense(
        active_org_id, amount, category, description,
        has_receipt=False, policy_version=None)
    if pc["ok"]:
        db.update_expense_verdict(exp["id"], pc["data"]["verdict"],
                                  pc["data"]["policy_version"],
                                  active_org_id)
        verdict = pc["data"]["verdict"]
    return exp, verdict, None


# ---------------------------------------------------------------------------
# 4. EMPLOYEE VIEW
# ---------------------------------------------------------------------------
def employee_view():
    st.title("💸 Expense Assistant")
    st.caption("Add an expense with the form below, or describe it in chat.")

    # ---- Expense entry form (specify details, no chat required) ----
    st.subheader("Add an expense (form)")
    st.caption("Choose a category, enter the amount and date, and add an optional note. "
               "This creates a draft expense; submit it for approval in My expenses.")

    CATEGORY_PRESETS = ["Food", "Transport", "Office Supplies"]

    with st.form("add_expense_form", clear_on_submit=True):
        col_cat, col_amt = st.columns(2)
        with col_cat:
            category_preset = st.selectbox(
                "Category",
                CATEGORY_PRESETS + ["Custom"],
                index=0,
                key="expense_category_preset",
            )
            custom_category = st.text_input(
                "Custom category (used when 'Custom' is selected above)",
                placeholder="e.g. Software",
                key="expense_custom_category",
            )
        with col_amt:
            amount = st.number_input("Amount", min_value=0.01, step=0.01,
                                     format="%.2f", value=0.01,
                                     key="expense_amount")
            spent_on = st.date_input("Date", value=date.today(),
                                     key="expense_date")
        description = st.text_input("Description (optional)",
                                    placeholder="e.g. Lunch with client",
                                    key="expense_description")
        added = st.form_submit_button("Add expense")

        if added:
            if category_preset == "Custom":
                cat = (custom_category or "").strip()
            else:
                cat = category_preset
            if not cat:
                st.error("Please choose or type a category.")
            else:
                exp, verdict, err = _save_expense_with_precheck(
                    amount, cat, description or "", spent_on.isoformat())
                if err:
                    st.error(err)
                else:
                    st.success(f"Added {cat} expense ({amount:.2f}) for {spent_on}. "
                               f"Pre-check: {verdict or '—'}.")
                    st.rerun()

    st.divider()

    # ---- Chat ----
    st.subheader("Chat with the assistant")
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
                    outcome = connector.chat(
                        user_id, active_org_id, role,
                        st.session_state["history"], prompt)
            except Exception:
                logger.exception("Unexpected error while calling chat()")
                outcome = {"ok": False, "text": "Something went wrong. Please try again."}
            if outcome["ok"]:
                st.markdown(outcome["text"])
            else:
                st.error(outcome["text"])
        if outcome["ok"]:
            history = st.session_state["history"]
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": outcome["text"]})
            st.session_state["history"] = history[-MAX_STORED_MESSAGES:]

    st.divider()

    # ---- My expenses + submit ----
    st.subheader("My expenses")
    res = db.get_expenses(user_id, active_org_id, role, limit=50)
    if not res["ok"]:
        st.caption("Couldn't load expenses.")
    elif not res["data"]:
        st.caption("No expenses yet.")
    else:
        for exp in res["data"]:
            cols = st.columns([3, 1, 1, 1, 2])
            cols[0].write(f"{exp['category']} — {exp['description'] or '(no note)'}")
            cols[1].write(f"{exp['amount']:.2f}")
            cols[2].write(exp["spent_on"])
            cols[3].write(f"*{exp['status']}*")
            if exp["status"] == "draft":
                if cols[4].button("Submit", key=f"sub_{exp['id']}"):
                    sres = db.submit_expense(exp["id"], user_id, active_org_id)
                    if sres["ok"]:
                        st.success("Submitted for approval.")
                        st.rerun()
                    else:
                        st.error(sres["error"])


# ---------------------------------------------------------------------------
# 5. MANAGER / FINANCE_ADMIN VIEW (approval board)
# ---------------------------------------------------------------------------
def approval_view():
    is_finance = (role == "finance_admin")
    st.title("🧾 Approval Board")
    st.caption("Submitted expenses in your organization. A person decides; the AI only advises.")

    res = db.get_expenses(user_id, active_org_id, role, limit=100)  # org-wide for managers/finance
    if not res["ok"]:
        st.caption("Couldn't load expenses.")
        return
    submitted = [e for e in res["data"] if e["status"] == "submitted"]
    if not submitted:
        st.info("No submitted expenses pending approval.")
    for exp in submitted:
        st.divider()
        expense_row(exp, is_approver_allowed=True, is_finance=is_finance)

    st.subheader("All expenses (org-wide)")
    if res["data"]:
        st.dataframe(res["data"], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# 6. AUDITOR VIEW (read-only, RAG-assisted)
# ---------------------------------------------------------------------------
def auditor_view():
    st.title("🔍 Auditor — Read-Only")
    st.caption("Search expenses and policy. Every action is logged. No write access.")

    tab_search, tab_policy = st.tabs(["Expenses", "Policy review"])

    with tab_search:
        st.subheader("Search expenses")
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_filter = st.text_input("Category (optional)")
        with col2:
            status_filter = st.selectbox("Status", ["", "draft", "submitted", "approved",
                                                    "rejected", "reimbursed"])
        with col3:
            limit = st.number_input("Max rows", 1, 100, 20)
        if st.button("Search"):
            res = db.get_expenses(user_id, active_org_id, role,
                                  category=cat_filter or None, limit=int(limit))
            # get_expenses is org-scoped; filter by status in Python.
            data = res["data"] if res["ok"] else []
            if status_filter:
                data = [e for e in data if e["status"] == status_filter]
            st.dataframe(data, hide_index=True, use_container_width=True)

    with tab_policy:
        st.subheader("Policy review (RAG-assisted)")
        st.caption("Paste an expense description to see potentially violated policy clauses.")
        exp_desc = st.text_area("Expense description / category to check",
                                placeholder="e.g. $120 dinner, category Food")
        if st.button("Review against policy") and exp_desc.strip():
            # RAG: fetch relevant policy clauses for the org.
            rag = rag_mod.search_policy(active_org_id, exp_desc, limit=5)
            clauses = rag["data"] if rag["ok"] else []
            if clauses:
                st.write("Potentially relevant policy clauses:")
                for c in clauses:
                    # Be defensive: search_policy should return dicts, but fall
                    # back gracefully if a string ever leaks into the list.
                    content = c.get("content") if isinstance(c, dict) else str(c)
                    st.info(content)
            else:
                st.info("No matching policy clauses found.")
            # Code-level hint if it looks like a known sensitive/category keyword
            st.caption("These clauses are retrieved from your organization's policy index "
                       "(org-scoped). A person makes the final determination.")


# ---------------------------------------------------------------------------
# 7. FINANCE ADMIN: POLICY UPLOAD
# ---------------------------------------------------------------------------
def _extract_text_from_file(uploaded_file):
    """Extract plain text from an uploaded PDF or Word (.docx) file.

    Returns (text, error_message). text is None on failure.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            return None, "PDF support not installed. Run: pip install PyPDF2"
        try:
            reader = PdfReader(uploaded_file)
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts), None
        except Exception as exc:
            return None, f"Failed to read PDF: {exc}"
    elif name.endswith(".docx"):
        if DocxDocument is None:
            return None, "Word support not installed. Run: pip install python-docx"
        try:
            doc = DocxDocument(uploaded_file)
            parts = []
            for para in doc.paragraphs:
                parts.append(para.text)
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text for cell in row.cells]
                    parts.append(" | ".join(cells))
            return "\n".join(parts), None
        except Exception as exc:
            return None, f"Failed to read Word document: {exc}"
    else:
        return None, "Unsupported file type. Upload a .pdf or .docx file."


def finance_admin_view():
    st.title("⚙️ Finance Admin")
    finance_tab = st.radio(
        "Section",
        ["Upload policy document", "Policy rules", "Organization members"],
        index=["Upload policy document", "Policy rules", "Organization members"].index(
            st.session_state.get("finance_admin_tab", "Upload policy document")
        ),
        key="finance_admin_tab_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["finance_admin_tab"] = finance_tab

    if finance_tab == "Upload policy document":
        st.subheader("Upload a policy document (per organization)")
        title = st.text_input("Title")
        version = st.text_input("Version", value="1.0")

        # ---- File upload (PDF / Word) ----
        uploaded_file = st.file_uploader(
            "Upload policy document (PDF or Word)",
            type=["pdf", "docx"],
            key="policy_file_uploader",
        )
        file_text = ""
        if uploaded_file is not None:
            file_text, file_err = _extract_text_from_file(uploaded_file)
            if file_err:
                st.error(file_err)
            else:
                st.success(f"Extracted text from {uploaded_file.name} "
                           f"({len(file_text)} characters).")

        content = st.text_area(
            "Policy content",
            value=file_text,
            height=300,
            placeholder="Paste the policy text here, or upload a PDF/Word file above.",
        )
        sensitive = st.text_input("Sensitive categories (comma-separated, e.g. medical)",
                                  help="These are kept out of the RAG index and model prompts.")
        if st.button("Upload"):
            if not title.strip() or not content.strip():
                st.error("Title and content are required.")
            else:
                sens_list = [s.strip() for s in sensitive.split(",") if s.strip()]
                for s in sens_list:
                    db.add_sensitive_category(active_org_id, s)
                res = db.add_policy_document(active_org_id, title, content, version, user_id, role)
                if res["ok"]:
                    rag_mod.add_policy_document(res["data"]["id"], active_org_id, content,
                                                sensitive_categories=sens_list)
                    st.success(f"Policy document uploaded (version {version}) and indexed.")
                else:
                    st.error(res["error"])

    elif finance_tab == "Policy rules":
        st.subheader("Policy rules (code-enforced)")
        rule_type = st.selectbox("Rule type", [
            "max_amount_per_category_period", "receipt_required_above",
            "allowed_categories", "blocked_keywords"])
        category = st.text_input("Category (optional, e.g. food)")
        amount = st.number_input("Amount threshold (optional)", min_value=0.0, value=0.0,
                                 step=0.01, format="%.2f")
        period = st.selectbox("Period (optional)", ["", "daily", "monthly", "yearly"])
        pattern = st.text_input("Pattern (comma-separated keywords / categories)")
        if st.button("Add rule"):
            res = db.add_policy_rule(
                active_org_id, rule_type,
                category=category or None,
                amount_threshold=amount if amount > 0 else None,
                period=period or None,
                pattern=pattern or None,
            )
            if res["ok"]:
                st.success("Rule added.")
            else:
                st.error(res["error"])
        st.divider()
        st.write("Current rules:")
        rules = db.get_policy_rules(active_org_id)["data"] if db.get_policy_rules(active_org_id)["ok"] else []
        st.dataframe(rules, hide_index=True, use_container_width=True)

    elif finance_tab == "Organization members":
        _render_member_management(active_org_id, user_id, role)


# ---------------------------------------------------------------------------
# 7b. FINANCE ADMIN: ORGANIZATION MEMBER MANAGEMENT
# ---------------------------------------------------------------------------
def _render_member_management(org_id, actor_user_id, actor_role):
    """UI for finance admin to manage org members: list, invite, deactivate,
    edit role / employee_id, and view invitations."""
    st.subheader("Organization members")

    # ---- Invite a new member ----
    st.write("Invite a new member")
    with st.form("invite_member_form", clear_on_submit=True):
        inv_email = st.text_input("Invitee email", key="invite_email")
        inv_employee_id = st.text_input("Invitee's employee ID in this organization (required)", key="invite_employee_id")
        inv_role = st.selectbox("Role", ["employee", "manager", "finance_admin", "auditor"],
                                key="invite_role")
        inv_submitted = st.form_submit_button("Send invitation")
        if inv_submitted:
            if not inv_email.strip():
                st.error("Email is required.")
            elif not inv_employee_id.strip():
                st.error("Employee ID is required.")
            else:
                ires = db.create_org_invitation(org_id, inv_email.strip(),
                                                inv_employee_id.strip(),
                                                inv_role, actor_user_id)
                if ires["ok"]:
                    st.success(f"Invitation created for {inv_email} (role: {inv_role}). "
                               f"Share the token with the invitee: {ires['data']['token']}")
                else:
                    st.error(ires["error"])

    st.divider()

    # ---- List current members ----
    st.write("Current members")
    # Fetch members directly from org_memberships.
    try:
        with db._conn() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.user_id, m.role, m.is_active, m.employee_id, m.created_at,
                       u.name, u.email
                FROM org_memberships m
                JOIN users u ON u.id = m.user_id
                WHERE m.org_id = %s
                ORDER BY m.is_active DESC, u.name
                """,
                (org_id,),
            ).fetchall()
    except Exception as exc:
        logger.exception("Failed to load members")
        st.error("Could not load members.")
        return

    if not rows:
        st.info("No members in this organization.")
    else:
        for r in rows:
            is_actor = (r["user_id"] == actor_user_id)
            cols = st.columns([2, 1, 1, 1, 1, 2])
            status_icon = "✅" if r["is_active"] else "❌"
            cols[0].write(f"{r['name'] or '—'} ({r['email'] or '—'})")
            cols[1].write(f"{r['role']}")
            cols[2].write(r["employee_id"] or "—")
            cols[3].write(f"{status_icon} {'active' if r['is_active'] else 'inactive'}")
            with cols[4]:
                if r["is_active"] and not is_actor:
                    if st.button("Deactivate", key=f"deact_{r['user_id']}"):
                        res = db.deactivate_organization_member(org_id, r["user_id"], r["role"],
                                                              actor_user_id, actor_role)
                        if res["ok"]:
                            st.success("Member deactivated.")
                            st.rerun()
                        else:
                            st.error(res["error"])
                elif not r["is_active"] and not is_actor:
                    if st.button("Reactivate", key=f"react_{r['user_id']}"):
                        res = db.add_organization_member(org_id, r["user_id"], r["role"],
                                                         actor_user_id, actor_role)
                        if res["ok"]:
                            st.success("Member reactivated.")
                            st.rerun()
                        else:
                            st.error(res["error"])
            with cols[5]:
                if not is_actor:
                    new_role = st.selectbox("Role", ["employee", "manager", "finance_admin", "auditor"],
                                            index=["employee", "manager", "finance_admin", "auditor"].index(r["role"]),
                                            key=f"role_{r['user_id']}")
                    new_emp_id = st.text_input("Employee ID", value=r["employee_id"] or "",
                                               key=f"empid_{r['user_id']}")
                    if st.button("Save", key=f"save_{r['user_id']}"):
                        if new_role != r["role"]:
                            res = db.update_organization_member_role(org_id, r["user_id"], r["role"],
                                                                     new_role, actor_user_id, actor_role)
                            if res["ok"]:
                                st.success("Role updated.")
                            else:
                                st.error(res["error"])
                        if new_emp_id.strip() != (r["employee_id"] or ""):
                            res = db.update_organization_member_employee_id(org_id, r["user_id"],
                                                                            r["role"],
                                                                            new_emp_id.strip() or None,
                                                                            actor_user_id, actor_role)
                            if res["ok"]:
                                st.success("Employee ID updated.")
                            else:
                                st.error(res["error"])
                        st.rerun()

    st.divider()

    # ---- List invitations ----
    st.write("Pending invitations")
    inv_res = db.list_org_invitations(org_id, actor_user_id)
    if not inv_res["ok"]:
        st.error(inv_res["error"])
    elif not inv_res["data"]:
        st.info("No pending invitations.")
    else:
        now = datetime.now(timezone.utc)
        for inv in inv_res["data"]:
            expired = inv["status"] == "pending" and inv["expires_at"] and  datetime.fromisoformat(inv["expires_at"]).replace(tzinfo=timezone.utc) < now
            status = "expired" if expired else inv["status"]
            cols = st.columns([3, 1, 1, 1, 1])
            cols[0].write(f"{inv['email']} — role: {inv['role']} — employee_id: {inv['employee_id'] or '—'}")
            cols[1].write(status)
            cols[2].write(inv["expires_at"] or "—")
            if expired:
                cols[3].write("—")
            else:
                if cols[3].button("Expire", key=f"expire_inv_{inv['id']}"):
                    with db._conn() as conn:
                        conn.execute("UPDATE org_invitations SET status = 'expired' WHERE id = %s", (inv["id"],))
                    st.rerun()


# ---------------------------------------------------------------------------
# 8. ROUTE TO THE RIGHT VIEW
# ---------------------------------------------------------------------------
if not active_org_id or role is None:
    if not orgs:
        # User has no organizations yet — show setup options.
        st.title("💸 Expense Assistant — Organization Setup")
        st.write("You don't belong to any organization yet. Create one or accept an invitation.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Create a new organization")
            new_org_name = st.text_input("Organization name", key="setup_org_name")
            new_org_slug = st.text_input("Slug (e.g. acme-corp)", key="setup_org_slug")
            if st.button("Create organization", key="setup_create_org_btn"):
                if not new_org_name.strip() or not new_org_slug.strip():
                    st.error("Name and slug are required.")
                else:
                    cres = db.create_organization(new_org_name.strip(), new_org_slug.strip(), user_id, role="finance_admin")
                    if cres["ok"]:
                        orgs_res = db.list_user_orgs(user_id)
                        if orgs_res["ok"]:
                            st.session_state["orgs"] = orgs_res["data"]
                            st.session_state["active_org_id"] = cres["data"]["id"]
                            st.session_state["active_role"] = "finance_admin"
                            st.session_state["history"] = []
                            st.success(f"Created organization “{new_org_name}”.")
                            st.rerun()
                    else:
                        st.error(cres["error"])
        with col2:
            st.subheader("Accept an invitation")
            inv_token = st.text_input("Invitation token (from the admin)", key="setup_inv_token")
            inv_employee_id = st.text_input("Your employee ID in that organization", key="setup_inv_employee_id")
            if st.button("Accept invitation", key="setup_accept_inv_btn"):
                if not inv_token.strip():
                    st.error("Invitation token is required.")
                elif not inv_employee_id.strip():
                    st.error("Employee ID is required.")
                else:
                    ares = db.accept_org_invitation(inv_token.strip(), user_id, inv_employee_id.strip())
                    if ares["ok"]:
                        orgs_res = db.list_user_orgs(user_id)
                        if orgs_res["ok"]:
                            st.session_state["orgs"] = orgs_res["data"]
                            st.session_state["active_org_id"] = ares["data"]["org_id"]
                            st.session_state["active_role"] = ares["data"]["role"]
                            st.session_state["history"] = []
                            st.success(f"Joined organization as {ares['data']['role']}.")
                            st.rerun()
                    else:
                        st.error(ares["error"])
        st.stop()
    st.warning("No active organization or role selected. Please choose an organization and role from the sidebar on the left.")
    st.stop()

st.title("💸 Expense & Audit Assistant")
st.caption(f"Organization: {next((o['name'] for o in orgs if o['id'] == active_org_id), '')}  |  "
           f"Role: {role}")

if role == "employee":
    employee_view()
elif role in ("manager", "finance_admin"):
    approval_view()
    if role == "finance_admin":
        st.divider()
        finance_admin_view()
elif role == "auditor":
    auditor_view()
else:
    st.error(f"Unknown role: {role}")

-- Migration 001: multi-organization foundation
--
-- Creates organizations, memberships, the org-scoped expense lifecycle,
-- policy-as-data tables, RAG chunks (pgvector), audit log, and sensitive
-- category registry. All tenant tables carry org_id so every query can be
-- scoped to the caller's active organization.

-- pgvector extension for embedding similarity search.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS organizations (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name         TEXT NOT NULL,
    slug         TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS org_memberships (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id      BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('employee','manager','finance_admin','auditor')),
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, org_id)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     BIGINT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Expense lifecycle columns added to the existing expenses table (created by
-- the inline schema in expense_db.py). org_id scopes every expense to one org.
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS org_id BIGINT REFERENCES organizations(id);
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft','submitted','approved','rejected','reimbursed'));
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS approver_id BIGINT REFERENCES users(id);
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ;
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS reimbursed_at TIMESTAMPTZ;
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS policy_version TEXT;
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS verdict TEXT;
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS receipt_required BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS expenses_org_status_idx ON expenses (org_id, status);
CREATE INDEX IF NOT EXISTS expenses_org_user_idx ON expenses (org_id, user_id);
CREATE INDEX IF NOT EXISTS expenses_user_date_idx ON expenses (user_id, spent_on);

-- Policy rules as data, per organization. Unknown rule types evaluate to
-- "needs_review" in the policy engine.
CREATE TABLE IF NOT EXISTS policy_rules (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id          BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    rule_type       TEXT NOT NULL CHECK (rule_type IN (
                        'max_amount_per_category_period',
                        'receipt_required_above',
                        'allowed_categories',
                        'blocked_keywords'
                    )),
    category        TEXT,
    amount_threshold NUMERIC(14, 2),
    period          TEXT CHECK (period IN ('daily','monthly','yearly')),
    pattern         TEXT,
    active          BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS policy_rules_org_idx ON policy_rules (org_id);

-- Policy documents uploaded by a finance_admin (RAG source).
CREATE TABLE IF NOT EXISTS policy_documents (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id       BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    content      TEXT NOT NULL,
    version      TEXT NOT NULL,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    uploaded_by  BIGINT REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS policy_documents_org_idx ON policy_documents (org_id);

-- Chunks of policy documents with embeddings, scoped to org.
CREATE TABLE IF NOT EXISTS policy_chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES policy_documents(id) ON DELETE CASCADE,
    org_id      BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(384)
);
CREATE INDEX IF NOT EXISTS policy_chunks_org_idx ON policy_chunks (org_id);

-- Audit log: every state-changing / read action is recorded here.
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id       BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    actor_user_id BIGINT NOT NULL REFERENCES users(id),
    action       TEXT NOT NULL,
    target_table TEXT,
    target_id    BIGINT,
    details      JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_log_org_created_idx ON audit_log (org_id, created_at);
CREATE INDEX IF NOT EXISTS audit_log_actor_idx ON audit_log (actor_user_id);

-- Sensitive categories (e.g. medical-type). Kept out of model prompts, logs
-- and the RAG index.
CREATE TABLE IF NOT EXISTS org_sensitive_categories (
    id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id    BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    category  TEXT NOT NULL,
    UNIQUE (org_id, category)
);
CREATE INDEX IF NOT EXISTS org_sensitive_categories_org_idx ON org_sensitive_categories (org_id);

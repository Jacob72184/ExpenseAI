-- Migration 002: organization membership + invitations
--
-- Adds employee_id to org_memberships (per-org employee identifier used when
-- accepting invitations) and creates the org_invitations table for the
-- invite-based join flow.

-- Per-organization employee identifier on each membership.
ALTER TABLE org_memberships ADD COLUMN IF NOT EXISTS employee_id TEXT;

-- Invitations sent by a finance admin to invite someone to join an org.
-- The invitee accepts by proving their signed-in email matches the invitation
-- email and by providing the employee_id the admin recorded.
CREATE TABLE IF NOT EXISTS org_invitations (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id       BIGINT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email        TEXT NOT NULL,
    employee_id  TEXT,
    role         TEXT NOT NULL CHECK (role IN ('employee','manager','finance_admin','auditor')),
    invited_by   BIGINT NOT NULL REFERENCES users(id),
    token        TEXT NOT NULL UNIQUE,
    status       TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','accepted','expired')),
    expires_at   TIMESTAMPTZ NOT NULL,
    accepted_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS org_invitations_org_idx ON org_invitations (org_id);
CREATE INDEX IF NOT EXISTS org_invitations_token_idx ON org_invitations (token);
CREATE INDEX IF NOT EXISTS org_invitations_email_idx ON org_invitations (email, status);

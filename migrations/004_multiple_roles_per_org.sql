-- Migration 004: allow multiple roles per user per organization
--
-- A person can hold multiple roles in the same organization (for example
-- both employee and manager). Change the unique constraint to include role
-- so a user can have multiple membership rows in one org.

ALTER TABLE org_memberships DROP CONSTRAINT IF EXISTS org_memberships_user_id_org_id_key;
ALTER TABLE org_memberships ADD CONSTRAINT org_memberships_user_id_org_id_role_key UNIQUE (user_id, org_id, role);

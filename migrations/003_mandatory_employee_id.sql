-- Migration 003: make employee_id mandatory on invitations
--
-- employee_id is now required when creating and accepting invitations, so a
-- typo'd or forwarded email cannot let the wrong person join an organization.
-- Existing pending invitations without an employee_id should be re-issued by
-- the admin before they can be accepted.

ALTER TABLE org_invitations ALTER COLUMN employee_id SET NOT NULL;

-- ============================================================
-- simple_customer_view.sql
-- Customer profile passthrough view for the reporting layer
-- Owner: Data Engineering  |  Schedule: on-demand
-- ============================================================

CREATE OR REPLACE VIEW analytics.customer_profile AS

SELECT
    account_id,
    full_name,
    email,
    signup_date,
    plan_tier,
    billing_country
FROM raw.crm_accounts
WHERE is_active = TRUE;

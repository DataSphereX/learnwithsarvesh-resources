-- ============================================================
-- daily_revenue_rollup.sql
-- Daily revenue rollup by plan tier and region, for finance reporting
-- Owner: Data Engineering  |  Schedule: daily 03:00 UTC
-- ============================================================

CREATE OR REPLACE TABLE warehouse.daily_revenue_rollup AS

WITH billed_amounts AS (
    SELECT
        s.account_id,
        s.plan_id,
        s.billing_date,
        s.monthly_amount,
        CASE
            WHEN s.currency <> 'USD' THEN s.monthly_amount * s.fx_rate_to_usd
            ELSE s.monthly_amount
        END AS amount_usd
    FROM raw.billing_subscriptions s
    WHERE s.status = 'active'
)

SELECT
    b.billing_date,
    a.plan_tier,
    a.billing_country,
    COUNT(DISTINCT b.account_id) AS paying_accounts,
    SUM(b.amount_usd)            AS revenue_usd,
    AVG(b.amount_usd)            AS avg_revenue_per_account
FROM billed_amounts b
JOIN raw.crm_accounts a
    ON b.account_id = a.account_id
GROUP BY b.billing_date, a.plan_tier, a.billing_country;

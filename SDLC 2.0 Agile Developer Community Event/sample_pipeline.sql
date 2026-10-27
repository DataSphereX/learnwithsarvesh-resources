-- ============================================================
-- customer_churn_features.sql
-- Feature engineering pipeline for the churn prediction model
-- Owner: Data Engineering  |  Schedule: daily 02:00 UTC
-- ============================================================

CREATE OR REPLACE TABLE analytics.customer_churn_features AS

WITH raw_accounts AS (
    SELECT
        account_id,
        signup_date,
        plan_tier,
        billing_country,
        is_active
    FROM raw.crm_accounts
    WHERE _ingested_at >= DATEADD(day, -90, CURRENT_DATE())
),

subscription_history AS (
    SELECT
        s.account_id,
        COUNT(DISTINCT s.subscription_id)      AS total_subscriptions,
        SUM(s.monthly_amount)                  AS lifetime_value,
        MAX(s.renewal_date)                    AS last_renewal_date,
        AVG(DATEDIFF(day, s.start_date, s.end_date)) AS avg_subscription_days
    FROM raw.billing_subscriptions s
    JOIN raw.billing_plans p
        ON s.plan_id = p.plan_id
    WHERE s.status IN ('active', 'churned')
    GROUP BY s.account_id
),

support_signals AS (
    SELECT
        t.account_id,
        COUNT(*)                               AS ticket_count_90d,
        SUM(CASE WHEN t.priority = 'P1' THEN 1 ELSE 0 END) AS critical_tickets,
        AVG(t.resolution_hours)                AS avg_resolution_hours,
        MAX(c.sentiment_score)                 AS worst_sentiment
    FROM raw.support_tickets t
    LEFT JOIN enriched.ticket_sentiment c
        ON t.ticket_id = c.ticket_id
    WHERE t.created_at >= DATEADD(day, -90, CURRENT_DATE())
    GROUP BY t.account_id
),

product_usage AS (
    SELECT
        u.account_id,
        SUM(u.session_count)                   AS sessions_30d,
        SUM(u.feature_events)                  AS feature_events_30d,
        COUNT(DISTINCT u.active_date)           AS active_days_30d
    FROM warehouse.daily_product_usage u
    WHERE u.active_date >= DATEADD(day, -30, CURRENT_DATE())
    GROUP BY u.account_id
)

SELECT
    a.account_id,
    a.plan_tier,
    a.billing_country,
    DATEDIFF(day, a.signup_date, CURRENT_DATE())        AS account_age_days,

    COALESCE(sh.total_subscriptions, 0)                 AS total_subscriptions,
    COALESCE(sh.lifetime_value, 0)                      AS lifetime_value,
    COALESCE(sh.avg_subscription_days, 0)               AS avg_subscription_days,

    COALESCE(ss.ticket_count_90d, 0)                    AS ticket_count_90d,
    COALESCE(ss.critical_tickets, 0)                    AS critical_tickets,
    COALESCE(ss.worst_sentiment, 0)                     AS worst_sentiment,

    COALESCE(pu.sessions_30d, 0)                        AS sessions_30d,
    COALESCE(pu.active_days_30d, 0)                     AS active_days_30d,

    -- Derived engagement ratio used directly as a model feature
    CASE
        WHEN pu.active_days_30d IS NULL OR pu.active_days_30d = 0 THEN 0
        ELSE pu.feature_events_30d / pu.active_days_30d
    END                                                 AS events_per_active_day,

    NOT a.is_active                                     AS churn_label

FROM raw_accounts a
LEFT JOIN subscription_history sh ON a.account_id = sh.account_id
LEFT JOIN support_signals     ss ON a.account_id = ss.account_id
LEFT JOIN product_usage       pu ON a.account_id = pu.account_id;

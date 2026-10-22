"""
Churn HW: target + features as-of + uplift table.
Train snap 2025-06-30, test snap 2025-09-25.
Lag=6 after snap: features < snap, label (snap+lag, snap+lag+horizon].
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DSN = os.environ["ACADEMY_DSN"]
OUT = Path(__file__).resolve().parent
DATA_END = "2025-12-31"
HORIZON = 90
LAG = 6  # дней после snap до старта окна метки (задержка кампании)
MIN_TENURE = 60
LOOKBACK_ALIVE = 90

TRAIN_SNAP = "2025-06-30"
TEST_SNAP = "2025-09-25"


def q(engine, sql: str, params=None) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine, params=params)


FEATURE_SQL = """
WITH params AS (
    SELECT
        DATE :snap AS snap,
        DATE :snap AS asof,
        (DATE :snap + :lag)::date AS label_start,
        (DATE :snap + :lag + :horizon)::date AS label_end
),
pop AS (
    SELECT
        c.customer_id,
        c.signup_date,
        c.gender,
        c.region,
        c.acq_channel,
        c.segment,
        c.is_payroll,
        c.declared_income_uzs,
        c.birth_date,
        p.snap,
        (p.snap - c.signup_date) AS tenure_days,
        EXTRACT(YEAR FROM AGE(p.snap, c.birth_date))::int AS age_years,
        (
            MAKE_DATE(
                EXTRACT(YEAR FROM p.snap)::int
                + CASE WHEN TO_CHAR(c.birth_date, 'MMDD') < TO_CHAR(p.snap, 'MMDD') THEN 1 ELSE 0 END,
                EXTRACT(MONTH FROM c.birth_date)::int,
                LEAST(EXTRACT(DAY FROM c.birth_date)::int, 28)
            ) - p.snap
        ) AS days_to_bday
    FROM params p
    JOIN churn_customers c
      ON c.signup_date <= p.snap - :min_tenure
    WHERE p.label_end <= DATE '2025-12-31'
),
txn AS (
    SELECT
        t.customer_id,
        t.txn_ts AS ts,
        t.txn_ts::date AS d,
        t.txn_type,
        t.mcc_group,
        t.channel,
        t.card_id,
        t.amount_uzs,
        t.status
    FROM churn_transactions t
    JOIN params p ON TRUE
    WHERE t.txn_ts::date < p.snap
),
txn_agg AS (
    SELECT
        customer_id,
        MAX(d) FILTER (WHERE status = 'approved') AS last_txn_date,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 14) AS txn_cnt_14,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30) AS txn_cnt_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 60) AS txn_cnt_60,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS txn_cnt_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 180) AS txn_cnt_180,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90), 0) AS gmv_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30), 0) AS gmv_30,
        COALESCE(AVG(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30), 0) AS avg_ticket_30,
        COALESCE(AVG(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90), 0) AS avg_ticket_90,
        COALESCE(STDDEV_POP(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90), 0) AS amt_std_90,
        COUNT(*) FILTER (WHERE status = 'declined' AND d >= (SELECT asof FROM params) - 90) AS declined_90,
        COUNT(*) FILTER (WHERE status = 'declined' AND d >= (SELECT asof FROM params) - 30) AS declined_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'purchase' AND d >= (SELECT asof FROM params) - 90) AS purchase_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'salary_in' AND d >= (SELECT asof FROM params) - 90) AS salary_90,
        MAX(d) FILTER (WHERE status = 'approved' AND txn_type = 'salary_in') AS last_salary_date,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'cash_out' AND d >= (SELECT asof FROM params) - 90) AS cashout_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND txn_type = 'cash_out' AND d >= (SELECT asof FROM params) - 90), 0) AS cashout_amt_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_out' AND d >= (SELECT asof FROM params) - 90) AS p2p_out_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_out' AND d >= (SELECT asof FROM params) - 90), 0) AS p2p_out_amt_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_in' AND d >= (SELECT asof FROM params) - 90), 0) AS p2p_in_amt_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND channel = 'app' AND d >= (SELECT asof FROM params) - 90) AS app_txn_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND channel = 'pos' AND d >= (SELECT asof FROM params) - 90) AS pos_txn_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND channel = 'atm' AND d >= (SELECT asof FROM params) - 90) AS atm_txn_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND channel = 'ecom' AND d >= (SELECT asof FROM params) - 90) AS ecom_txn_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND mcc_group = 'grocery' AND d >= (SELECT asof FROM params) - 30) AS grocery_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND mcc_group = 'grocery' AND d >= (SELECT asof FROM params) - 90) AS grocery_90,
        COUNT(DISTINCT mcc_group) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30) AS n_mcc_30,
        COUNT(DISTINCT mcc_group) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS n_mcc_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND EXTRACT(HOUR FROM ts) NOT BETWEEN 7 AND 22) AS night_txn_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND EXTRACT(DOW FROM ts) IN (0, 6)) AS weekend_txn_90,
        COUNT(DISTINCT card_id) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS cards_used_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30 AND amount_uzs < 100000) AS micro_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90 AND amount_uzs < 100000) AS micro_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30
                           AND mcc_group IN ('entertainment', 'restaurants', 'clothing')) AS fun_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND mcc_group IN ('entertainment', 'restaurants', 'clothing')) AS fun_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30
                           AND mcc_group IN ('utilities', 'telecom')) AS bills_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND mcc_group IN ('utilities', 'telecom')) AS bills_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90 AND mcc_group = 'fuel') AS fuel_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30 AND mcc_group = 'fuel') AS fuel_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_out' AND d >= (SELECT asof FROM params) - 90
                           AND amount_uzs % 100000 = 0) AS round_p2p_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND txn_type IN ('salary_in', 'p2p_in')), 0) AS inflow_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND txn_type IN ('purchase', 'cash_out', 'p2p_out', 'utility_payment')), 0) AS outflow_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND EXTRACT(DOW FROM ts) = 5) AS friday_90,
        COUNT(DISTINCT EXTRACT(HOUR FROM ts)::int) FILTER (
            WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS n_hours_90,
        COALESCE(STDDEV_POP(EXTRACT(HOUR FROM ts)) FILTER (
            WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90), 0) AS hour_std_90,
        MAX(d) FILTER (WHERE status = 'approved' AND mcc_group = 'grocery') AS last_grocery_date,
        MAX(d) FILTER (WHERE status = 'approved' AND channel = 'pos') AS last_pos_date,
        COUNT(*) FILTER (WHERE status = 'declined' AND d >= (SELECT asof FROM params) - 14) AS declined_14,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 180
                           AND d < (SELECT asof FROM params) - 90) AS txn_cnt_91_180,
        COUNT(DISTINCT d) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 7) AS active_days_7,
        COUNT(DISTINCT d) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30) AS active_days_30,
        COUNT(DISTINCT d) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS active_days_90,
        COUNT(DISTINCT DATE_TRUNC('week', d)) FILTER (
            WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 84) AS weeks_active_12,
        COALESCE(MAX(amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90), 0) AS max_amt_90,
        COUNT(DISTINCT amount_uzs) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90) AS n_distinct_amt_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND channel = 'pos' AND d >= (SELECT asof FROM params) - 30) AS pos_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30
                           AND EXTRACT(DOW FROM ts) IN (0, 6)) AS weekend_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_in' AND d >= (SELECT asof FROM params) - 30) AS p2p_in_30,
        COUNT(*) FILTER (WHERE status = 'approved' AND txn_type = 'p2p_in' AND d >= (SELECT asof FROM params) - 90) AS p2p_in_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND EXTRACT(DAY FROM d) <= 5) AS txn_month_start_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE status = 'declined' AND d >= (SELECT asof FROM params) - 90), 0) AS declined_amt_90,
        COUNT(*) FILTER (WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 90
                           AND txn_type = 'mobile_topup') AS topup_90
    FROM txn
    GROUP BY 1
),
last_hit AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        txn_type AS last_txn_type,
        channel AS last_channel,
        mcc_group AS last_mcc,
        status AS last_status,
        amount_uzs AS last_amount
    FROM txn
    ORDER BY customer_id, ts DESC
),
fav_mcc AS (
    SELECT customer_id, mcc_group AS fav_mcc
    FROM (
        SELECT customer_id, mcc_group, COUNT(*) AS n,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY COUNT(*) DESC) AS rn
        FROM txn
        WHERE status = 'approved'
          AND d >= (SELECT asof FROM params) - 180
          AND d < (SELECT asof FROM params) - 30
        GROUP BY 1, 2
    ) s
    WHERE rn = 1
),
fav_now AS (
    SELECT t.customer_id, COUNT(*) AS fav_mcc_30
    FROM txn t
    JOIN fav_mcc f ON f.customer_id = t.customer_id AND f.fav_mcc = t.mcc_group
    WHERE t.status = 'approved'
      AND t.d >= (SELECT asof FROM params) - 30
    GROUP BY 1
),
camp_hist AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE contacted_at < (SELECT asof FROM params)) AS camps_before,
        MAX(contacted_at) FILTER (WHERE contacted_at < (SELECT asof FROM params)) AS last_camp_date,
        MAX(responded) FILTER (WHERE contacted_at < (SELECT asof FROM params)) AS ever_responded
    FROM churn_retention_campaigns
    GROUP BY 1
),
daily AS (
    SELECT DISTINCT customer_id, d
    FROM txn
    WHERE status = 'approved'
      AND d >= (SELECT asof FROM params) - 365
),
gap_raw AS (
    SELECT
        customer_id,
        d,
        d - LAG(d) OVER (PARTITION BY customer_id ORDER BY d) AS gap_days
    FROM daily
),
rhythm AS (
    SELECT
        customer_id,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) AS median_gap,
        AVG(gap_days) AS mean_gap,
        STDDEV_POP(gap_days) AS std_gap,
        MAX(gap_days) AS max_gap_hist,
        COUNT(*) FILTER (WHERE gap_days >= 30) AS n_comebacks_30,
        COUNT(*) FILTER (WHERE gap_days >= 60) AS n_comebacks_60
    FROM gap_raw
    WHERE gap_days IS NOT NULL
    GROUP BY 1
),
last_gaps AS (
    SELECT
        customer_id,
        (ARRAY_AGG(gap_days ORDER BY d DESC))[1] AS gap_last,
        (ARRAY_AGG(gap_days ORDER BY d DESC))[2] AS gap_prev,
        (ARRAY_AGG(gap_days ORDER BY d DESC))[3] AS gap_prev2,
        (ARRAY_AGG(gap_days ORDER BY d DESC))[4] AS gap_prev3
    FROM gap_raw
    WHERE gap_days IS NOT NULL
    GROUP BY 1
),
gap_seq AS (
    SELECT
        customer_id,
        d AS returned_on,
        gap_days,
        LAG(gap_days) OVER (PARTITION BY customer_id ORDER BY d) AS prev_gap
    FROM gap_raw
    WHERE gap_days IS NOT NULL
),
yoyo AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE gap_days BETWEEN 14 AND 28) AS n_gap_20,
        COUNT(*) FILTER (WHERE gap_days BETWEEN 45 AND 80) AS n_gap_60,
        COUNT(*) FILTER (WHERE gap_days >= 14) AS n_medium_plus_gaps,
        MAX(returned_on) FILTER (WHERE gap_days BETWEEN 14 AND 28) AS last_20_comeback,
        MAX(returned_on) FILTER (WHERE gap_days BETWEEN 45 AND 80) AS last_60_comeback,
        MAX(returned_on) FILTER (WHERE gap_days >= 45) AS last_long_comeback,
        COUNT(*) FILTER (
            WHERE prev_gap BETWEEN 14 AND 28 AND gap_days BETWEEN 45 AND 80
        ) AS n_seq_20_then_60,
        MAX(CASE WHEN prev_gap BETWEEN 14 AND 28 AND gap_days BETWEEN 45 AND 80
                 THEN 1 ELSE 0 END) AS had_20_then_60
    FROM gap_seq
    GROUP BY 1
),
ghost_days AS (
    SELECT
        a.customer_id,
        COUNT(*) AS login_days_30,
        COUNT(*) FILTER (WHERE tx.d IS NULL) AS login_no_pay_days
    FROM (
        SELECT DISTINCT customer_id, event_ts::date AS d
        FROM churn_app_events
        WHERE event_ts::date < (SELECT asof FROM params)
          AND event_ts::date >= (SELECT asof FROM params) - 30
          AND event_type = 'login'
    ) a
    LEFT JOIN (
        SELECT DISTINCT customer_id, d
        FROM txn
        WHERE status = 'approved' AND d >= (SELECT asof FROM params) - 30
    ) tx ON tx.customer_id = a.customer_id AND tx.d = a.d
    GROUP BY 1
),
comeback_q AS (
    SELECT
        y.customer_id,
        COUNT(*) FILTER (
            WHERE t.status = 'approved'
              AND t.d > y.last_long_comeback
              AND t.d <= y.last_long_comeback + 21
        ) AS txn_21_after_hole,
        COUNT(*) FILTER (
            WHERE t.status = 'approved'
              AND t.d >= y.last_long_comeback - 81
              AND t.d < y.last_long_comeback - 45
        ) AS txn_21_before_hole
    FROM yoyo y
    LEFT JOIN txn t ON t.customer_id = y.customer_id
    WHERE y.last_long_comeback IS NOT NULL
    GROUP BY 1
),
future AS (
    SELECT
        t.customer_id,
        COUNT(*) FILTER (
            WHERE t.status = 'approved'
              AND t.txn_ts::date > (SELECT snap FROM params) + :lag
              AND t.txn_ts::date <= (SELECT snap FROM params) + :lag + :horizon
        ) AS txn_next_h
    FROM churn_transactions t
    GROUP BY 1
),
cards AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE issue_date <= (SELECT asof FROM params)) AS cards_issued,
        COUNT(*) FILTER (WHERE status = 'active' AND issue_date <= (SELECT asof FROM params)
                           AND (expiry_date IS NULL OR expiry_date >= (SELECT asof FROM params))) AS cards_active,
        COUNT(*) FILTER (WHERE status IN ('blocked', 'closed') AND issue_date <= (SELECT asof FROM params)) AS cards_dead,
        COUNT(*) FILTER (WHERE card_type = 'credit' AND issue_date <= (SELECT asof FROM params)) AS cards_credit,
        MIN(expiry_date - (SELECT asof FROM params)) FILTER (
            WHERE status = 'active' AND expiry_date IS NOT NULL AND issue_date <= (SELECT asof FROM params)
        ) AS days_to_expiry,
        COUNT(*) FILTER (
            WHERE status = 'blocked' AND issue_date <= (SELECT asof FROM params)
        ) AS cards_blocked,
        MAX(issue_date) FILTER (WHERE issue_date <= (SELECT asof FROM params)) AS newest_card_date,
        COUNT(*) FILTER (WHERE card_type = 'credit' AND status = 'active'
                           AND issue_date <= (SELECT asof FROM params)) AS credit_active
    FROM churn_cards
    GROUP BY 1
),
prod AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (
            WHERE opened_at <= (SELECT asof FROM params)
              AND (closed_at IS NULL OR closed_at > (SELECT asof FROM params))
              AND product_code <> 'card_basic'
        ) AS products_open,
        COUNT(*) FILTER (
            WHERE closed_at IS NOT NULL
              AND closed_at < (SELECT asof FROM params)
              AND closed_at >= (SELECT asof FROM params) - 90
        ) AS products_closed_90
    FROM churn_products
    GROUP BY 1
),
pe AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'deposit_withdrawal') AS deposit_wd_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'deposit_topup') AS deposit_top_90,
        COALESCE(SUM(amount_uzs) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'deposit_withdrawal'), 0)
          - COALESCE(SUM(amount_uzs) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'deposit_topup'), 0) AS deposit_net_out_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 30
                           AND event_type = 'cashback_payout') AS cashback_30,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_date < (SELECT asof FROM params) - 30
                           AND event_type = 'cashback_payout') AS cashback_31_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 30
                           AND event_type = 'autopay_charge') AS autopay_30,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_date < (SELECT asof FROM params) - 30
                           AND event_type = 'autopay_charge') AS autopay_31_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'transfer_out') AS abroad_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'loan_disbursement') AS loan_disb_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'loan_payment') AS loan_pay_90,
        COUNT(*) FILTER (WHERE event_date < (SELECT asof FROM params)
                           AND event_date >= (SELECT asof FROM params) - 90
                           AND event_type = 'premium_charge') AS insurance_90
    FROM churn_product_events
    GROUP BY 1
),
tickets AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND opened_at::date >= (SELECT asof FROM params) - 90) AS tickets_90,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND opened_at::date >= (SELECT asof FROM params) - 90
                           AND topic IN ('fee_complaint', 'card_blocked', 'failed_transfer', 'app_issue', 'service_quality')) AS tickets_bad_90,
        AVG(satisfaction) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                                    AND opened_at::date >= (SELECT asof FROM params) - 180) AS sat_180,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND opened_at::date >= (SELECT asof FROM params) - 90
                           AND topic = 'fee_complaint') AS tickets_fee_90,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND opened_at::date >= (SELECT asof FROM params) - 90
                           AND topic = 'card_blocked') AS tickets_block_90,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND opened_at::date >= (SELECT asof FROM params) - 90
                           AND topic = 'how_to') AS tickets_howto_90,
        COUNT(*) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                           AND (resolved_at IS NULL OR resolved_at::date >= (SELECT asof FROM params))) AS tickets_open,
        MIN(satisfaction) FILTER (WHERE opened_at::date < (SELECT asof FROM params)
                                    AND opened_at::date >= (SELECT asof FROM params) - 180) AS sat_min_180,
        AVG(EXTRACT(EPOCH FROM (resolved_at - opened_at)) / 3600.0) FILTER (
            WHERE opened_at::date < (SELECT asof FROM params)
              AND resolved_at IS NOT NULL
              AND opened_at::date >= (SELECT asof FROM params) - 180
        ) AS ticket_hours
    FROM churn_support_tickets
    GROUP BY 1
),
subs AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        monthly_price_uzs AS sub_price,
        status AS sub_status,
        CASE WHEN status = 'active' AND started_at <= (SELECT asof FROM params)
                  AND (canceled_at IS NULL OR canceled_at > (SELECT asof FROM params))
             THEN 1 ELSE 0 END AS has_active_sub
    FROM churn_subscriptions
    WHERE started_at <= (SELECT asof FROM params)
    ORDER BY customer_id, started_at DESC
),
fail_chg AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE charged_at < (SELECT asof FROM params)
                           AND charged_at >= (SELECT asof FROM params) - 90
                           AND status = 'failed') AS failed_charges_90,
        COUNT(*) FILTER (WHERE charged_at < (SELECT asof FROM params)
                           AND charged_at >= (SELECT asof FROM params) - 90
                           AND status = 'success') AS ok_charges_90,
        COUNT(*) FILTER (WHERE charged_at < (SELECT asof FROM params)
                           AND charged_at >= (SELECT asof FROM params) - 90
                           AND status = 'failed' AND fail_reason = 'insufficient_funds') AS nsf_fails_90
    FROM churn_subscription_charges
    GROUP BY 1
),
sub_hist AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (
            WHERE canceled_at IS NOT NULL
              AND canceled_at < (SELECT asof FROM params)
              AND canceled_at >= (SELECT asof FROM params) - 90
        ) AS sub_cancel_90,
        COUNT(*) FILTER (
            WHERE canceled_at IS NOT NULL
              AND canceled_at < (SELECT asof FROM params)
              AND canceled_at >= (SELECT asof FROM params) - 90
              AND cancel_reason IN ('fee_increase', 'too_expensive')
        ) AS sub_cancel_fee_90
    FROM churn_subscriptions
    GROUP BY 1
),
tariff AS (
    SELECT
        customer_id,
        new_monthly_fee_uzs AS card_fee,
        old_monthly_fee_uzs AS card_fee_old,
        CASE WHEN effective_date <= (SELECT asof FROM params) THEN 1 ELSE 0 END AS tariff_already_on
    FROM churn_tariff_changes
),
app AS (
    SELECT
        customer_id,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30) AS app_events_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'support_chat') AS app_support_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 14
                           AND event_type = 'login') AS app_login_14,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 14) AS app_events_14,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'open_offers') AS app_offers_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'check_balance') AS app_balance_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'settings') AS app_settings_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'push_open') AS app_push_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'p2p_form') AS app_p2p_form_30,
        COUNT(*) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                           AND event_ts::date >= (SELECT asof FROM params) - 30
                           AND event_type = 'open_card') AS app_open_card_30,
        MAX(event_ts::date) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                                      AND event_type = 'login') AS last_login_date,
        COUNT(DISTINCT platform) FILTER (WHERE event_ts::date < (SELECT asof FROM params)
                                           AND event_ts::date >= (SELECT asof FROM params) - 90) AS n_platforms_90
    FROM churn_app_events
    GROUP BY 1
)
SELECT
    pop.customer_id,
    pop.snap,
    pop.signup_date,
    pop.gender,
    pop.region,
    pop.acq_channel,
    pop.segment,
    pop.is_payroll,
    pop.declared_income_uzs,
    pop.age_years,
    pop.days_to_bday,
    pop.tenure_days,
    ta.last_txn_date,
    (pop.snap - ta.last_txn_date) AS recency_days,
    COALESCE(ta.txn_cnt_14, 0) AS txn_cnt_14,
    COALESCE(ta.txn_cnt_30, 0) AS txn_cnt_30,
    COALESCE(ta.txn_cnt_60, 0) AS txn_cnt_60,
    COALESCE(ta.txn_cnt_90, 0) AS txn_cnt_90,
    COALESCE(ta.txn_cnt_180, 0) AS txn_cnt_180,
    COALESCE(ta.gmv_30, 0) AS gmv_30,
    COALESCE(ta.gmv_90, 0) AS gmv_90,
    COALESCE(ta.declined_90, 0) AS declined_90,
    CASE WHEN COALESCE(ta.txn_cnt_90, 0) + COALESCE(ta.declined_90, 0) > 0
         THEN COALESCE(ta.declined_90, 0)::float / (ta.txn_cnt_90 + ta.declined_90)
         ELSE 0 END AS decline_rate_90,
    COALESCE(ta.purchase_90, 0) AS purchase_90,
    COALESCE(ta.salary_90, 0) AS salary_90,
    COALESCE(ta.cashout_90, 0) AS cashout_90,
    COALESCE(ta.p2p_out_90, 0) AS p2p_out_90,
    COALESCE(ta.app_txn_90, 0) AS app_txn_90,
    COALESCE(ta.avg_ticket_30, 0) AS avg_ticket_30,
    COALESCE(ta.avg_ticket_90, 0) AS avg_ticket_90,
    COALESCE(ta.amt_std_90, 0) AS amt_std_90,
    COALESCE(ta.declined_30, 0) AS declined_30,
    ta.last_salary_date,
    COALESCE(ta.cashout_amt_90, 0) AS cashout_amt_90,
    COALESCE(ta.p2p_out_amt_90, 0) AS p2p_out_amt_90,
    COALESCE(ta.p2p_in_amt_90, 0) AS p2p_in_amt_90,
    COALESCE(ta.pos_txn_90, 0) AS pos_txn_90,
    COALESCE(ta.atm_txn_90, 0) AS atm_txn_90,
    COALESCE(ta.ecom_txn_90, 0) AS ecom_txn_90,
    COALESCE(ta.grocery_30, 0) AS grocery_30,
    COALESCE(ta.grocery_90, 0) AS grocery_90,
    COALESCE(ta.n_mcc_30, 0) AS n_mcc_30,
    COALESCE(ta.n_mcc_90, 0) AS n_mcc_90,
    COALESCE(ta.night_txn_90, 0) AS night_txn_90,
    COALESCE(ta.weekend_txn_90, 0) AS weekend_txn_90,
    COALESCE(ta.cards_used_90, 0) AS cards_used_90,
    rh.median_gap,
    rh.mean_gap,
    rh.std_gap,
    rh.max_gap_hist,
    COALESCE(rh.n_comebacks_30, 0) AS n_comebacks_30,
    COALESCE(rh.n_comebacks_60, 0) AS n_comebacks_60,
    COALESCE(ta.micro_30, 0) AS micro_30,
    COALESCE(ta.micro_90, 0) AS micro_90,
    COALESCE(ta.fun_30, 0) AS fun_30,
    COALESCE(ta.fun_90, 0) AS fun_90,
    COALESCE(ta.bills_30, 0) AS bills_30,
    COALESCE(ta.bills_90, 0) AS bills_90,
    COALESCE(ta.fuel_30, 0) AS fuel_30,
    COALESCE(ta.fuel_90, 0) AS fuel_90,
    COALESCE(ta.round_p2p_90, 0) AS round_p2p_90,
    COALESCE(ta.inflow_90, 0) AS inflow_90,
    COALESCE(ta.outflow_90, 0) AS outflow_90,
    COALESCE(ta.friday_90, 0) AS friday_90,
    COALESCE(ta.n_hours_90, 0) AS n_hours_90,
    COALESCE(ta.hour_std_90, 0) AS hour_std_90,
    ta.last_grocery_date,
    ta.last_pos_date,
    COALESCE(ta.declined_14, 0) AS declined_14,
    COALESCE(ta.txn_cnt_91_180, 0) AS txn_cnt_91_180,
    lh.last_txn_type,
    lh.last_channel,
    lh.last_mcc,
    lh.last_status,
    lh.last_amount,
    f.fav_mcc,
    COALESCE(fn.fav_mcc_30, 0) AS fav_mcc_30,
    lg.gap_last,
    lg.gap_prev,
    lg.gap_prev2,
    lg.gap_prev3,
    COALESCE(yo.n_gap_20, 0) AS n_gap_20,
    COALESCE(yo.n_gap_60, 0) AS n_gap_60,
    COALESCE(yo.n_medium_plus_gaps, 0) AS n_medium_plus_gaps,
    yo.last_20_comeback,
    yo.last_60_comeback,
    yo.last_long_comeback,
    COALESCE(yo.n_seq_20_then_60, 0) AS n_seq_20_then_60,
    COALESCE(yo.had_20_then_60, 0) AS had_20_then_60,
    COALESCE(ta.active_days_7, 0) AS active_days_7,
    COALESCE(ta.active_days_30, 0) AS active_days_30,
    COALESCE(ta.active_days_90, 0) AS active_days_90,
    COALESCE(ta.weeks_active_12, 0) AS weeks_active_12,
    COALESCE(ta.max_amt_90, 0) AS max_amt_90,
    COALESCE(ta.n_distinct_amt_90, 0) AS n_distinct_amt_90,
    COALESCE(ta.pos_30, 0) AS pos_30,
    COALESCE(ta.weekend_30, 0) AS weekend_30,
    COALESCE(ta.p2p_in_30, 0) AS p2p_in_30,
    COALESCE(ta.p2p_in_90, 0) AS p2p_in_90,
    COALESCE(ta.txn_month_start_90, 0) AS txn_month_start_90,
    COALESCE(ta.declined_amt_90, 0) AS declined_amt_90,
    COALESCE(ta.topup_90, 0) AS topup_90,
    COALESCE(gd.login_days_30, 0) AS login_days_30,
    COALESCE(gd.login_no_pay_days, 0) AS login_no_pay_days,
    COALESCE(cq.txn_21_after_hole, 0) AS txn_21_after_hole,
    COALESCE(cq.txn_21_before_hole, 0) AS txn_21_before_hole,
    COALESCE(ch.camps_before, 0) AS camps_before,
    ch.last_camp_date,
    COALESCE(ch.ever_responded, 0) AS ever_responded,
    COALESCE(cd.cards_issued, 0) AS cards_issued,
    COALESCE(cd.cards_active, 0) AS cards_active,
    COALESCE(cd.cards_dead, 0) AS cards_dead,
    COALESCE(cd.cards_credit, 0) AS cards_credit,
    cd.days_to_expiry,
    COALESCE(cd.cards_blocked, 0) AS cards_blocked,
    cd.newest_card_date,
    COALESCE(cd.credit_active, 0) AS credit_active,
    COALESCE(pr.products_open, 0) AS products_open,
    COALESCE(pr.products_closed_90, 0) AS products_closed_90,
    COALESCE(pe.deposit_wd_90, 0) AS deposit_wd_90,
    COALESCE(pe.deposit_top_90, 0) AS deposit_top_90,
    COALESCE(pe.deposit_net_out_90, 0) AS deposit_net_out_90,
    COALESCE(pe.cashback_30, 0) AS cashback_30,
    COALESCE(pe.cashback_31_90, 0) AS cashback_31_90,
    COALESCE(pe.autopay_30, 0) AS autopay_30,
    COALESCE(pe.autopay_31_90, 0) AS autopay_31_90,
    COALESCE(pe.abroad_90, 0) AS abroad_90,
    COALESCE(pe.loan_disb_90, 0) AS loan_disb_90,
    COALESCE(pe.loan_pay_90, 0) AS loan_pay_90,
    COALESCE(pe.insurance_90, 0) AS insurance_90,
    COALESCE(tk.tickets_90, 0) AS tickets_90,
    COALESCE(tk.tickets_bad_90, 0) AS tickets_bad_90,
    tk.sat_180,
    COALESCE(tk.tickets_fee_90, 0) AS tickets_fee_90,
    COALESCE(tk.tickets_block_90, 0) AS tickets_block_90,
    COALESCE(tk.tickets_howto_90, 0) AS tickets_howto_90,
    COALESCE(tk.tickets_open, 0) AS tickets_open,
    tk.sat_min_180,
    tk.ticket_hours,
    COALESCE(sb.has_active_sub, 0) AS has_active_sub,
    COALESCE(sb.sub_price, 0) AS sub_price,
    COALESCE(fc.failed_charges_90, 0) AS failed_charges_90,
    COALESCE(fc.ok_charges_90, 0) AS ok_charges_90,
    COALESCE(fc.nsf_fails_90, 0) AS nsf_fails_90,
    COALESCE(sh.sub_cancel_90, 0) AS sub_cancel_90,
    COALESCE(sh.sub_cancel_fee_90, 0) AS sub_cancel_fee_90,
    COALESCE(tf.card_fee_old, 8139) AS card_fee_old,
    COALESCE(tf.card_fee, 23793) AS card_fee_new,
    COALESCE(tf.tariff_already_on, 0) AS tariff_already_on,
    COALESCE(ap.app_events_30, 0) AS app_events_30,
    COALESCE(ap.app_support_30, 0) AS app_support_30,
    COALESCE(ap.app_login_14, 0) AS app_login_14,
    COALESCE(ap.app_events_14, 0) AS app_events_14,
    COALESCE(ap.app_offers_30, 0) AS app_offers_30,
    COALESCE(ap.app_balance_30, 0) AS app_balance_30,
    COALESCE(ap.app_settings_30, 0) AS app_settings_30,
    COALESCE(ap.app_push_30, 0) AS app_push_30,
    COALESCE(ap.app_p2p_form_30, 0) AS app_p2p_form_30,
    COALESCE(ap.app_open_card_30, 0) AS app_open_card_30,
    ap.last_login_date,
    COALESCE(ap.n_platforms_90, 0) AS n_platforms_90,
    COALESCE(fu.txn_next_h, 0) AS txn_next_h,
    CASE WHEN COALESCE(fu.txn_next_h, 0) = 0 THEN 1 ELSE 0 END AS target
FROM pop
JOIN txn_agg ta ON ta.customer_id = pop.customer_id
LEFT JOIN rhythm rh ON rh.customer_id = pop.customer_id
LEFT JOIN last_gaps lg ON lg.customer_id = pop.customer_id
LEFT JOIN yoyo yo ON yo.customer_id = pop.customer_id
LEFT JOIN ghost_days gd ON gd.customer_id = pop.customer_id
LEFT JOIN comeback_q cq ON cq.customer_id = pop.customer_id
LEFT JOIN last_hit lh ON lh.customer_id = pop.customer_id
LEFT JOIN fav_mcc f ON f.customer_id = pop.customer_id
LEFT JOIN fav_now fn ON fn.customer_id = pop.customer_id
LEFT JOIN camp_hist ch ON ch.customer_id = pop.customer_id
LEFT JOIN future fu ON fu.customer_id = pop.customer_id
LEFT JOIN cards cd ON cd.customer_id = pop.customer_id
LEFT JOIN prod pr ON pr.customer_id = pop.customer_id
LEFT JOIN pe ON pe.customer_id = pop.customer_id
LEFT JOIN tickets tk ON tk.customer_id = pop.customer_id
LEFT JOIN subs sb ON sb.customer_id = pop.customer_id
LEFT JOIN sub_hist sh ON sh.customer_id = pop.customer_id
LEFT JOIN fail_chg fc ON fc.customer_id = pop.customer_id
LEFT JOIN tariff tf ON tf.customer_id = pop.customer_id
LEFT JOIN app ap ON ap.customer_id = pop.customer_id
WHERE ta.last_txn_date >= pop.snap - :lookback
"""


UPLIFT_SQL = """
WITH months AS (
    SELECT generate_series(DATE '2024-06-01', DATE '2025-09-01', interval '1 month')::date AS snap
),
pop AS (
    SELECT
        m.snap,
        c.customer_id,
        c.segment,
        MAX(t.txn_ts::date) FILTER (WHERE t.txn_ts::date < m.snap) AS last_before,
        BOOL_OR(t.status = 'approved' AND t.txn_ts::date > m.snap AND t.txn_ts::date <= m.snap + 90) AS returned_90,
        COALESCE(SUM(t.amount_uzs) FILTER (
            WHERE t.status = 'approved' AND t.txn_ts::date > m.snap AND t.txn_ts::date <= m.snap + 90
        ), 0) AS gmv_next_90
    FROM months m
    JOIN churn_customers c ON c.signup_date <= m.snap - 60
    LEFT JOIN churn_transactions t
      ON t.customer_id = c.customer_id
    GROUP BY 1, 2, 3
),
camp AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', contacted_at)::date AS camp_month,
        contacted_at,
        campaign_code,
        channel,
        responded
    FROM churn_retention_campaigns
)
SELECT
    p.snap,
    p.customer_id,
    p.segment,
    (p.snap - p.last_before) AS recency_days,
    CASE WHEN camp.customer_id IS NOT NULL THEN 1 ELSE 0 END AS treated,
    COALESCE(camp.responded, 0) AS responded,
    camp.campaign_code,
    camp.channel,
    p.returned_90::int AS returned_90,
    p.gmv_next_90
FROM pop p
LEFT JOIN camp
  ON camp.customer_id = p.customer_id
 AND camp.camp_month = p.snap
WHERE p.last_before IS NOT NULL
  AND (p.snap - p.last_before) BETWEEN 30 AND 90
"""


MONTHLY_CHURN_SQL = """
WITH months AS (
    SELECT generate_series(DATE '2024-04-01', DATE '2025-09-01', interval '1 month')::date AS snap
),
base AS (
    SELECT
        m.snap,
        c.customer_id,
        MAX(t.txn_ts::date) FILTER (WHERE t.status = 'approved' AND t.txn_ts::date < m.snap) AS last_before,
        BOOL_OR(t.status = 'approved' AND t.txn_ts::date > m.snap AND t.txn_ts::date <= m.snap + 90) AS active_next90
    FROM months m
    JOIN churn_customers c ON c.signup_date <= m.snap - 60
    LEFT JOIN churn_transactions t ON t.customer_id = c.customer_id
    GROUP BY 1, 2
)
SELECT
    snap,
    COUNT(*) FILTER (WHERE last_before >= snap - 90) AS n_at_risk,
    COUNT(*) FILTER (WHERE last_before >= snap - 90 AND NOT active_next90) AS n_churn,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE last_before >= snap - 90 AND NOT active_next90)
        / NULLIF(COUNT(*) FILTER (WHERE last_before >= snap - 90), 0),
        2
    ) AS churn90_pct
FROM base
GROUP BY 1
ORDER BY 1
"""


def _safe_div(a, b):
    return a / b.where(b != 0)


def add_value_cols(df: pd.DataFrame) -> pd.DataFrame:
    fee = df["card_fee_new"].where(df["tariff_already_on"] == 1, df["card_fee_old"])
    df["monthly_margin_uzs"] = fee + df["has_active_sub"] * df["sub_price"]
    df["value_retain_90d_uzs"] = df["monthly_margin_uzs"] * 3
    df["expected_loss_uzs"] = df["target"] * df["value_retain_90d_uzs"]
    return add_signature_features(df)


def add_signature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Именные фичи — не RFM ради RFM, а гипотеза про поведение."""
    rec = df["recency_days"].astype(float)
    txn90 = df["txn_cnt_90"].astype(float)
    gmv90 = df["gmv_90"].astype(float)

    # 1. Слом собственного ритма: тишина / обычный gap этого клиента
    df["rhythm_break"] = _safe_div(rec, df["median_gap"].astype(float)).fillna(0).clip(0, 30)
    # 2. Рваный ритм (коэф. вариации пауз)
    df["gap_cv"] = _safe_div(df["std_gap"].astype(float), df["mean_gap"].astype(float)).fillna(0).clip(0, 10)
    # 3. Текущая пауза — рекорд за год?
    df["pause_is_record"] = (
        (df["max_gap_hist"].notna()) & (rec > df["max_gap_hist"].astype(float))
    ).astype(int)
    # 4. Затухание активности: 30д vs ожидаемое от 90д
    df["activity_fade"] = (1 - _safe_div(df["txn_cnt_30"] * 3.0, txn90)).fillna(0).clip(-2, 1)
    df["gmv_fade"] = (1 - _safe_div(df["gmv_30"] * 3.0, gmv90)).fillna(0).clip(-2, 1)
    # 5. Карта перестала быть «продуктовой» — ломается привычка
    df["grocery_fade"] = (1 - _safe_div(df["grocery_30"] * 3.0, df["grocery_90"].astype(float))).fillna(0).clip(-2, 1)
    # 6. Жизнь сжалась: меньше категорий MCC
    df["life_shrink"] = (1 - _safe_div(df["n_mcc_30"].astype(float), df["n_mcc_90"].astype(float))).fillna(0).clip(0, 1)
    # 7. Зарплатник без зарплаты — карта больше не зарплатная
    salary_rec = (pd.to_datetime(df["snap"]) - pd.to_datetime(df["last_salary_date"])).dt.days
    df["salary_recency"] = salary_rec.fillna(999).clip(0, 999)
    df["missed_payroll"] = ((df["is_payroll"] == 1) & (df["salary_90"] == 0)).astype(int)
    # 8. Слив: кэш-аут + исходящий p2p
    df["drain_share"] = _safe_div(df["cashout_amt_90"] + df["p2p_out_amt_90"], gmv90).fillna(0).clip(0, 5)
    df["p2p_net_out"] = df["p2p_out_amt_90"] - df["p2p_in_amt_90"]
    # 9. Цифровой призрак: логинится, не платит
    df["digital_ghost"] = ((df["app_login_14"] > 0) & (df["txn_cnt_14"] == 0)).astype(int)
    df["login_to_txn"] = _safe_div(df["app_events_30"].astype(float), df["txn_cnt_30"].astype(float) + 1).fillna(0).clip(0, 50)
    # 10. Автоплатёж оборвался / кэшбэк пропал
    df["autopay_broken"] = ((df["autopay_31_90"] > 0) & (df["autopay_30"] == 0)).astype(int)
    df["cashback_stopped"] = ((df["cashback_31_90"] > 0) & (df["cashback_30"] == 0)).astype(int)
    # 11. Слив депозита, закрытие продуктов
    df["deposit_drain"] = (df["deposit_net_out_90"] > 0).astype(int)
    df["unbundling"] = ((df["products_closed_90"] > 0) | (df["sub_cancel_90"] > 0)).astype(int)
    # 12. Боль от тарифа: жалобы + отмена «дорого» + нагрузка комиссии на оборот
    df["fee_pain"] = (
        (df["tickets_fee_90"] > 0).astype(int)
        + (df["sub_cancel_fee_90"] > 0).astype(int)
        + ((df["tariff_already_on"] == 1) & (df["activity_fade"] > 0.3)).astype(int)
    )
    df["fee_burden"] = _safe_div(df["monthly_margin_uzs"], df["gmv_30"].replace(0, pd.NA)).fillna(0).clip(0, 2)
    # 13. Кредит только гасит, не берёт — сворачивает лимит
    df["loan_winddown"] = ((df["loan_pay_90"] > 0) & (df["loan_disb_90"] == 0)).astype(int)
    # 14. Ночные / выходные — смена образа жизни
    df["night_share"] = _safe_div(df["night_txn_90"].astype(float), txn90).fillna(0)
    df["weekend_share"] = _safe_div(df["weekend_txn_90"].astype(float), txn90).fillna(0)
    df["pos_share"] = _safe_div(df["pos_txn_90"].astype(float), txn90).fillna(0)
    # 15. Средний чек сдулся + отказы участились
    df["ticket_shrink"] = (1 - _safe_div(df["avg_ticket_30"], df["avg_ticket_90"].replace(0, pd.NA))).fillna(0).clip(-2, 1)
    df["decline_accel"] = _safe_div(df["declined_30"] * 3.0, (df["declined_90"] + 1).astype(float)).fillna(0)
    df["support_per_txn"] = _safe_div(df["tickets_90"].astype(float) * 100, txn90 + 1).fillna(0)
    df["expiry_soon"] = ((df["days_to_expiry"].notna()) & (df["days_to_expiry"] <= 45) & (df["days_to_expiry"] >= 0)).astype(int)
    df["income_load"] = _safe_div(gmv90, df["declared_income_uzs"].astype(float)).fillna(0).clip(0, 20)

    # 16. Переезд в другой банк: не обрыв, а лестница вниз (месяц -3 → -2 → -1)
    n1 = df["txn_cnt_30"].astype(float)
    n2 = (df["txn_cnt_60"] - df["txn_cnt_30"]).astype(float).clip(lower=0)
    n3 = (df["txn_cnt_90"] - df["txn_cnt_60"]).astype(float).clip(lower=0)
    df["txn_m1"] = n1
    df["txn_m2"] = n2
    df["txn_m3"] = n3
    df["mono_fade"] = ((n3 > n2) & (n2 > n1) & (n3 >= 3)).astype(int)
    df["usage_halved"] = ((n3 >= 4) & (n1 <= n3 * 0.5)).astype(int)
    df["faded_last_month"] = ((n3 > 0) & (n1 == 0)).astype(int)
    # балл «карта перестала быть основной»: затухание + быт + нет резкого слива
    df["bank_switch_score"] = (
        df["mono_fade"]
        + df["usage_halved"]
        + (df["grocery_fade"] > 0.3).astype(int)
        + (df["life_shrink"] > 0.4).astype(int)
        + ((df["activity_fade"] > 0.25) & (df["drain_share"] < 0.4)).astype(int)
    )

    snap = pd.to_datetime(df["snap"])
    # Кофе пропал: мелкие покупки (быт) сдулись
    df["coffee_died"] = (1 - _safe_div(df["micro_30"] * 3.0, df["micro_90"].astype(float))).fillna(0).clip(-2, 1)
    # Развлечения уехали первыми (ресторан/кино/одежда)
    df["fun_died"] = (1 - _safe_div(df["fun_30"] * 3.0, df["fun_90"].astype(float))).fillna(0).clip(-2, 1)
    # Остались только счета — зомби-карта
    bills_share = _safe_div(df["bills_90"].astype(float), txn90).fillna(0)
    df["bills_only_zombie"] = ((bills_share > 0.5) & (df["grocery_90"] <= 1) & (df["fun_90"] <= 1)).astype(int)
    df["bills_share"] = bills_share
    df["fuel_died"] = (1 - _safe_div(df["fuel_30"] * 3.0, df["fuel_90"].astype(float))).fillna(0).clip(-2, 1)
    # Круглые переводы = уводит пачкой в другой банк
    df["round_transfer_share"] = _safe_div(df["round_p2p_90"].astype(float), df["p2p_out_90"].astype(float) + 1).fillna(0)
    df["net_outflow"] = df["outflow_90"] - df["inflow_90"]
    df["became_leaky"] = (df["outflow_90"] > df["inflow_90"] * 1.2).astype(int)
    # Любимая категория брошена
    df["abandoned_favorite"] = ((df["fav_mcc"].notna()) & (df["fav_mcc_30"] == 0)).astype(int)
    # Паузы разгоняются: 5д → 12д → 28д
    gl, gp, gp2 = df["gap_last"].astype(float), df["gap_prev"].astype(float), df["gap_prev2"].astype(float)
    df["gaps_widening"] = ((gl > gp) & (gp > gp2) & gp2.notna()).astype(int)
    # Последний жест — прощание
    df["goodbye_atm"] = (df["last_txn_type"] == "cash_out").astype(int)
    df["goodbye_p2p"] = (df["last_txn_type"] == "p2p_out").astype(int)
    df["last_was_declined"] = (df["last_status"] == "declined").astype(int)
    df["days_without_grocery"] = (snap - pd.to_datetime(df["last_grocery_date"])).dt.days.fillna(999).clip(0, 999)
    df["days_without_pos"] = (snap - pd.to_datetime(df["last_pos_date"])).dt.days.fillna(999).clip(0, 999)
    # Робот vs человек: всегда в один час = автоплатёж, не жизнь
    df["robot_payer"] = ((df["n_hours_90"] <= 2) & (txn90 >= 4)).astype(int)
    df["human_spread"] = df["hour_std_90"].fillna(0)
    df["friday_share"] = _safe_div(df["friday_90"].astype(float), txn90).fillna(0)
    # Приложение: нервничает / роется в настройках / открыл перевод и не послал
    df["balance_anxiety"] = _safe_div(df["app_balance_30"].astype(float), df["txn_cnt_30"].astype(float) + 1).fillna(0).clip(0, 30)
    df["settings_rummage"] = (df["app_settings_30"] >= 3).astype(int)
    df["ghost_push"] = ((df["app_push_30"] > 0) & (df["txn_cnt_14"] == 0)).astype(int)
    df["p2p_abandoned"] = ((df["app_p2p_form_30"] > 0) & (df["p2p_out_90"] == 0)).astype(int)
    df["looks_at_card"] = ((df["app_open_card_30"] > 0) & (df["txn_cnt_14"] == 0)).astype(int)
    login_rec = (snap - pd.to_datetime(df["last_login_date"])).dt.days
    df["app_colder_than_card"] = (login_rec.fillna(999) > rec).astype(int)
    df["card_colder_than_app"] = (rec > login_rec.fillna(rec) + 7).astype(int)
    # Колода карт: много выпустили, пользуется одной / кредитка лежит
    df["idle_cards"] = (df["cards_active"] - df["cards_used_90"]).clip(lower=0)
    df["credit_pillow"] = ((df["credit_active"] > 0) & (df["last_txn_type"] != "purchase")).astype(int)
    newest = (snap - pd.to_datetime(df["newest_card_date"])).dt.days
    df["new_card_then_fade"] = ((newest.fillna(999) <= 60) & (df["activity_fade"] > 0.3)).astype(int)
    # Долгий спад: полгода назад был живее чем этот квартал
    df["long_decline"] = ((df["txn_cnt_91_180"] > 0) & (txn90 < df["txn_cnt_91_180"] * 0.6)).astype(int)
    # Тикеты: не разобрались / висит / злой
    df["confused"] = (df["tickets_howto_90"] > 0).astype(int)
    df["open_ticket"] = (df["tickets_open"] > 0).astype(int)
    df["angry_sat"] = ((df["sat_min_180"].notna()) & (df["sat_min_180"] <= 2)).astype(int)
    # Уже писали — не вернули (исторический fail удержания)
    df["retention_failed"] = (
        (df["camps_before"] > 0)
        & ((snap - pd.to_datetime(df["last_camp_date"])).dt.days.fillna(999) <= 180)
        & (df["activity_fade"] > 0.2)
    ).astype(int)
    df["days_after_tariff"] = 0
    after = snap >= pd.Timestamp("2025-07-01")
    df.loc[after, "days_after_tariff"] = (snap[after] - pd.Timestamp("2025-07-01")).dt.days
    df["post_tariff_mouse"] = (
        after.astype(int) * ((df["txn_cnt_30"] < df["txn_cnt_91_180"] / 3.0) & (df["txn_cnt_91_180"] >= 6)).astype(int)
    )
    df["age_years"] = df["age_years"].fillna(35)

    # Йо-йо: 20 дней тишины → ожил → ~60 дней → ожил → снова затих
    snap = pd.to_datetime(df["snap"])
    rec = df["recency_days"].astype(float)
    df["yoyo"] = ((df["n_gap_20"] >= 1) & (df["n_gap_60"] >= 1)).astype(int)
    df["seq_20_then_60"] = (df["had_20_then_60"] > 0).astype(int)
    days_after_60 = (snap - pd.to_datetime(df["last_60_comeback"])).dt.days
    days_after_long = (snap - pd.to_datetime(df["last_long_comeback"])).dt.days
    df["days_since_60_comeback"] = days_after_60.fillna(999).clip(0, 999)
    df["days_since_long_comeback"] = days_after_long.fillna(999).clip(0, 999)
    # Вернулся после длинной ямы и опять замолкает (твой сценарий «потом исчез»)
    df["vanished_after_comeback"] = (
        (df["n_gap_60"] >= 1)
        & (days_after_60.fillna(999) <= 120)
        & (rec >= 14)
    ).astype(int)
    df["vanished_after_yoyo"] = (
        (df["yoyo"] == 1) & (df["seq_20_then_60"] == 1) & (rec >= 14)
    ).astype(int)

    # Плотность жизни: не «сколько транзакций», а сколько дней вообще жил
    df["life_density_30"] = (df["active_days_30"] / 30.0).clip(0, 1)
    df["life_density_90"] = (df["active_days_90"] / 90.0).clip(0, 1)
    df["empty_weeks"] = (12 - df["weeks_active_12"]).clip(lower=0)
    df["weekend_shift"] = _safe_div(df["weekend_30"].astype(float), df["txn_cnt_30"].astype(float) + 1).fillna(0) - df["weekend_share"]
    df["pos_collapse"] = (1 - _safe_div(df["pos_30"] * 3.0, df["pos_txn_90"].astype(float))).fillna(0).clip(-2, 1)
    df["friends_left"] = ((df["p2p_in_90"] >= 2) & (df["p2p_in_30"] == 0)).astype(int)
    df["only_bills_calendar"] = _safe_div(df["txn_month_start_90"].astype(float), txn90).fillna(0)
    df["whale_day"] = _safe_div(df["max_amt_90"], df["avg_ticket_90"].replace(0, pd.NA)).fillna(0).clip(0, 50)
    df["same_sum_robot"] = (
        (df["n_distinct_amt_90"] <= 3) & (df["txn_cnt_90"] >= 5)
    ).astype(int)
    df["window_shopper"] = _safe_div(df["login_no_pay_days"].astype(float), df["login_days_30"].astype(float) + 1).fillna(0)
    df["comeback_weaker"] = (
        (df["txn_21_before_hole"] >= 3)
        & (df["txn_21_after_hole"] < df["txn_21_before_hole"] * 0.5)
    ).astype(int)
    df["slow_support"] = (df["ticket_hours"].fillna(0) > 48).astype(int)
    df["bday_soon"] = ((df["days_to_bday"].fillna(200) >= 0) & (df["days_to_bday"].fillna(200) <= 21)).astype(int)
    # Белая ворона: тише своих в том же регионе и сегменте
    rec_med = df.groupby(["region", "segment"], dropna=False)["recency_days"].transform("median")
    fade_med = df.groupby(["region", "segment"], dropna=False)["activity_fade"].transform("median")
    rec_std = df.groupby(["region", "segment"], dropna=False)["recency_days"].transform("std").replace(0, pd.NA)
    df["weirder_than_peers"] = _safe_div(df["recency_days"] - rec_med, rec_std).fillna(0).clip(-5, 5)
    df["fading_vs_peers"] = (df["activity_fade"] > fade_med + 0.2).astype(int)
    df["topup_only"] = (
        (df["topup_90"] > 0) & (df["purchase_90"] <= 1) & (df["grocery_90"] == 0)
    ).astype(int)
    return df


FEATURE_DICT = [
    ("rhythm_break", "Слом ритма", "recency / свой медианный gap. >3 = молчит дольше, чем обычно живёт"),
    ("gap_cv", "Рваный ритм", "std/mean пауз за год. Хаотичные паузы ≠ стабильный payroll"),
    ("pause_is_record", "Рекордная пауза", "текущая тишина длиннее любой за год — не «как всегда»"),
    ("n_comebacks_30", "Былые возвраты", "сколько раз уже возвращался после паузы 30+. Есть иммунитет или нет"),
    ("activity_fade", "Затухание", "1 - (txn_30*3/txn_90). Последний месяц тише квартала"),
    ("gmv_fade", "Оборот сдулся", "то же по сумме, не по штукам"),
    ("grocery_fade", "Привычка сломалась", "продукты — якорь карты. Пропали grocery — карта выпала из быта"),
    ("life_shrink", "Жизнь сжалась", "меньше MCC за 30д, чем за 90д. Пользование сузилось"),
    ("missed_payroll", "Зарплата ушла", "is_payroll=1, но salary_in за 90д нет. Карта больше не зарплатная"),
    ("salary_recency", "Дней без ЗП", "как давно последний salary_in"),
    ("drain_share", "Слив", "(cash_out + p2p_out) / GMV. Выводит деньги, не живёт в экосистеме"),
    ("p2p_net_out", "Чистый исходящий p2p", "уходит к другим, не приходит"),
    ("digital_ghost", "Цифровой призрак", "логин в приложении за 14д есть, операций нет"),
    ("login_to_txn", "Смотрит, не платит", "события приложения / операции. Открывает офферы и молчит"),
    ("autopay_broken", "Автоплатёж оборвался", "коммуналка шла 31–90д назад, в последние 30 — тишина"),
    ("cashback_stopped", "Кэшбэк пропал", "выплаты были, потом нет — отвязался от программы"),
    ("deposit_drain", "Слив вклада", "снятия с депозита больше пополнений"),
    ("unbundling", "Расформирование", "закрыл продукт или отменил подписку за 90д"),
    ("fee_pain", "Боль от тарифа", "жалоба на комиссию + отмена «дорого» + затухание после revision"),
    ("fee_burden", "Комиссия/оборот", "наша маржа / GMV за 30д. Тариф съедает смысл карты"),
    ("loan_winddown", "Кредит сворачивает", "платит по линии, новых выдач нет"),
    ("abroad_90", "Переводы за рубеж", "собирается уехать / выводит"),
    ("night_share", "Ночная доля", "сдвиг образа жизни"),
    ("ticket_shrink", "Чек сдулся", "средний чек 30д vs 90д"),
    ("decline_accel", "Отказы ускорились", "declined_30 относительно квартала"),
    ("expiry_soon", "Карта истекает", "до expiry ≤ 45 дней — удобный момент «не перевыпускать»"),
    ("nsf_fails_90", "Не хватает денег", "failed charge = insufficient_funds"),
    ("sub_cancel_fee_90", "Отмена из-за цены", "cancel_reason fee_increase / too_expensive"),
    ("mono_fade", "Лестница вниз", "операций в месяце -3 > -2 > -1. Плавный уход, не обрыв"),
    ("usage_halved", "Половина от квартала", "последний месяц ≤ 50% от месяца -3"),
    ("faded_last_month", "Последний месяц ноль", "ещё жил 90–60 дней назад, в последние 30 тишина"),
    ("bank_switch_score", "Переезд в другой банк", "лестница + grocery/MCC сжались, без слива кэша (не обнал, а смена основной карты)"),
    ("coffee_died", "Кофе пропал", "мелкие покупки <100к сдулись — повседневность ушла"),
    ("fun_died", "Тусовки уехали", "ресторан/кино/одежда падают первыми при смене банка"),
    ("bills_only_zombie", "Зомби-карта", "остались счета, grocery и fun почти ноль"),
    ("fuel_died", "Заправка пропала", "бензин ушёл на другую карту"),
    ("round_transfer_share", "Круглые переводы", "p2p суммами 100/200/500к — уводит пачкой"),
    ("became_leaky", "Дырявый кошелёк", "за 90д больше ушло, чем пришло"),
    ("abandoned_favorite", "Бросил любимое", "топ-MCC за полгода в последний месяц = 0"),
    ("gaps_widening", "Паузы разгоняются", "последние 3 промежутка всё длиннее"),
    ("goodbye_atm", "Прощание в банкомате", "последняя операция — снятие наличных"),
    ("goodbye_p2p", "Прощание переводом", "последнее действие — увёл деньги"),
    ("last_was_declined", "Ушёл после отказа", "последняя попытка отклонена"),
    ("days_without_grocery", "Дней без продуктов", "как давно не было grocery"),
    ("days_without_pos", "Дней без кассы", "как давно не прикладывал в магазине"),
    ("robot_payer", "Робот, не человек", "все операции в 1–2 часа суток — только автоплатёж"),
    ("balance_anxiety", "Проверяет баланс", "check_balance / операции. Нервничает, не тратит"),
    ("settings_rummage", "Роется в настройках", "≥3 settings за 30д — собирается что-то отключить"),
    ("ghost_push", "Пуш открыл, не купил", "push_open есть, операций за 14д нет"),
    ("p2p_abandoned", "Перевод не отправил", "открыл форму p2p, исходящих нет"),
    ("looks_at_card", "Смотрит карту", "open_card в приложении, покупок нет"),
    ("app_colder_than_card", "Приложение бросил раньше", "логин холоднее, чем карта"),
    ("card_colder_than_app", "Карта молчит, приложение живёт", "ещё смотрит, уже не платит"),
    ("idle_cards", "Лишние карты", "активных карт больше, чем использовал"),
    ("new_card_then_fade", "Новая карта и тишина", "выпустили карту ≤60д и активность уже падает"),
    ("long_decline", "Долгий спад", "этот квартал слабее предыдущего"),
    ("confused", "Не понимает продукт", "тикеты how_to"),
    ("open_ticket", "Висящая заявка", "тикет не закрыт на snap"),
    ("angry_sat", "Злой после поддержки", "satisfaction ≤2"),
    ("retention_failed", "Уже писали — не помогло", "кампания была, затухание продолжается"),
    ("post_tariff_mouse", "После тарифа мышь", "после 1.07 стал втрое тише, чем кварталом раньше"),
    ("n_gap_20", "Сколько раз пропадал ~20д", "закрытые паузы 14–28 дней, после которых вернулся"),
    ("n_gap_60", "Сколько раз пропадал ~60д", "закрытые паузы 45–80 дней и всё-таки ожил"),
    ("yoyo", "Йо-йо", "в истории были и короткая (~20), и длинная (~60) яма"),
    ("seq_20_then_60", "Сначала 20, потом 60", "именно в таком порядке: короткая яма, потом длиннее"),
    ("vanished_after_comeback", "Вернулся с 60д и снова затих", "ожил после длинной паузы и опять молчит ≥14д"),
    ("vanished_after_yoyo", "Йо-йо и исчез", "20→60 уже было, сейчас снова тишина"),
    ("life_density_30", "Плотность жизни", "сколько дней из 30 вообще была хоть одна операция"),
    ("empty_weeks", "Пустые недели", "из 12 недель сколько совсем без операций"),
    ("weekend_shift", "Выходные уехали", "доля выходных в последнем месяце vs квартал"),
    ("pos_collapse", "Касса сдулась", "POS за 30д vs ожидание от 90д"),
    ("friends_left", "Переводы перестали приходить", "p2p_in были, в последнем месяце ноль"),
    ("only_bills_calendar", "Только 1–5 число", "живёт в начале месяца — счета, не быт"),
    ("whale_day", "Один кит", "макс чек / средний. Крупный выброс и тишина"),
    ("same_sum_robot", "Одни и те же суммы", "мало уникальных чеков — автоплатежи, не человек"),
    ("window_shopper", "Заходит и не платит", "дни с логином без операции / дни с логином"),
    ("comeback_weaker", "Вернулся бледной тенью", "после ямы 60д активность вдвое слабее, чем до неё"),
    ("slow_support", "Долго решали тикет", "среднее время > 48 часов"),
    ("bday_soon", "Скоро ДР", "день рождения в ближайшие 21 день"),
    ("weirder_than_peers", "Белая ворона", "recency минус медиана своего региона+сегмента, в сигмах"),
    ("fading_vs_peers", "Тускнеет сильнее соседей", "activity_fade выше медианы группы"),
    ("topup_only", "Только мобильная связь", "остался topup, покупок и grocery почти нет"),
]


def main() -> None:
    engine = create_engine(DSN)
    params_common = {"horizon": HORIZON, "lag": LAG, "min_tenure": MIN_TENURE, "lookback": LOOKBACK_ALIVE}

    print("train features…")
    train = q(engine, FEATURE_SQL, {**params_common, "snap": TRAIN_SNAP})
    train = add_value_cols(train)
    print("  rows", len(train), "churn", train["target"].mean())

    print("test features…")
    test = q(engine, FEATURE_SQL, {**params_common, "snap": TEST_SNAP})
    test = add_value_cols(test)
    print("  rows", len(test), "churn", test["target"].mean())

    out = OUT / "data"
    out.mkdir(exist_ok=True)

    if not (out / "churn_monthly.csv").exists():
        print("monthly churn…")
        q(engine, MONTHLY_CHURN_SQL).to_csv(out / "churn_monthly.csv", index=False)
    if not (out / "uplift_silent_30_90.csv").exists():
        print("uplift observational…")
        q(engine, UPLIFT_SQL).to_csv(out / "uplift_silent_30_90.csv", index=False)

    train.to_csv(out / "churn_train_2025-06-30.csv", index=False)
    test.to_csv(out / f"churn_test_{TEST_SNAP}.csv", index=False)
    pd.DataFrame(FEATURE_DICT, columns=["feature", "name_ru", "why"]).to_csv(out / "feature_dictionary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "snap": TRAIN_SNAP,
                "lag": LAG,
                "label_start": str((pd.Timestamp(TRAIN_SNAP) + pd.Timedelta(days=LAG)).date()),
                "n": len(train),
                "churn_n": int(train["target"].sum()),
                "churn_rate": float(train["target"].mean()),
                "avg_value_90d": float(train["value_retain_90d_uzs"].mean()),
                "loss_if_all_churners": float(train.loc[train["target"] == 1, "value_retain_90d_uzs"].sum()),
            },
            {
                "split": "test",
                "snap": TEST_SNAP,
                "lag": LAG,
                "label_start": str((pd.Timestamp(TEST_SNAP) + pd.Timedelta(days=LAG)).date()),
                "n": len(test),
                "churn_n": int(test["target"].sum()),
                "churn_rate": float(test["target"].mean()),
                "avg_value_90d": float(test["value_retain_90d_uzs"].mean()),
                "loss_if_all_churners": float(test.loc[test["target"] == 1, "value_retain_90d_uzs"].sum()),
            },
        ]
    )
    summary.to_csv(out / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("saved to", out)


if __name__ == "__main__":
    main()

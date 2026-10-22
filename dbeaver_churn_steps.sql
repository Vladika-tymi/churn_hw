-- какие таблицы есть
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename LIKE 'churn\_%'
ORDER BY 1;

-- посмотрел транзакции
SELECT *
FROM churn_transactions
LIMIT 20;

-- какие статусы
SELECT status, COUNT(*) AS n
FROM churn_transactions
GROUP BY 1
ORDER BY n DESC;

-- период и сколько людей
SELECT
    COUNT(*) AS n_txn,
    COUNT(DISTINCT customer_id) AS n_clients,
    MIN(txn_ts::date) AS min_d,
    MAX(txn_ts::date) AS max_d
FROM churn_transactions
WHERE status = 'approved';

-- один клиент
SELECT
    customer_id,
    txn_ts::date AS txn_date,
    txn_type,
    mcc_group,
    channel,
    amount_uzs
FROM churn_transactions
WHERE status = 'approved'
  AND customer_id = 12345
ORDER BY txn_date;

-- паузы у него
WITH daily AS (
    SELECT DISTINCT customer_id, txn_ts::date AS txn_date
    FROM churn_transactions
    WHERE status = 'approved'
      AND customer_id = 12345
)
SELECT
    txn_date,
    LAG(txn_date) OVER (ORDER BY txn_date) AS prev_date,
    txn_date - LAG(txn_date) OVER (ORDER BY txn_date) AS gap_days
FROM daily
ORDER BY txn_date;

-- gaps по всем
WITH daily AS (
    SELECT DISTINCT customer_id, txn_ts::date AS txn_date
    FROM churn_transactions
    WHERE status = 'approved'
),
gaps AS (
    SELECT
        txn_date - LAG(txn_date) OVER (
            PARTITION BY customer_id ORDER BY txn_date
        ) AS gap_days
    FROM daily
)
SELECT
    COUNT(*) AS n_gaps,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gap_days) AS p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY gap_days) AS p90,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_days) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY gap_days) AS p99
FROM gaps
WHERE gap_days IS NOT NULL;

-- после какой тишины уже не возвращаются
WITH daily AS (
    SELECT DISTINCT customer_id, txn_ts::date AS txn_date
    FROM churn_transactions
    WHERE status = 'approved'
),
spells AS (
    SELECT
        COALESCE(
            LEAD(txn_date) OVER (PARTITION BY customer_id ORDER BY txn_date),
            DATE '2025-12-31'
        ) - txn_date AS silence_days,
        CASE
            WHEN LEAD(txn_date) OVER (PARTITION BY customer_id ORDER BY txn_date) IS NOT NULL
            THEN 1 ELSE 0
        END AS returned
    FROM daily
)
SELECT 28 AS n_days,
       COUNT(*) FILTER (WHERE silence_days >= 28) AS n_spells,
       ROUND(100.0 * AVG(returned::numeric) FILTER (WHERE silence_days >= 28), 1) AS pct_returned
FROM spells
UNION ALL
SELECT 60,
       COUNT(*) FILTER (WHERE silence_days >= 60),
       ROUND(100.0 * AVG(returned::numeric) FILTER (WHERE silence_days >= 60), 1)
FROM spells
UNION ALL
SELECT 90,
       COUNT(*) FILTER (WHERE silence_days >= 90),
       ROUND(100.0 * AVG(returned::numeric) FILTER (WHERE silence_days >= 90), 1)
FROM spells;

-- таргет на 30 июня
WITH snap AS (
    SELECT DATE '2025-06-30' AS d
),
last_before AS (
    SELECT t.customer_id, MAX(t.txn_ts::date) AS last_d
    FROM churn_transactions t, snap s
    WHERE t.status = 'approved' AND t.txn_ts::date < s.d
    GROUP BY 1
),
next_90 AS (
    SELECT t.customer_id, COUNT(*) AS n_after
    FROM churn_transactions t, snap s
    WHERE t.status = 'approved'
      AND t.txn_ts::date > s.d
      AND t.txn_ts::date <= s.d + 90
    GROUP BY 1
)
SELECT
    COUNT(*) AS n_alive,
    COUNT(*) FILTER (WHERE COALESCE(n.n_after, 0) = 0) AS n_churn,
    ROUND(100.0 * COUNT(*) FILTER (WHERE COALESCE(n.n_after, 0) = 0) / COUNT(*), 2) AS churn_pct
FROM last_before l
JOIN snap s ON TRUE
JOIN churn_customers c ON c.customer_id = l.customer_id
LEFT JOIN next_90 n ON n.customer_id = l.customer_id
WHERE l.last_d >= s.d - 90
  AND c.signup_date <= s.d - 60;

-- то же на 30 сентября
WITH snap AS (
    SELECT DATE '2025-09-30' AS d
),
last_before AS (
    SELECT customer_id, MAX(txn_ts::date) AS last_d
    FROM churn_transactions, snap s
    WHERE status = 'approved' AND txn_ts::date < s.d
    GROUP BY 1
),
next_90 AS (
    SELECT customer_id, COUNT(*) AS n_after
    FROM churn_transactions, snap s
    WHERE status = 'approved'
      AND txn_ts::date > s.d
      AND txn_ts::date <= s.d + 90
    GROUP BY 1
)
SELECT
    COUNT(*) AS n_alive,
    COUNT(*) FILTER (WHERE COALESCE(n.n_after, 0) = 0) AS n_churn,
    ROUND(100.0 * COUNT(*) FILTER (WHERE COALESCE(n.n_after, 0) = 0) / COUNT(*), 2) AS churn_pct
FROM last_before l
JOIN snap s ON TRUE
JOIN churn_customers c ON c.customer_id = l.customer_id
LEFT JOIN next_90 n ON n.customer_id = l.customer_id
WHERE l.last_d >= s.d - 90
  AND c.signup_date <= s.d - 60;

-- отток по месяцам
WITH months AS (
    SELECT generate_series(DATE '2024-07-01', DATE '2025-09-01', INTERVAL '1 month')::date AS snap
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
    COUNT(*) FILTER (WHERE last_before >= snap - 90) AS n_alive,
    COUNT(*) FILTER (WHERE last_before >= snap - 90 AND NOT active_next90) AS n_churn,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE last_before >= snap - 90 AND NOT active_next90)
        / NULLIF(COUNT(*) FILTER (WHERE last_before >= snap - 90), 0),
        2
    ) AS churn_pct
FROM base
GROUP BY 1
ORDER BY 1;

-- тариф, похоже из-за него
SELECT
    reason,
    MIN(effective_date) AS from_d,
    ROUND(AVG(old_monthly_fee_uzs)) AS old_fee,
    ROUND(AVG(new_monthly_fee_uzs)) AS new_fee,
    COUNT(*) AS n
FROM churn_tariff_changes
GROUP BY reason;

-- лаг 5 дней: фичи до asof, таргет после snap
-- snap = день скоринга, asof = snap - 5
WITH p AS (
    SELECT
        DATE '2025-06-30' AS snap,
        DATE '2025-06-30' - 5 AS asof,
        90 AS horizon
),
pop AS (
    SELECT c.customer_id, p.snap, p.asof
    FROM churn_customers c, p
    WHERE c.signup_date <= p.snap - 60
),
feat AS (
    SELECT
        t.customer_id,
        MAX(t.txn_ts::date) FILTER (WHERE t.status = 'approved') AS last_d,
        COUNT(*) FILTER (WHERE t.status = 'approved') AS txn_asof_90
    FROM churn_transactions t, p
    WHERE t.txn_ts::date < p.asof
      AND t.txn_ts::date >= p.asof - 90
    GROUP BY t.customer_id
),
lab AS (
    SELECT
        t.customer_id,
        COUNT(*) FILTER (
            WHERE t.status = 'approved'
              AND t.txn_ts::date > p.snap
              AND t.txn_ts::date <= p.snap + p.horizon
        ) AS txn_next_h
    FROM churn_transactions t, p
    GROUP BY t.customer_id
)
SELECT
    COUNT(*) AS n,
    SUM(CASE WHEN COALESCE(lab.txn_next_h, 0) = 0 THEN 1 ELSE 0 END) AS churn_n,
    ROUND(100.0 * AVG(CASE WHEN COALESCE(lab.txn_next_h, 0) = 0 THEN 1 ELSE 0 END), 2) AS churn_pct
FROM pop
JOIN feat ON feat.customer_id = pop.customer_id AND feat.last_d >= pop.asof - 90
LEFT JOIN lab ON lab.customer_id = pop.customer_id;

-- простые фичи на 30 июня (с лагом: только до asof)
WITH snap AS (
    SELECT DATE '2025-06-30' AS d, DATE '2025-06-30' - 5 AS asof
),
x AS (
    SELECT
        t.customer_id,
        MAX(t.txn_ts::date) FILTER (WHERE t.status = 'approved') AS last_d,
        COUNT(*) FILTER (WHERE t.status = 'approved' AND t.txn_ts::date >= s.asof - 30) AS txn_30,
        COUNT(*) FILTER (WHERE t.status = 'approved' AND t.txn_ts::date >= s.asof - 90) AS txn_90,
        MAX(t.txn_ts::date) FILTER (WHERE t.status = 'approved' AND t.mcc_group = 'grocery') AS last_grocery
    FROM churn_transactions t, snap s
    WHERE t.txn_ts::date < s.asof
    GROUP BY t.customer_id, s.d, s.asof
)
SELECT
    customer_id,
    (SELECT d FROM snap) - last_d AS recency_days,
    txn_30,
    txn_90,
    (SELECT d FROM snap) - last_grocery AS days_without_grocery
FROM x
WHERE last_d >= (SELECT d FROM snap) - 90
LIMIT 30;

-- лестница по месяцам, один клиент
WITH snap AS (
    SELECT DATE '2025-06-30' AS d
)
SELECT
    customer_id,
    COUNT(*) FILTER (WHERE txn_ts::date >= d - 90 AND txn_ts::date < d - 60) AS m3,
    COUNT(*) FILTER (WHERE txn_ts::date >= d - 60 AND txn_ts::date < d - 30) AS m2,
    COUNT(*) FILTER (WHERE txn_ts::date >= d - 30 AND txn_ts::date < d) AS m1
FROM churn_transactions, snap
WHERE status = 'approved'
  AND customer_id = 12345
  AND txn_ts::date < d
GROUP BY customer_id;

-- кусок датасета: recency + target
WITH snap AS (
    SELECT DATE '2025-06-30' AS d
),
feat AS (
    SELECT
        t.customer_id,
        MAX(t.txn_ts::date) FILTER (WHERE t.status = 'approved') AS last_d,
        COUNT(*) FILTER (WHERE t.status = 'approved' AND t.txn_ts::date >= s.d - 30) AS txn_30,
        COUNT(*) FILTER (WHERE t.status = 'approved' AND t.txn_ts::date >= s.d - 90) AS txn_90
    FROM churn_transactions t, snap s
    WHERE t.txn_ts::date < s.d
    GROUP BY t.customer_id, s.d
),
tgt AS (
    SELECT t.customer_id, COUNT(*) AS n_after
    FROM churn_transactions t, snap s
    WHERE t.status = 'approved'
      AND t.txn_ts::date > s.d
      AND t.txn_ts::date <= s.d + 90
    GROUP BY 1
)
SELECT
    f.customer_id,
    c.region,
    c.segment,
    c.is_payroll,
    (SELECT d FROM snap) - f.last_d AS recency_days,
    f.txn_30,
    f.txn_90,
    CASE WHEN COALESCE(t.n_after, 0) = 0 THEN 1 ELSE 0 END AS target
FROM feat f
JOIN snap s ON TRUE
JOIN churn_customers c ON c.customer_id = f.customer_id
LEFT JOIN tgt t ON t.customer_id = f.customer_id
WHERE f.last_d >= s.d - 90
  AND c.signup_date <= s.d - 60
LIMIT 100;

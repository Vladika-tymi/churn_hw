"""Generate churn_target_uplift.ipynb"""
import json
from pathlib import Path

cells = []


def md(src: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src: str):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
    )


md(
    """# Отток клиентов: таргет, uplift, датасет

Директор: «теряем клиентов, нужна модель кто уйдёт следующим».

События «расторг договор» в базе нет. Таргет определяем сами по тишине в `churn_transactions`.
Готовую метку `churn_customers_360.is_churned` **не используем** — там просто «нет операций 31 день».

**Срезы:** train `2025-06-30`, test `2025-09-30` (out-of-time, не random split).
Данные до `2025-12-31`.
"""
)

code(
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)

DSN = os.environ["ACADEMY_DSN"]
engine = create_engine(DSN)
DATA = Path("data")

def q(sql: str, params=None) -> pd.DataFrame:
    return pd.read_sql(sql, engine, params=params)
"""
)

md(
    """## 0. Какие таблицы берём

Основа: `churn_customers`, `churn_transactions` (`status='approved'`), `churn_cards`,
`churn_support_tickets`, `churn_subscriptions`, `churn_subscription_charges`,
`churn_tariff_changes`, `churn_app_events`, `churn_retention_campaigns`.

Не берём: `churn_temp`, `churn_customers_crm_old`, `churn_customers_360` (чужой таргет + агрегаты «на сегодня» = leakage).
`churn_transactions_archive` — сначала сверить, не дубль ли.
"""
)

code(
    """q(\"\"\"
SELECT tablename
FROM pg_tables
WHERE schemaname = 'public' AND tablename LIKE 'churn\\\\_%'
ORDER BY 1
\"\"\")"""
)

md(
    """## 1. Почему порог тишины 90 дней, а не 30

Сначала gaps между соседними операциями (ритм жизни), потом return rate
(если молчит N дней — какая доля потом всё-таки возвращается).
"""
)

code(
    """gaps = q(\"\"\"
WITH daily AS (
    SELECT DISTINCT customer_id, txn_ts::date AS txn_date
    FROM churn_transactions
    WHERE status = 'approved'
),
g AS (
    SELECT txn_date - LAG(txn_date) OVER (PARTITION BY customer_id ORDER BY txn_date) AS gap_days
    FROM daily
)
SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gap_days) AS p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY gap_days) AS p90,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_days) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY gap_days) AS p99
FROM g WHERE gap_days IS NOT NULL
\"\"\")
gaps"""
)

code(
    """ret = q(\"\"\"
WITH daily AS (
    SELECT DISTINCT customer_id, txn_ts::date AS txn_date
    FROM churn_transactions
    WHERE status = 'approved'
),
spells AS (
    SELECT
        COALESCE(LEAD(txn_date) OVER (PARTITION BY customer_id ORDER BY txn_date), DATE '2025-12-31')
            - txn_date AS silence_days,
        CASE WHEN LEAD(txn_date) OVER (PARTITION BY customer_id ORDER BY txn_date) IS NOT NULL
             THEN 1 ELSE 0 END AS returned
    FROM daily
),
th AS (SELECT generate_series(14, 119, 7) AS n)
SELECT th.n AS threshold_n,
       COUNT(*) FILTER (WHERE s.silence_days >= th.n) AS n_spells,
       ROUND(100.0 * AVG(s.returned::numeric) FILTER (WHERE s.silence_days >= th.n), 1) AS pct_returned
FROM th CROSS JOIN spells s
GROUP BY 1 ORDER BY 1
\"\"\")
ret"""
)

md(
    """**Как читаем:** p50 gap = 3, p99 = 37 — обычные паузы короткие.
Но после 28 дней тишины ещё ~83% возвращаются, после 91 дня — ~26%.
30 дней (как в 360) = ложные уходы. Берём **90 дней без approved-операции**.
"""
)

md(
    """## 2. Популяция и таргет

На дату snapshot клиент в выборке, если:

1. зарегистрирован ≥ 60 дней назад (не новичок);
2. была хотя бы одна approved-операция за 90 дней до snap (ещё «живой», не труп);
3. окно snap+90 закрыто к 31.12.2025.

`target = 1`, если в (snap, snap+90] нет ни одной approved-операции.

Фичи считаются **строго до snap**.
"""
)

code(
    """train = pd.read_csv(DATA / "churn_train_2025-06-30.csv", parse_dates=["snap", "signup_date", "last_txn_date"])
test = pd.read_csv(DATA / "churn_test_2025-09-30.csv", parse_dates=["snap", "signup_date", "last_txn_date"])
monthly = pd.read_csv(DATA / "churn_monthly.csv", parse_dates=["snap"])
upl = pd.read_csv(DATA / "uplift_silent_30_90.csv", parse_dates=["snap"])

for name, df in [("train", train), ("test", test)]:
    print(name, "n=", len(df), "churn=", int(df.target.sum()), "rate=", round(df.target.mean(), 4),
          "avg_value_90d=", int(df.value_retain_90d_uzs.mean()))
"""
)

md(
    """Если CSV ещё нет — один раз запусти `python build_churn_dataset.py` (долго, тяжёлые агрегаты в Postgres).
"""
)

md(
    """## 3. Отток по месяцам и почему вырос
"""
)

code(
    """print(monthly.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.snap, monthly.churn90_pct, marker="o")
ax.axvline(pd.Timestamp("2025-07-01"), color="C1", ls="--", label="tariff_revision_2025")
ax.set_ylabel("churn 90д, %")
ax.set_title("Доля ушедших среди живых на начало месяца")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

code(
    """q(\"\"\"
SELECT MIN(effective_date), MAX(effective_date),
       ROUND(AVG(old_monthly_fee_uzs)) old_fee,
       ROUND(AVG(new_monthly_fee_uzs)) new_fee,
       COUNT(*) n
FROM churn_tariff_changes
\"\"\")"""
)

md(
    """Июнь 2025: **4.92%** → июль **6.39%** → сентябрь **7.72%**.
Лето 2024 скачка не дало (6.42 → 6.13). Сезонность отпадает.
1 июля 2025 комиссия карты **8 139 → 23 793** сум (`tariff_revision_2025`).
"""
)

md(
    """## 4. Сценарий uplift: кому писали и что было после

Исторические кампании почти все ушли в уже затихших (recency 30–90), не в активных.
Treatment = контакт в этом месяце. Control = та же тишина, без контакта.
Исход: была ли approved-операция в следующие 90 дней.
"""
)

code(
    """print(upl.groupby("treated").agg(n=("customer_id", "count"), returned=("returned_90", "mean"), gmv90=("gmv_next_90", "mean")))

upl["bucket"] = pd.cut(upl.recency_days, [29, 45, 60, 90], labels=["30-45", "46-60", "61-90"])
print(upl.groupby(["bucket", "treated"], observed=True).agg(n=("customer_id", "count"), returned=("returned_90", "mean")))
"""
)

md(
    """**Что случилось:** control возвращается в ~71% (естественная пауза). Treatment — ~4%.
Даже `responded=1` не означает, что карта снова заработала.

Это не A/B «оффер вредный». Кампании целились в уже уходящих. Для бизнеса дальше:
удерживать **ещё живых** с высоким риском, а не повторять winback по «трупам».
"""
)

md(
    """## 5. Выгода удержания — из наших таблиц, не с рынка

```
value_retain_90d = 3 × (комиссия_карты_на_snap + цена_подписки_если_активна)
```

- комиссия: `churn_tariff_changes` (до 1.07.2025 старая, потом новая);
- подписка: `churn_subscriptions.monthly_price_uzs`, если план активен на snap;
- ×3 = горизонт 90 дней, который мы реально размечаем.

Не используем «средний LTV по рынку».
"""
)

code(
    """print("train avg monthly margin", int(train.monthly_margin_uzs.mean()),
      "value90", int(train.value_retain_90d_uzs.mean()))
print("test  avg monthly margin", int(test.monthly_margin_uzs.mean()),
      "value90", int(test.value_retain_90d_uzs.mean()))
print("потери если все test-ушедшие ушли", int(test.loc[test.target==1, "value_retain_90d_uzs"].sum()))
train[["monthly_margin_uzs", "value_retain_90d_uzs", "has_active_sub", "tariff_already_on", "target"]].describe()
"""
)

md(
    """## 6. Фичи (as-of snapshot)

Все агрегаты `< snap`. Нет `customers_360`, нет NPS (опросы только ноя–дек 2025).

База RFM есть, но защита держится на **своих** фичах — словарь в `data/feature_dictionary.csv`.

| Фича | Гипотеза |
|------|----------|
| `rhythm_break` | тишина / *свой* медианный gap, не чужой «30 дней» |
| `pause_is_record` | текущая пауза длиннее любой за год |
| `grocery_fade` | карта выпала из быта (продукты) |
| `life_shrink` | меньше MCC — жизнь на карте сжалась |
| `missed_payroll` | зарплатник без `salary_in` |
| `digital_ghost` | логин есть, операций нет |
| `autopay_broken` / `cashback_stopped` | сервисы отвалились |
| `deposit_drain` / `unbundling` | забирает вклад, закрывает продукты |
| `fee_pain` / `fee_burden` | боль от июльского тарифа, не «средний рынок» |
| `loan_winddown` | гасит линию, новых выдач нет |
| `expiry_soon` | карта истекает — удобно не перевыпускать |
| `coffee_died` / `fun_died` | мелкие покупки и тусовки уехали первыми |
| `bills_only_zombie` | остались только счета |
| `abandoned_favorite` | бросил свою любимую категорию |
| `goodbye_atm` / `goodbye_p2p` | последний жест — снял или увёл |
| `gaps_widening` | паузы всё длиннее |
| `balance_anxiety` / `settings_rummage` | нервничает в приложении |
| `retention_failed` | уже писали — не помогло |
"""
)

md(
    """## 7. Модель: обучение на июне, проверка на сентябре
"""
)

code(
    """num = [
    "tenure_days", "recency_days", "txn_cnt_14", "txn_cnt_30", "txn_cnt_60", "txn_cnt_90",
    "gmv_30", "gmv_90", "declined_90", "decline_rate_90",
    "cards_active", "cards_dead", "tickets_90", "tickets_bad_90",
    "has_active_sub", "failed_charges_90", "app_events_30", "is_payroll",
    "tariff_already_on", "value_retain_90d_uzs",
    "rhythm_break", "gap_cv", "pause_is_record", "n_comebacks_30",
    "activity_fade", "gmv_fade", "grocery_fade", "life_shrink",
    "missed_payroll", "salary_recency", "drain_share", "digital_ghost",
    "login_to_txn", "autopay_broken", "cashback_stopped", "deposit_drain",
    "unbundling", "fee_pain", "fee_burden", "loan_winddown", "abroad_90",
    "night_share", "ticket_shrink", "decline_accel", "expiry_soon",
    "nsf_fails_90", "sub_cancel_fee_90", "products_open",
    "mono_fade", "usage_halved", "faded_last_month", "bank_switch_score",
    "coffee_died", "fun_died", "bills_only_zombie", "abandoned_favorite",
    "gaps_widening", "goodbye_atm", "goodbye_p2p", "last_was_declined",
    "days_without_grocery", "days_without_pos", "robot_payer",
    "balance_anxiety", "settings_rummage", "ghost_push", "p2p_abandoned",
    "card_colder_than_app", "idle_cards", "long_decline", "confused",
    "open_ticket", "angry_sat", "retention_failed", "post_tariff_mouse",
    "became_leaky", "round_transfer_share",
]
cat = ["gender", "region", "acq_channel", "segment"]

def prep(df):
    x = df.copy()
    for c in num:
        x[c] = x[c].fillna(0)
    for c in cat:
        x[c] = x[c].fillna("unk")
    return x

tr, te = prep(train), prep(test)

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])),
    ("lr", LogisticRegression(max_iter=800, class_weight="balanced")),
])
pipe.fit(tr[num + cat], tr["target"])
te = te.copy()
te["p"] = pipe.predict_proba(te[num + cat])[:, 1]

print("OOT AUC", round(roc_auc_score(te.target, te.p), 4))
print("OOT AP ", round(average_precision_score(te.target, te.p), 4), "base", round(te.target.mean(), 4))
"""
)

code(
    """fpr, tpr, _ = roc_curve(te.target, te.p)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], ls="--", c="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC out-of-time (сен 2025)")
plt.tight_layout(); plt.show()
"""
)

md(
    """## 8. Трешхолд = бюджет, не 0.5

Сортируем по `p × value_retain_90d`: сначала те, на ком банк потеряет больше.
"""
)

code(
    """te["ev"] = te["p"] * te["value_retain_90d_uzs"]

def show(df, k, how):
    top = df.head(k)
    print(
        f"{how} top-{k}: precision={top.target.mean():.3f} "
        f"capture={top.target.sum()/df.target.sum():.3f} "
        f"true_value={int(top.loc[top.target==1,'value_retain_90d_uzs'].sum()):,}"
    )

by_p = te.sort_values("p", ascending=False)
by_ev = te.sort_values("ev", ascending=False)
for k in (200, 500, 1000):
    show(by_p, k, "p only")
    show(by_ev, k, "p*value")
    print("---")
"""
)

md(
    """**Кому что делать, если денег мало**

1. Top по `p*value`, ещё живые — персональный звонок / waiver на **новую** комиссию (она и разогнала отток).
2. Высокий p, низкая маржа (нет подписки, дешёвый тариф) — дешёвый push, не колл-центр.
3. Уже тишина 60–90 дней — исторический winback почти не возвращает, слот бюджета не сжигать.

Пересобрать датасеты: `python build_churn_dataset.py`.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "churn_target_uplift.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(cells))

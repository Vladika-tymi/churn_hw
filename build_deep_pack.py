"""Lighter deep pack: reuse CSVs, light SQL for labels/segments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text

from build_churn_dataset import DSN, LAG

OUT = Path(__file__).resolve().parent / "data"
TRAIN_CSV = OUT / "churn_train_2025-06-30.csv"
TEST_CSV = OUT / "churn_test_2025-09-30.csv"  # content is snap 2025-09-25 / lag6

NUM = [
    "tenure_days", "recency_days", "txn_cnt_30", "txn_cnt_60", "txn_cnt_90",
    "gmv_30", "gmv_90", "declined_90", "decline_rate_90", "purchase_90",
    "salary_90", "cashout_90", "p2p_out_90", "cards_active", "cards_dead",
    "tickets_90", "tickets_bad_90", "has_active_sub", "sub_price",
    "failed_charges_90", "app_events_30", "app_support_30", "is_payroll",
    "declared_income_uzs", "tariff_already_on", "value_retain_90d_uzs",
    "rhythm_break", "pause_is_record", "mono_fade", "bank_switch_score",
    "days_without_grocery", "days_without_pos", "abandoned_favorite",
    "empty_weeks", "retention_failed", "weirder_than_peers", "fee_pain",
]
CAT = ["gender", "region", "acq_channel", "segment"]


def eng():
    return create_engine(DSN, pool_pre_ping=True)


def avail(df, cols):
    return [c for c in cols if c in df.columns]


def prep(train, test, num, cat):
    train, test = train.copy(), test.copy()
    for c in num:
        train[c] = train[c].fillna(0)
        test[c] = test[c].fillna(0)
    for c in cat:
        train[c] = train[c].fillna("unk")
        test[c] = test[c].fillna("unk")
    return train, test


def model(num, cat):
    pre = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    )
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1200, class_weight="balanced"))])


def ap_scorer(est, X, y):
    return average_precision_score(y, est.predict_proba(X)[:, 1])


def labels_for(engine, snap: str, horizon: int, lag: int = LAG) -> pd.DataFrame:
    sql = text(
        """
        SELECT
            t.customer_id,
            COUNT(*) FILTER (
                WHERE t.status = 'approved'
                  AND t.txn_ts::date > DATE :snap + :lag
                  AND t.txn_ts::date <= DATE :snap + :lag + :horizon
            ) AS txn_next_h
        FROM churn_transactions t
        GROUP BY 1
        """
    )
    df = pd.read_sql(sql, engine, params={"snap": snap, "lag": lag, "horizon": horizon})
    df["target"] = (df["txn_next_h"].fillna(0) == 0).astype(int)
    return df[["customer_id", "target"]]


def attach_target(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    out = features.drop(columns=["target"], errors="ignore").merge(labels, on="customer_id", how="inner")
    return out


def horizon_auc(engine, train_f, test_f, train_snap, test_snap):
    rows = []
    for h in (60, 90, 120):
        print("horizon", h)
        # for h=120, test snap 2025-09-25 +6+120 exceeds 2025-12-31 → filter closed windows
        tr = attach_target(train_f, labels_for(engine, train_snap, h))
        te = attach_target(test_f, labels_for(engine, test_snap, h))
        # drop rows whose label window is not closed
        end = pd.Timestamp(test_snap) + pd.Timedelta(days=LAG + h)
        if end > pd.Timestamp("2025-12-31"):
            # keep only conceptually: still compute but flag
            closed = False
        else:
            closed = True
        num, cat = avail(tr, NUM), avail(tr, CAT)
        tr, te = prep(tr, te, num, cat)
        clf = model(num, cat)
        clf.fit(tr[num + cat], tr["target"])
        proba = clf.predict_proba(te[num + cat])[:, 1]
        rows.append(
            {
                "horizon": h,
                "train_snap": train_snap,
                "test_snap": test_snap,
                "window_closed": closed,
                "train_n": len(tr),
                "test_n": len(te),
                "train_rate": float(tr["target"].mean()),
                "test_rate": float(te["target"].mean()),
                "auc": float(roc_auc_score(te["target"], proba)),
                "ap": float(average_precision_score(te["target"], proba)),
            }
        )
    return pd.DataFrame(rows)


def load_or_build_snap_csv(engine, snap: str, path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    from build_churn_dataset import FEATURE_SQL, MIN_TENURE, LOOKBACK_ALIVE, add_value_cols, q

    print("building snap", snap)
    df = q(
        engine,
        FEATURE_SQL,
        {"snap": snap, "horizon": 90, "lag": LAG, "min_tenure": MIN_TENURE, "lookback": LOOKBACK_ALIVE},
    )
    df = add_value_cols(df)
    df.to_csv(path, index=False)
    return df


def feature_stability(engine):
    # windows: importance on Dec-2024 and Jun-2025 test cuts
    pairs = [
        ("2024-09-30", "2024-12-31", OUT / "snap_2024-09-30.csv", OUT / "snap_2024-12-31.csv"),
        ("2025-03-31", "2025-06-30", OUT / "snap_2025-03-31.csv", OUT / "snap_2025-06-30.csv"),
    ]
    rows = []
    for tr_s, te_s, tr_p, te_p in pairs:
        print("stability", tr_s, "->", te_s)
        e = eng()
        train = load_or_build_snap_csv(e, tr_s, tr_p)
        test = load_or_build_snap_csv(e, te_s, te_p)
        e.dispose()
        num, cat = avail(train, NUM), avail(train, CAT)
        train, test = prep(train, test, num, cat)
        clf = model(num, cat)
        clf.fit(train[num + cat], train["target"])
        r = permutation_importance(
            clf, test[num + cat], test["target"], n_repeats=3, random_state=42, scoring=ap_scorer, n_jobs=1
        )
        imp = pd.DataFrame({"feature": num + cat, "importance": r.importances_mean})
        imp = imp.sort_values("importance", ascending=False).head(8)
        imp["window"] = f"{tr_s[:7]}→{te_s[:7]}"
        rows.append(imp)
    return pd.concat(rows, ignore_index=True)


def tariff_hurt(engine):
    sql = text(
        """
WITH months AS (
    SELECT DATE '2025-06-01' AS snap UNION ALL SELECT DATE '2025-09-01'
),
agg AS (
    SELECT
        m.snap,
        c.customer_id,
        c.segment,
        c.is_payroll,
        MAX(t.txn_ts::date) FILTER (WHERE t.status='approved' AND t.txn_ts::date < m.snap) AS last_before,
        BOOL_OR(t.status='approved' AND t.txn_ts::date > m.snap AND t.txn_ts::date <= m.snap + 90) AS active_next90
    FROM months m
    JOIN churn_customers c ON c.signup_date <= m.snap - 60
    LEFT JOIN churn_transactions t ON t.customer_id = c.customer_id
    GROUP BY 1,2,3,4
),
alive AS (SELECT * FROM agg WHERE last_before >= snap - 90)
SELECT snap::text AS snap, 'payroll_' || is_payroll::text AS grp, COUNT(*) n,
       ROUND(100.0*AVG((NOT active_next90)::int),2) churn_pct
FROM alive GROUP BY 1,2
UNION ALL
SELECT snap::text, COALESCE(segment,'unk'), COUNT(*),
       ROUND(100.0*AVG((NOT active_next90)::int),2)
FROM alive GROUP BY 1,2
ORDER BY 1,2
"""
    )
    return pd.read_sql(sql, engine)


def onboarding(engine):
    sql = text(
        """
WITH snap AS (SELECT DATE '2025-06-30' AS d),
x AS (
  SELECT c.customer_id,
         (s.d - c.signup_date) AS tenure_days,
         MAX(t.txn_ts::date) FILTER (WHERE t.status='approved' AND t.txn_ts::date < s.d) AS last_before,
         BOOL_OR(t.status='approved' AND t.txn_ts::date > s.d + 6 AND t.txn_ts::date <= s.d + 96) AS active
  FROM churn_customers c
  CROSS JOIN snap s
  LEFT JOIN churn_transactions t ON t.customer_id=c.customer_id
  GROUP BY 1,2
)
SELECT CASE WHEN tenure_days < 60 THEN 'onboarding_<60' ELSE 'main_>=60' END AS bucket,
       COUNT(*) AS n_customers,
       COUNT(*) FILTER (WHERE last_before >= (SELECT d FROM snap)-90) AS n_alive_like,
       ROUND(100.0 * AVG(CASE WHEN last_before >= (SELECT d FROM snap)-90 AND NOT COALESCE(active,false) THEN 1 ELSE 0 END)
             FILTER (WHERE last_before >= (SELECT d FROM snap)-90), 2) AS churn_pct
FROM x GROUP BY 1 ORDER BY 1
"""
    )
    return pd.read_sql(sql, engine)


def local_shap(train, test):
    num, cat = avail(train, NUM), avail(train, CAT)
    train, test = prep(train, test, num, cat)
    clf = model(num, cat)
    clf.fit(train[num + cat], train["target"])
    test = test.copy()
    test["p"] = clf.predict_proba(test[num + cat])[:, 1]

    cases = []
    fee = test[(test.get("fee_pain", 0) >= 1) & (test["p"] >= 0.65)].sort_values("p", ascending=False)
    if len(fee):
        cases.append(("A: high risk + fee_pain", fee.iloc[[0]]))
    mf = test[(test.get("mono_fade", 0) == 1) & (test["days_without_grocery"] >= 60) & (test["p"] >= 0.5)].sort_values(
        "p", ascending=False
    )
    if len(mf):
        cases.append(("B: mono_fade + grocery silence", mf.iloc[[0]]))
    while len(cases) < 2:
        cases.append((f"top risk #{len(cases)+1}", test.sort_values("p", ascending=False).iloc[[len(cases)]]))

    # try shap LinearExplainer
    try:
        import shap

        pre = clf.named_steps["pre"]
        lr = clf.named_steps["lr"]
        Xtr = pre.transform(train[num + cat])
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray()
        explainer = shap.LinearExplainer(lr, Xtr)
        use_shap = True
    except Exception as ex:
        print("shap fallback:", ex)
        use_shap = False
        explainer = None
        pre = clf.named_steps["pre"]
        lr = clf.named_steps["lr"]

    ohe = pre.named_transformers_["cat"]
    names = num + list(ohe.get_feature_names_out(cat))
    rows = []
    for label, part in cases:
        row = part.iloc[0]
        Xt = pre.transform(part[num + cat])
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        if use_shap:
            sv = explainer.shap_values(Xt)[0]
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            contrib = np.array(sv).reshape(-1)
        else:
            contrib = Xt[0] * lr.coef_[0]
        top = np.argsort(np.abs(contrib))[::-1][:8]
        for i in top:
            rows.append(
                {
                    "case": label,
                    "customer_id": int(row["customer_id"]),
                    "p_churn": round(float(row["p"]), 3),
                    "target": int(row["target"]),
                    "feature": names[i],
                    "contribution": round(float(contrib[i]), 4),
                    "fee_pain": float(row.get("fee_pain", 0) or 0),
                    "mono_fade": float(row.get("mono_fade", 0) or 0),
                    "days_without_grocery": float(row.get("days_without_grocery", 0) or 0),
                    "retention_failed": float(row.get("retention_failed", 0) or 0),
                    "method": "shap_linear" if use_shap else "coef_x_value",
                }
            )
    return pd.DataFrame(rows)


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train_snap = "2025-06-30"
    test_snap = "2025-09-25"

    e = eng()
    print("=== horizons ===")
    h = horizon_auc(e, train, test, train_snap, test_snap)
    h.to_csv(OUT / "deep_horizons_auc.csv", index=False)
    print(h.to_string(index=False))
    e.dispose()

    print("=== tariff ===")
    e = eng()
    t = tariff_hurt(e)
    t.to_csv(OUT / "deep_tariff_segments.csv", index=False)
    print(t.to_string(index=False))
    e.dispose()

    print("=== onboarding ===")
    e = eng()
    o = onboarding(e)
    o.to_csv(OUT / "deep_onboarding.csv", index=False)
    print(o.to_string(index=False))
    e.dispose()

    print("=== local shap ===")
    loc = local_shap(train, test)
    loc.to_csv(OUT / "deep_local_shaplike.csv", index=False)
    print(loc.to_string(index=False))

    print("=== stability (may rebuild snaps) ===")
    e = eng()
    s = feature_stability(e)
    s.to_csv(OUT / "deep_feature_stability.csv", index=False)
    print(s.to_string(index=False))
    e.dispose()

    print("DONE", OUT)


if __name__ == "__main__":
    main()

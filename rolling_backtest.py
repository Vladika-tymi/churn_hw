from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine

from build_churn_dataset import (
    DSN,
    FEATURE_SQL,
    HORIZON,
    LAG,
    LOOKBACK_ALIVE,
    MIN_TENURE,
    add_value_cols,
    q,
)

OUT = Path(__file__).resolve().parent / "data"
SNAPS = [
    "2024-06-30",
    "2024-09-30",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
    "2025-09-25",
]

NUM = [
    "tenure_days",
    "recency_days",
    "txn_cnt_30",
    "txn_cnt_60",
    "txn_cnt_90",
    "gmv_30",
    "gmv_90",
    "declined_90",
    "decline_rate_90",
    "purchase_90",
    "salary_90",
    "cashout_90",
    "p2p_out_90",
    "cards_active",
    "cards_dead",
    "tickets_90",
    "tickets_bad_90",
    "has_active_sub",
    "sub_price",
    "failed_charges_90",
    "app_events_30",
    "app_support_30",
    "is_payroll",
    "declared_income_uzs",
    "tariff_already_on",
    "value_retain_90d_uzs",
    "rhythm_break",
    "pause_is_record",
    "mono_fade",
    "bank_switch_score",
    "days_without_grocery",
    "days_without_pos",
    "abandoned_favorite",
    "empty_weeks",
    "retention_failed",
    "weirder_than_peers",
]
CAT = ["gender", "region", "acq_channel", "segment"]


def load_snapshot(engine, snap: str) -> pd.DataFrame:
    params = {
        "snap": snap,
        "horizon": HORIZON,
        "lag": LAG,
        "min_tenure": MIN_TENURE,
        "lookback": LOOKBACK_ALIVE,
    }
    df = q(engine, FEATURE_SQL, params)
    df = add_value_cols(df)
    df["snap"] = pd.to_datetime(df["snap"])
    return df


def prep(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    for c in NUM:
        train[c] = train[c].fillna(0)
        test[c] = test[c].fillna(0)
    for c in CAT:
        train[c] = train[c].fillna("unk")
        test[c] = test[c].fillna("unk")
    return train, test


def build_model():
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]
    )


def main():
    engine = create_engine(DSN)
    OUT.mkdir(exist_ok=True)

    data = {}
    snapshot_rows = []
    for snap in SNAPS:
        print("building", snap)
        df = load_snapshot(engine, snap)
        data[snap] = df
        snapshot_rows.append(
            {
                "snap": snap,
                "asof": str(pd.Timestamp(snap).date()),
                "label_start": str((pd.Timestamp(snap) + pd.Timedelta(days=LAG)).date()),
                "n": len(df),
                "churn_n": int(df["target"].sum()),
                "churn_rate": float(df["target"].mean()),
            }
        )

    pd.DataFrame(snapshot_rows).to_csv(OUT / "rolling_snapshots.csv", index=False)

    rows = []
    for train_snap, test_snap in zip(SNAPS[:-1], SNAPS[1:]):
        train = data[train_snap]
        test = data[test_snap]
        train, test = prep(train, test)
        clf = build_model()
        clf.fit(train[NUM + CAT], train["target"])
        proba = clf.predict_proba(test[NUM + CAT])[:, 1]
        rows.append(
            {
                "train_snap": train_snap,
                "test_snap": test_snap,
                "train_n": len(train),
                "test_n": len(test),
                "train_rate": float(train["target"].mean()),
                "test_rate": float(test["target"].mean()),
                "auc": float(roc_auc_score(test["target"], proba)),
                "ap": float(average_precision_score(test["target"], proba)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rolling_backtest.csv", index=False)
    print(out.to_string(index=False))

    try:
        from build_churn_pptx import build_backtest_chart

        chart = build_backtest_chart(OUT / "rolling_backtest.csv")
        print("chart", chart)
    except Exception as exc:
        print("chart skipped:", exc)


if __name__ == "__main__":
    main()

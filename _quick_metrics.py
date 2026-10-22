import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

p = Path(r"c:\Users\user\OneDrive\Desktop\dzuzcard\churn_hw\data")
train = pd.read_csv(p / "churn_train_2025-06-30.csv")
test = pd.read_csv(p / "churn_test_2025-09-30.csv")
upl = pd.read_csv(p / "uplift_silent_30_90.csv")
monthly = pd.read_csv(p / "churn_monthly.csv")

print("MONTHLY")
print(monthly.to_string(index=False))

print("\nUPLIFT treated vs control")
print(
    upl.groupby("treated").agg(
        n=("customer_id", "count"),
        ret=("returned_90", "mean"),
        gmv=("gmv_next_90", "mean"),
    )
)
upl["bucket"] = pd.cut(upl.recency_days, [29, 45, 60, 90], labels=["30-45", "46-60", "61-90"])
print("\nby recency")
print(upl.groupby(["bucket", "treated"], observed=True).agg(n=("customer_id", "count"), ret=("returned_90", "mean")))

num = [
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
]
cat = ["gender", "region", "acq_channel", "segment"]
for c in num:
    train[c] = train[c].fillna(0)
    test[c] = test[c].fillna(0)
for c in cat:
    train[c] = train[c].fillna("unk")
    test[c] = test[c].fillna("unk")

pre = ColumnTransformer(
    [
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ]
)
clf = Pipeline(
    [
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=800, class_weight="balanced")),
    ]
)
clf.fit(train[num + cat], train["target"])
proba = clf.predict_proba(test[num + cat])[:, 1]
print("\nTEST AUC", round(roc_auc_score(test.target, proba), 4), "AP", round(average_precision_score(test.target, proba), 4), "base", round(test.target.mean(), 4))

test = test.copy()
test["p"] = proba
test = test.sort_values("p", ascending=False)
for k in [200, 500, 1000, 2000]:
    top = test.head(k)
    cap = top.target.sum() / test.target.sum()
    print(
        "top%s: precision=%.3f capture=%.3f true_value=%s"
        % (k, top.target.mean(), cap, int(top.loc[top.target == 1, "value_retain_90d_uzs"].sum()))
    )

test["ev"] = test["p"] * test["value_retain_90d_uzs"]
t2 = test.sort_values("ev", ascending=False)
print("\nby expected loss p*value")
for k in [200, 500, 1000]:
    top = t2.head(k)
    print(
        "top%s: churners=%s true_value=%s"
        % (k, int(top.target.sum()), int(top.loc[top.target == 1, "value_retain_90d_uzs"].sum()))
    )

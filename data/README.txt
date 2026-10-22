churn_train_2025-06-30.csv  — train, snap 2025-06-30, label с 2025-07-06 (snap+6) на 90д
churn_test_2025-09-25.csv   — OOT test, snap 2025-09-25, label с 2025-10-01
churn_monthly.csv           — доля оттока по месяцам среди «живых»
rolling_backtest.csv        — train t -> test t+1, AUC/AP
rolling_backtest_chart.png  — график backtest
dataset_summary.csv         — размеры, lag, label_start, bad rate

Правило: HORIZON=90, LAG=6 после snap (фичи < snap, target (snap+6, snap+6+90]).
Пересборка: py build_churn_dataset.py && py rolling_backtest.py && py build_churn_pptx.py

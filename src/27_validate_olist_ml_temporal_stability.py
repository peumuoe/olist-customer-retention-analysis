"""Validate fixed Olist logistic model with rolling backtests and bootstrap CIs."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
BACKTEST_PATH = TABLES / "olist_ml_rolling_time_backtest.csv"
BOOTSTRAP_PATH = TABLES / "olist_ml_test_bootstrap_confidence_intervals.csv"
STABILITY_PATH = TABLES / "olist_ml_temporal_stability_summary.csv"
CHECK_PATH = TABLES / "olist_ml_temporal_validation_build_summary.csv"
PREDICTION_PATH = ROOT / "data" / "processed" / "olist_ml_test_predictions.csv"

spec = spec_from_file_location("ml26", ROOT / "src" / "26_train_olist_final_repurchase_models.py")
ml26 = module_from_spec(spec)
spec.loader.exec_module(ml26)


def top_k_metrics(y, p, fraction=0.10):
    n = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-p)[:n]
    positives = int(np.asarray(y)[chosen].sum())
    base = np.mean(y)
    rate = positives / n
    return positives, positives / np.sum(y), rate / base


def rolling_backtest(data):
    n = len(data)
    folds = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]
    rows = []
    features = ml26.NUMERIC + ml26.CATEGORICAL
    for fold_id, (train_end_fraction, eval_end_fraction) in enumerate(folds, 1):
        train_end = int(n * train_end_fraction)
        eval_end = int(n * eval_end_fraction)
        train = data.iloc[:train_end]
        evaluate = data.iloc[train_end:eval_end]
        model = Pipeline([
            ("prep", ml26.make_preprocessor()),
            ("model", LogisticRegression(C=0.01, max_iter=3000, solver="lbfgs")),
        ])
        model.fit(train[features], train[ml26.TARGET].astype(int))
        y = evaluate[ml26.TARGET].astype(int).to_numpy()
        p = model.predict_proba(evaluate[features])[:, 1]
        captured, recall, lift = top_k_metrics(y, p)
        rows.append({
            "fold_id": fold_id,
            "train_row_count": len(train),
            "train_positive_count": int(train[ml26.TARGET].sum()),
            "train_end_timestamp": train["order_purchase_timestamp"].max(),
            "evaluation_row_count": len(evaluate),
            "evaluation_positive_count": int(y.sum()),
            "evaluation_start_timestamp": evaluate["order_purchase_timestamp"].min(),
            "evaluation_end_timestamp": evaluate["order_purchase_timestamp"].max(),
            "evaluation_positive_rate_pct": y.mean() * 100,
            "mean_predicted_probability_pct": p.mean() * 100,
            "pr_auc": average_precision_score(y, p),
            "pr_auc_vs_prevalence_ratio": average_precision_score(y, p) / y.mean(),
            "roc_auc": roc_auc_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "top10_selected_count": int(np.ceil(len(y) * 0.10)),
            "top10_captured_positive_count": captured,
            "recall_at_10pct": recall,
            "lift_at_10pct": lift,
        })
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(6)
    return result


def bootstrap_test(iterations=2000, seed=42):
    pred = pd.read_csv(PREDICTION_PATH)
    y = pred[ml26.TARGET].astype(int).to_numpy()
    p = pred["score_logistic_raw"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0:
            continue
        captured, recall, lift = top_k_metrics(yb, pb)
        rows.append({
            "pr_auc": average_precision_score(yb, pb),
            "roc_auc": roc_auc_score(yb, pb),
            "lift_at_10pct": lift,
            "recall_at_10pct": recall,
            "top10_captured_positive_count": captured,
        })
    samples = pd.DataFrame(rows)
    point_captured, point_recall, point_lift = top_k_metrics(y, p)
    point = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "lift_at_10pct": point_lift,
        "recall_at_10pct": point_recall,
        "top10_captured_positive_count": point_captured,
    }
    summary = []
    for metric in point:
        summary.append({
            "metric": metric,
            "point_estimate": point[metric],
            "bootstrap_mean": samples[metric].mean(),
            "ci95_low": samples[metric].quantile(0.025),
            "ci95_high": samples[metric].quantile(0.975),
            "bootstrap_iterations": len(samples),
        })
    result = pd.DataFrame(summary)
    for column in ["point_estimate", "bootstrap_mean", "ci95_low", "ci95_high"]:
        result[column] = result[column].round(6)
    return result, samples


def main():
    data = ml26.load_data()
    backtest = rolling_backtest(data)
    bootstrap, samples = bootstrap_test()
    stability = pd.DataFrame([{
        "fold_count": len(backtest),
        "folds_pr_auc_above_prevalence": int((backtest["pr_auc_vs_prevalence_ratio"] > 1).sum()),
        "folds_lift10_above_1": int((backtest["lift_at_10pct"] > 1).sum()),
        "mean_pr_auc_vs_prevalence_ratio": backtest["pr_auc_vs_prevalence_ratio"].mean(),
        "min_pr_auc_vs_prevalence_ratio": backtest["pr_auc_vs_prevalence_ratio"].min(),
        "max_pr_auc_vs_prevalence_ratio": backtest["pr_auc_vs_prevalence_ratio"].max(),
        "mean_lift_at_10pct": backtest["lift_at_10pct"].mean(),
        "min_lift_at_10pct": backtest["lift_at_10pct"].min(),
        "max_lift_at_10pct": backtest["lift_at_10pct"].max(),
        "positive_rate_min_pct": backtest["evaluation_positive_rate_pct"].min(),
        "positive_rate_max_pct": backtest["evaluation_positive_rate_pct"].max(),
        "stability_conclusion": "weak_and_time_varying" if (backtest["lift_at_10pct"] <= 1).any() else "positive_all_folds",
    }])
    numeric = stability.select_dtypes(include="number").columns
    stability[numeric] = stability[numeric].round(6)
    checks = pd.DataFrame([
        ("eligible_row_count", len(data), 86772, len(data) == 86772),
        ("rolling_fold_count", len(backtest), 4, len(backtest) == 4),
        ("folds_with_positives", int((backtest["evaluation_positive_count"] > 0).sum()), 4, (backtest["evaluation_positive_count"] > 0).all()),
        ("bootstrap_iteration_count", len(samples), 2000, len(samples) == 2000),
        ("bootstrap_metric_count", len(bootstrap), 5, len(bootstrap) == 5),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    backtest.to_csv(BACKTEST_PATH, index=False)
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    stability.to_csv(STABILITY_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    print("\nROLLING BACKTEST\n", backtest.to_string(index=False))
    print("\nBOOTSTRAP CI\n", bootstrap.to_string(index=False))
    print("\nSTABILITY\n", stability.to_string(index=False))
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

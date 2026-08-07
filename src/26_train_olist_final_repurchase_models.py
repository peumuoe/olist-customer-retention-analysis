"""Train and evaluate leakage-safe Olist order-time repurchase models."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "olist_asof_crm_segment_customer_mart.csv"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"

SPLIT_PATH = TABLES / "olist_ml_time_split_summary.csv"
VALIDATION_PATH = TABLES / "olist_ml_validation_candidates.csv"
TEST_PATH = TABLES / "olist_ml_test_model_summary.csv"
LIFT_PATH = TABLES / "olist_ml_test_lift_recall.csv"
CALIBRATION_PATH = TABLES / "olist_ml_test_calibration_deciles.csv"
SEGMENT_PATH = TABLES / "olist_ml_test_segment_performance.csv"
IMPORTANCE_PATH = TABLES / "olist_ml_feature_importance.csv"
PREDICTION_PATH = ROOT / "data" / "processed" / "olist_ml_test_predictions.csv"
CHECK_PATH = TABLES / "olist_ml_build_summary.csv"

TARGET = "target_repeat_over_24h_within_30d"
NUMERIC = [
    "item_count", "product_count", "seller_count", "category_count",
    "item_price_sum", "freight_value_sum", "payment_value_sum",
    "payment_installments_max", "payment_type_count", "freight_share",
    "estimated_delivery_lead_days", "purchase_hour", "purchase_weekday",
    "purchase_month",
]
CATEGORICAL = [
    "customer_state", "primary_category", "primary_payment_type",
    "order_value_band", "item_count_group", "seller_count_group",
    "installment_group",
]
KS = [0.01, 0.05, 0.10, 0.20]


def load_data():
    data = pd.read_csv(MART_PATH, low_memory=False)
    data = data.loc[data["eligible_30d"].eq(True)].copy()
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])
    data["order_estimated_delivery_date"] = pd.to_datetime(data["order_estimated_delivery_date"])
    data["estimated_delivery_lead_days"] = (
        data["order_estimated_delivery_date"] - data["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour
    data["purchase_weekday"] = data["order_purchase_timestamp"].dt.weekday
    data["purchase_month"] = data["order_purchase_timestamp"].dt.month
    for col in CATEGORICAL:
        data[col] = data[col].astype("string").fillna("unknown")
    data = data.sort_values(["order_purchase_timestamp", "customer_unique_id"]).reset_index(drop=True)
    n = len(data)
    boundaries = [int(n * x) for x in [0.60, 0.70, 0.80]]
    data["ml_split"] = "test"
    data.loc[:boundaries[0] - 1, "ml_split"] = "train"
    data.loc[boundaries[0]:boundaries[1] - 1, "ml_split"] = "validation_select"
    data.loc[boundaries[1]:boundaries[2] - 1, "ml_split"] = "calibration"
    return data


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def metrics(y, p):
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
        "mean_predicted_probability_pct": np.mean(p) * 100,
        "observed_positive_rate_pct": np.mean(y) * 100,
    }


def platt_fit(scores, y):
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2000)
    calibrator.fit(logits, y)
    return calibrator


def platt_predict(calibrator, scores):
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def lift_rows(model_name, y, p):
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    base = np.mean(y_sorted)
    total_pos = y_sorted.sum()
    rows = []
    for k in KS:
        n = max(1, int(np.ceil(len(y_sorted) * k)))
        positives = int(y_sorted[:n].sum())
        rate = positives / n
        rows.append({
            "model_name": model_name, "top_fraction": k,
            "selected_customer_count": n, "captured_positive_count": positives,
            "recall_at_k_pct": positives / total_pos * 100,
            "positive_rate_at_k_pct": rate * 100,
            "lift_at_k": rate / base,
        })
    return rows


def calibration_rows(model_name, y, p):
    frame = pd.DataFrame({"y": np.asarray(y), "p": p})
    frame["probability_decile"] = pd.qcut(
        frame["p"].rank(method="first"), 10, labels=False
    ) + 1
    return frame.groupby("probability_decile", as_index=False).agg(
        customer_count=("y", "size"), positive_count=("y", "sum"),
        mean_predicted_probability=("p", "mean"), observed_positive_rate=("y", "mean"),
    ).assign(model_name=model_name)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = load_data()
    features = NUMERIC + CATEGORICAL
    split = data.groupby("ml_split", sort=False).agg(
        row_count=(TARGET, "size"), positive_count=(TARGET, "sum"),
        start_timestamp=("order_purchase_timestamp", "min"),
        end_timestamp=("order_purchase_timestamp", "max"),
    ).reset_index()
    split["positive_rate_pct"] = split["positive_count"] / split["row_count"] * 100

    parts = {name: data.loc[data["ml_split"].eq(name)] for name in [
        "train", "validation_select", "calibration", "test"
    ]}
    x = {name: frame[features] for name, frame in parts.items()}
    y = {name: frame[TARGET].astype(int).to_numpy() for name, frame in parts.items()}

    validation_rows = []
    logistic_candidates = {}
    for c in [0.01, 0.1, 1.0, 10.0]:
        model = Pipeline([
            ("prep", make_preprocessor()),
            ("model", LogisticRegression(C=c, max_iter=3000, solver="lbfgs")),
        ])
        model.fit(x["train"], y["train"])
        pred = model.predict_proba(x["validation_select"])[:, 1]
        row = {"model_family": "logistic", "candidate": f"C={c}"} | metrics(y["validation_select"], pred)
        validation_rows.append(row)
        logistic_candidates[c] = model
    best_c = max(
        [r for r in validation_rows if r["model_family"] == "logistic"],
        key=lambda r: r["pr_auc"],
    )["candidate"]
    best_logistic = logistic_candidates[float(best_c.split("=")[1])]

    prep = make_preprocessor()
    x_train_t = prep.fit_transform(x["train"])
    x_select_t = prep.transform(x["validation_select"])
    x_cal_t = prep.transform(x["calibration"])
    x_test_t = prep.transform(x["test"])
    ratio = (len(y["train"]) - y["train"].sum()) / y["train"].sum()
    xgb_candidates = {}
    configs = [
        ("depth2_unweighted", 2, 0.05, 300, 1.0),
        ("depth3_unweighted", 3, 0.05, 300, 1.0),
        ("depth2_sqrt_weight", 2, 0.05, 300, float(np.sqrt(ratio))),
    ]
    for name, depth, lr, trees, weight in configs:
        model = XGBClassifier(
            n_estimators=trees, max_depth=depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_lambda=5.0, scale_pos_weight=weight, eval_metric="logloss",
            random_state=42, n_jobs=4,
        )
        model.fit(x_train_t, y["train"])
        pred = model.predict_proba(x_select_t)[:, 1]
        validation_rows.append({"model_family": "xgboost", "candidate": name} | metrics(y["validation_select"], pred))
        xgb_candidates[name] = model
    best_xgb_name = max(
        [r for r in validation_rows if r["model_family"] == "xgboost"],
        key=lambda r: r["pr_auc"],
    )["candidate"]
    best_xgb = xgb_candidates[best_xgb_name]

    log_cal_raw = best_logistic.predict_proba(x["calibration"])[:, 1]
    xgb_cal_raw = best_xgb.predict_proba(x_cal_t)[:, 1]
    log_platt = platt_fit(log_cal_raw, y["calibration"])
    xgb_platt = platt_fit(xgb_cal_raw, y["calibration"])
    log_test_raw = best_logistic.predict_proba(x["test"])[:, 1]
    xgb_test_raw = best_xgb.predict_proba(x_test_t)[:, 1]
    predictions = {
        "constant_baseline": np.full(len(y["test"]), y["train"].mean()),
        "logistic_raw": log_test_raw,
        "logistic_platt": platt_predict(log_platt, log_test_raw),
        "xgboost_raw": xgb_test_raw,
        "xgboost_platt": platt_predict(xgb_platt, xgb_test_raw),
    }
    test_rows, lifts, calibrations = [], [], []
    for name, pred in predictions.items():
        test_rows.append({"model_name": name} | metrics(y["test"], pred))
        lifts.extend(lift_rows(name, y["test"], pred))
        calibrations.append(calibration_rows(name, y["test"], pred))

    test_pred = parts["test"][[
        "customer_unique_id", "order_purchase_timestamp", "order_time_segment", TARGET
    ]].copy()
    for name, pred in predictions.items():
        test_pred[f"score_{name}"] = pred
    segment_rows = []
    for model_name in ["logistic_raw", "xgboost_platt"]:
        score = predictions[model_name]
        temp = test_pred.copy()
        temp["score"] = score
        for segment, group in temp.groupby("order_time_segment"):
            segment_rows.append({
                "model_name": model_name, "segment_name": segment,
                "customer_count": len(group), "positive_count": int(group[TARGET].sum()),
                "positive_rate_pct": group[TARGET].mean() * 100,
                "pr_auc": average_precision_score(group[TARGET], group["score"])
                if group[TARGET].sum() > 0 else np.nan,
                "mean_predicted_probability_pct": group["score"].mean() * 100,
            })

    feature_names = prep.get_feature_names_out()
    importance = pd.concat([
        pd.DataFrame({
            "model_name": "logistic_raw",
            "feature": best_logistic.named_steps["prep"].get_feature_names_out(),
            "importance": best_logistic.named_steps["model"].coef_[0],
            "importance_type": "coefficient",
        }),
        pd.DataFrame({
            "model_name": "xgboost_raw", "feature": feature_names,
            "importance": best_xgb.feature_importances_, "importance_type": "gain_proxy",
        }),
    ], ignore_index=True)
    importance["absolute_importance"] = importance["importance"].abs()
    importance = importance.sort_values(["model_name", "absolute_importance"], ascending=[True, False])

    checks = pd.DataFrame([
        ("eligible_row_count", len(data), 86772, len(data) == 86772),
        ("positive_count", int(data[TARGET].sum()), 535, data[TARGET].sum() == 535),
        ("split_count_sum", int(split["row_count"].sum()), len(data), split["row_count"].sum() == len(data)),
        ("all_splits_have_positives", int((split["positive_count"] > 0).sum()), 4, (split["positive_count"] > 0).all()),
        ("test_prediction_count", len(test_pred), len(parts["test"]), len(test_pred) == len(parts["test"])),
        ("prediction_range_violations", int(sum(((p < 0) | (p > 1) | ~np.isfinite(p)).sum() for p in predictions.values())), 0, all(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() for p in predictions.values())),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    split.to_csv(SPLIT_PATH, index=False)
    pd.DataFrame(validation_rows).to_csv(VALIDATION_PATH, index=False)
    pd.DataFrame(test_rows).to_csv(TEST_PATH, index=False)
    pd.DataFrame(lifts).to_csv(LIFT_PATH, index=False)
    pd.concat(calibrations, ignore_index=True).to_csv(CALIBRATION_PATH, index=False)
    pd.DataFrame(segment_rows).to_csv(SEGMENT_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    test_pred.to_csv(PREDICTION_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    joblib.dump(best_logistic, MODELS / "olist_repurchase_logistic.joblib")
    joblib.dump({"preprocessor": prep, "model": best_xgb, "platt": xgb_platt}, MODELS / "olist_repurchase_xgboost_platt.joblib")
    (MODELS / "olist_repurchase_model_metadata.json").write_text(json.dumps({
        "target": TARGET, "best_logistic_candidate": best_c,
        "best_xgboost_candidate": best_xgb_name,
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSPLITS\n", split.to_string(index=False))
    print("\nVALIDATION\n", pd.DataFrame(validation_rows).to_string(index=False))
    print("\nTEST\n", pd.DataFrame(test_rows).to_string(index=False))
    print("\nLIFT\n", pd.DataFrame(lifts).to_string(index=False))
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

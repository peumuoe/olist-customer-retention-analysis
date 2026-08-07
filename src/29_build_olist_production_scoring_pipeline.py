"""Refit the frozen Olist logistic model and build reproducible scoring artifacts.

The frozen model settings come from the untouched-test evaluation in script 26.
This script does not create a new performance estimate. It refits the selected
specification on all outcome-mature history and produces historical backfill
scores, deterministic targeting ranks, and drift-monitoring reference tables.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "olist_asof_crm_segment_customer_mart.csv"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"

SCORE_PATH = ROOT / "data" / "processed" / "olist_repurchase_production_scores.csv"
THRESHOLD_PATH = TABLES / "olist_ml_production_score_thresholds.csv"
REFERENCE_PATH = TABLES / "olist_ml_monitoring_reference.csv"
POLICY_PATH = TABLES / "olist_ml_monitoring_policy.csv"
SUMMARY_PATH = TABLES / "olist_ml_production_pipeline_build_summary.csv"
MODEL_PATH = MODELS / "olist_repurchase_logistic_production.joblib"
METADATA_PATH = MODELS / "olist_repurchase_production_metadata.json"

TARGET = "target_repeat_over_24h_within_30d"
MODEL_VERSION = "olist_logistic_c001_20260724"
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
FEATURES = NUMERIC + CATEGORICAL


def make_model():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100)),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=0.01, max_iter=3000, solver="lbfgs")),
    ])


def load_feature_frame():
    data = pd.read_csv(MART_PATH, low_memory=False)
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
    return data.sort_values(
        ["order_purchase_timestamp", "customer_unique_id"]
    ).reset_index(drop=True)


def add_deterministic_ranks(scores):
    ranked = scores.sort_values(
        ["repurchase_score", "customer_unique_id"], ascending=[False, True]
    ).copy()
    ranked["global_rank"] = np.arange(1, len(ranked) + 1)
    ranked["global_rank_pct"] = ranked["global_rank"] / len(ranked)
    ranked["global_top_5pct_flag"] = ranked["global_rank_pct"].le(0.05)
    ranked["global_top_10pct_flag"] = ranked["global_rank_pct"].le(0.10)
    ranked["global_top_20pct_flag"] = ranked["global_rank_pct"].le(0.20)

    ranked["value_band_rank"] = ranked.groupby("order_value_band").cumcount() + 1
    ranked["value_band_count"] = ranked.groupby("order_value_band")[
        "customer_unique_id"
    ].transform("size")
    ranked["value_band_rank_pct"] = (
        ranked["value_band_rank"] / ranked["value_band_count"]
    )
    ranked["value_band_top_10pct_flag"] = ranked["value_band_rank_pct"].le(0.10)
    ranked["recommended_pilot_flag"] = (
        ranked["order_value_band"].isin(["Q3", "Q4_high"])
        & ranked["value_band_top_10pct_flag"]
    )
    ranked["targeting_policy"] = np.select(
        [
            ranked["recommended_pilot_flag"],
            ranked["order_value_band"].isin(["Q1_low", "Q2"]),
        ],
        ["q3_q4_within_band_top10_pilot", "monitor_only_q1_q2"],
        default="not_selected",
    )
    return ranked.sort_values(
        ["order_purchase_timestamp", "customer_unique_id"]
    ).reset_index(drop=True)


def build_thresholds(scores):
    rows = []
    for fraction in [0.05, 0.10, 0.20]:
        selected = scores.loc[scores["global_rank_pct"].le(fraction)]
        rows.append({
            "threshold_scope": "global",
            "group_value": "all",
            "top_fraction": fraction,
            "population_count": len(scores),
            "selected_count": len(selected),
            "minimum_selected_score": selected["repurchase_score"].min(),
            "tie_break_rule": "score_desc_then_customer_unique_id_asc",
            "operating_status": "comparison_scenario",
        })
    for band, group in scores.groupby("order_value_band", sort=True):
        selected = group.loc[group["value_band_top_10pct_flag"]]
        rows.append({
            "threshold_scope": "order_value_band",
            "group_value": band,
            "top_fraction": 0.10,
            "population_count": len(group),
            "selected_count": len(selected),
            "minimum_selected_score": selected["repurchase_score"].min(),
            "tie_break_rule": "score_desc_then_customer_unique_id_asc",
            "operating_status": (
                "pilot_candidate" if band in ["Q3", "Q4_high"] else "monitor_only"
            ),
        })
    return pd.DataFrame(rows)


def build_monitoring_reference(train, scores):
    rows = []
    score_decile = pd.qcut(
        scores["repurchase_score"].rank(method="first"),
        10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    score_counts = score_decile.value_counts(sort=False)
    for level, count in score_counts.items():
        rows.append({
            "reference_type": "score_decile",
            "dimension": "repurchase_score",
            "level": str(level),
            "customer_count": int(count),
            "population_share": count / len(scores),
        })
    for dimension in ["order_value_band", "customer_state", "primary_category"]:
        counts = scores[dimension].fillna("unknown").value_counts(dropna=False)
        for level, count in counts.items():
            rows.append({
                "reference_type": "feature_distribution",
                "dimension": dimension,
                "level": str(level),
                "customer_count": int(count),
                "population_share": count / len(scores),
            })
    outcome = train.groupby("order_value_band", dropna=False)[TARGET].agg(
        ["size", "sum", "mean"]
    )
    for level, row in outcome.iterrows():
        rows.append({
            "reference_type": "mature_outcome",
            "dimension": "order_value_band",
            "level": str(level),
            "customer_count": int(row["size"]),
            "population_share": float(row["mean"]),
        })
    return pd.DataFrame(rows)


def build_monitoring_policy():
    return pd.DataFrame([
        {
            "monitor": "data_schema_and_required_features",
            "warning_rule": "any missing required column",
            "alert_rule": "scoring must stop",
            "action": "repair input contract before scoring",
        },
        {
            "monitor": "score_or_feature_population_stability_index",
            "warning_rule": "PSI >= 0.10",
            "alert_rule": "PSI >= 0.25",
            "action": "inspect period, channel, state and category mix",
        },
        {
            "monitor": "mature_outcome_calibration_ratio",
            "warning_rule": "outside 0.67 to 1.50",
            "alert_rule": "outside 0.50 to 2.00",
            "action": "recalibrate or suspend probability interpretation",
        },
        {
            "monitor": "mature_outcome_lift_at_10pct",
            "warning_rule": "below 1.20",
            "alert_rule": "below 1.00",
            "action": "suspend score-based targeting and investigate",
        },
        {
            "monitor": "top10_group_representation_ratio",
            "warning_rule": "outside 0.67 to 1.50",
            "alert_rule": "outside 0.50 to 2.00",
            "action": "review targeting concentration before launch",
        },
    ])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    data = load_feature_frame()
    train = data.loc[data["eligible_30d"].eq(True)].copy()
    model = make_model()
    model.fit(train[FEATURES], train[TARGET].astype(int))

    raw_scores = data[[
        "customer_unique_id", "order_id", "order_purchase_timestamp",
        "eligible_30d", "order_value_band", "customer_state", "primary_category",
        "order_time_segment",
    ]].copy()
    raw_scores["repurchase_score"] = model.predict_proba(data[FEATURES])[:, 1]
    raw_scores["model_version"] = MODEL_VERSION
    raw_scores["score_type"] = "historical_backfill_not_live_campaign_list"
    scores = add_deterministic_ranks(raw_scores)

    thresholds = build_thresholds(scores)
    reference = build_monitoring_reference(train, scores)
    policy = build_monitoring_policy()
    feature_names = model.named_steps["prep"].get_feature_names_out()
    latest_timestamp = data["order_purchase_timestamp"].max()
    checks = pd.DataFrame([
        ("source_row_count", len(data), 93358, len(data) == 93358),
        ("mature_training_row_count", len(train), 86772, len(train) == 86772),
        ("training_positive_count", int(train[TARGET].sum()), 535, int(train[TARGET].sum()) == 535),
        ("score_row_count", len(scores), len(data), len(scores) == len(data)),
        ("unique_customer_count", scores["customer_unique_id"].nunique(), len(scores), scores["customer_unique_id"].nunique() == len(scores)),
        ("score_range_violations", int((~scores["repurchase_score"].between(0, 1)).sum()), 0, scores["repurchase_score"].between(0, 1).all()),
        ("missing_scores", int(scores["repurchase_score"].isna().sum()), 0, scores["repurchase_score"].notna().all()),
        ("recommended_value_band_violations", int((scores["recommended_pilot_flag"] & ~scores["order_value_band"].isin(["Q3", "Q4_high"])).sum()), 0, (~scores["recommended_pilot_flag"] | scores["order_value_band"].isin(["Q3", "Q4_high"])).all()),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    joblib.dump(model, MODEL_PATH)
    scores.to_csv(SCORE_PATH, index=False)
    thresholds.to_csv(THRESHOLD_PATH, index=False)
    reference.to_csv(REFERENCE_PATH, index=False)
    policy.to_csv(POLICY_PATH, index=False)
    checks.to_csv(SUMMARY_PATH, index=False)
    METADATA_PATH.write_text(json.dumps({
        "model_version": MODEL_VERSION,
        "created_date": "2026-07-24",
        "model_family": "logistic_regression",
        "frozen_hyperparameters": {"C": 0.01, "penalty": "l2", "solver": "lbfgs"},
        "target": TARGET,
        "training_population": "eligible_30d == True",
        "training_row_count": len(train),
        "training_positive_count": int(train[TARGET].sum()),
        "training_data_start": str(train["order_purchase_timestamp"].min()),
        "training_data_end": str(train["order_purchase_timestamp"].max()),
        "source_data_cutoff": str(latest_timestamp),
        "score_population": "all historical first-order customer rows",
        "score_output_warning": "historical backfill; not a current live campaign list",
        "probability_warning": "use primarily as a ranking score; campaign incrementality is unmeasured",
        "recommended_pilot_policy": "within-value-band top 10% for Q3 and Q4_high",
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "transformed_feature_count": len(feature_names),
        "tie_break_rule": "score descending, then customer_unique_id ascending",
        "validated_test_metrics_source": "outputs/tables/olist_ml_test_model_summary.csv",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCHECKS\n", checks.to_string(index=False))
    print("\nTHRESHOLDS\n", thresholds.to_string(index=False))
    print("\nPOLICY COUNTS\n", scores["targeting_policy"].value_counts().to_string())
    print("\nSCORE SUMMARY\n", scores["repurchase_score"].describe().to_string())


if __name__ == "__main__":
    main()

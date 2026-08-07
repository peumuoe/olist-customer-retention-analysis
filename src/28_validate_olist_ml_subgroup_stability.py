"""Evaluate Olist logistic model performance and selection across subgroups."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PREDICTION_PATH = PROCESSED / "olist_ml_test_predictions.csv"
MART_PATH = PROCESSED / "olist_asof_crm_segment_customer_mart.csv"
PERFORMANCE_PATH = TABLES / "olist_ml_subgroup_performance.csv"
SELECTION_PATH = TABLES / "olist_ml_top10_subgroup_selection.csv"
SUMMARY_PATH = TABLES / "olist_ml_subgroup_stability_summary.csv"
CHECK_PATH = TABLES / "olist_ml_subgroup_validation_build_summary.csv"

TARGET = "target_repeat_over_24h_within_30d"
SCORE = "score_logistic_raw"
DIMENSIONS = [
    "order_time_segment", "order_value_band",
    "seller_exploration_category_flag", "customer_state", "primary_category",
]


def top10_metrics(y, p):
    n = max(1, int(np.ceil(len(y) * 0.10)))
    chosen = np.argsort(-p)[:n]
    captured = int(np.asarray(y)[chosen].sum())
    rate = captured / n
    return captured, captured / np.sum(y), rate / np.mean(y)


def load_data():
    pred = pd.read_csv(PREDICTION_PATH)
    mart = pd.read_csv(
        MART_PATH,
        usecols=["customer_unique_id", "order_value_band",
                 "seller_exploration_category_flag", "customer_state", "primary_category"],
        low_memory=False,
    )
    data = pred.merge(mart, on="customer_unique_id", how="left", validate="one_to_one")
    for column in DIMENSIONS:
        data[column] = data[column].astype("string").fillna("unknown")
    data["global_top10_flag"] = False
    top_n = int(np.ceil(len(data) * 0.10))
    top_index = data[SCORE].nlargest(top_n).index
    data.loc[top_index, "global_top10_flag"] = True
    return data, top_n


def build_performance(data):
    rows = []
    for dimension in DIMENSIONS:
        for value, group in data.groupby(dimension, dropna=False):
            y = group[TARGET].astype(int).to_numpy()
            p = group[SCORE].to_numpy()
            positives = int(y.sum())
            if positives > 0:
                captured, recall, lift = top10_metrics(y, p)
                pr_auc = average_precision_score(y, p)
                roc_auc = roc_auc_score(y, p) if positives < len(y) else np.nan
            else:
                captured, recall, lift, pr_auc, roc_auc = 0, np.nan, np.nan, np.nan, np.nan
            observed = y.mean()
            predicted = p.mean()
            rows.append({
                "subgroup_dimension": dimension, "subgroup_value": str(value),
                "customer_count": len(group), "positive_count": positives,
                "observed_positive_rate_pct": observed * 100,
                "mean_predicted_probability_pct": predicted * 100,
                "calibration_ratio_predicted_to_observed": predicted / observed if observed else np.nan,
                "pr_auc": pr_auc, "roc_auc": roc_auc,
                "within_group_top10_selected_count": int(np.ceil(len(group) * 0.10)),
                "within_group_top10_captured_positive_count": captured,
                "within_group_recall_at_10pct": recall,
                "within_group_lift_at_10pct": lift,
                "evidence_tier": (
                    "adequate_descriptive" if len(group) >= 500 and positives >= 5
                    else "sparse_do_not_conclude"
                ),
            })
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(6)
    return result


def build_selection(data, top_n):
    rows = []
    for dimension in DIMENSIONS:
        for value, group in data.groupby(dimension, dropna=False):
            selected = int(group["global_top10_flag"].sum())
            base_share = len(group) / len(data)
            selected_share = selected / top_n
            rows.append({
                "subgroup_dimension": dimension, "subgroup_value": str(value),
                "customer_count": len(group), "base_customer_share_pct": base_share * 100,
                "global_top10_selected_count": selected,
                "global_top10_composition_pct": selected_share * 100,
                "within_subgroup_selection_rate_pct": selected / len(group) * 100,
                "representation_ratio_top10_vs_base": selected_share / base_share,
                "evidence_tier": "adequate_descriptive" if len(group) >= 500 else "sparse_do_not_conclude",
            })
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(6)
    return result


def main():
    data, top_n = load_data()
    performance = build_performance(data)
    selection = build_selection(data, top_n)
    adequate = performance.loc[performance["evidence_tier"].eq("adequate_descriptive")]
    summary = pd.DataFrame([{
        "test_customer_count": len(data),
        "test_positive_count": int(data[TARGET].sum()),
        "global_top10_customer_count": top_n,
        "subgroup_row_count": len(performance),
        "adequate_descriptive_row_count": len(adequate),
        "sparse_row_count": int(performance["evidence_tier"].eq("sparse_do_not_conclude").sum()),
        "adequate_groups_lift10_above_1": int(adequate["within_group_lift_at_10pct"].gt(1).sum()),
        "adequate_groups_lift10_at_or_below_1": int(adequate["within_group_lift_at_10pct"].le(1).sum()),
        "adequate_groups_calibration_ratio_below_0_5": int(adequate["calibration_ratio_predicted_to_observed"].lt(0.5).sum()),
        "adequate_groups_calibration_ratio_above_2": int(adequate["calibration_ratio_predicted_to_observed"].gt(2).sum()),
        "max_adequate_representation_ratio": selection.loc[selection["evidence_tier"].eq("adequate_descriptive"), "representation_ratio_top10_vs_base"].max(),
        "min_adequate_representation_ratio": selection.loc[selection["evidence_tier"].eq("adequate_descriptive"), "representation_ratio_top10_vs_base"].min(),
    }])
    summary[summary.select_dtypes(include="number").columns] = summary.select_dtypes(include="number").round(6)
    checks = pd.DataFrame([
        ("test_customer_count", len(data), 17355, len(data) == 17355),
        ("test_positive_count", int(data[TARGET].sum()), 108, data[TARGET].sum() == 108),
        ("missing_joined_feature_count", int(data[DIMENSIONS].isna().sum().sum()), 0, data[DIMENSIONS].notna().all().all()),
        ("global_top10_count", int(data["global_top10_flag"].sum()), top_n, data["global_top10_flag"].sum() == top_n),
        ("dimension_count", performance["subgroup_dimension"].nunique(), len(DIMENSIONS), performance["subgroup_dimension"].nunique() == len(DIMENSIONS)),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    performance.to_csv(PERFORMANCE_PATH, index=False)
    selection.to_csv(SELECTION_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    print("\nSUMMARY\n", summary.to_string(index=False))
    print("\nEXECUTION SEGMENTS\n", performance.loc[performance["subgroup_dimension"].eq("order_time_segment")].to_string(index=False))
    print("\nVALUE BANDS\n", performance.loc[performance["subgroup_dimension"].eq("order_value_band")].to_string(index=False))
    print("\nTOP REPRESENTATION\n", selection.loc[selection["evidence_tier"].eq("adequate_descriptive")].sort_values("representation_ratio_top10_vs_base", ascending=False).head(15).to_string(index=False))
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

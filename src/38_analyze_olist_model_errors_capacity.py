"""Audit frozen-model errors and CRM capacity/cost sensitivity."""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PRED_PATH = PROCESSED / "olist_ml_test_predictions.csv"
MART_PATH = PROCESSED / "olist_first_purchase_customer_mart.csv"
TARGET = "target_repeat_over_24h_within_30d"
SCORE = "score_logistic_raw"
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def load_frame() -> pd.DataFrame:
    pred = pd.read_csv(PRED_PATH, parse_dates=["order_purchase_timestamp"])
    features = pd.read_csv(
        MART_PATH,
        usecols=[
            "customer_unique_id", "payment_value_sum", "freight_value_sum",
            "item_count", "seller_count", "payment_installments_max",
            "order_value_band", "primary_category", "customer_state",
            "primary_payment_type",
        ],
        low_memory=False,
    )
    return pred.merge(features, on="customer_unique_id", how="left", validate="one_to_one")


def add_top10_error_group(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values([SCORE, "customer_unique_id"], ascending=[False, True]).copy()
    selected_n = int(np.ceil(len(result) * 0.10))
    result["selected_top10"] = False
    result.iloc[:selected_n, result.columns.get_loc("selected_top10")] = True
    y = result[TARGET].astype(bool)
    result["error_group"] = np.select(
        [result["selected_top10"] & y, result["selected_top10"] & ~y,
         ~result["selected_top10"] & y, ~result["selected_top10"] & ~y],
        ["TP_selected_repurchase", "FP_selected_no_repurchase",
         "FN_not_selected_repurchase", "TN_not_selected_no_repurchase"],
        default="unclassified",
    )
    return result


def confusion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["error_group"].value_counts()
    tp = int(counts.get("TP_selected_repurchase", 0))
    fp = int(counts.get("FP_selected_no_repurchase", 0))
    fn = int(counts.get("FN_not_selected_repurchase", 0))
    tn = int(counts.get("TN_not_selected_no_repurchase", 0))
    rows = [
        ("TP", tp, "selected and repurchased"),
        ("FP", fp, "selected but did not repurchase"),
        ("FN", fn, "not selected but repurchased"),
        ("TN", tn, "not selected and did not repurchase"),
        ("precision", tp / (tp + fp), "share of selected customers who repurchased"),
        ("recall", tp / (tp + fn), "share of repurchasers captured"),
        ("false_discovery_rate", fp / (tp + fp), "share of selected without observed repurchase"),
        ("specificity", tn / (tn + fp), "share of non-repurchasers not selected"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def numeric_error_profile(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "payment_value_sum", "freight_value_sum", "item_count",
        "seller_count", "payment_installments_max", SCORE,
    ]
    rows = []
    for group, subset in frame.groupby("error_group"):
        for feature in numeric:
            rows.append({
                "error_group": group,
                "feature": feature,
                "customer_count": len(subset),
                "mean": subset[feature].mean(),
                "median": subset[feature].median(),
                "p25": subset[feature].quantile(0.25),
                "p75": subset[feature].quantile(0.75),
            })
    return pd.DataFrame(rows)


def missed_positive_profile(frame: pd.DataFrame) -> pd.DataFrame:
    positives = frame.loc[frame[TARGET].astype(bool)].copy()
    dimensions = [
        "order_value_band", "primary_category", "customer_state",
        "primary_payment_type", "order_time_segment",
    ]
    rows = []
    for dimension in dimensions:
        grouped = positives.assign(
            segment_value=positives[dimension].astype("string").fillna("unknown")
        ).groupby("segment_value").agg(
            positive_count=(TARGET, "size"),
            captured_count=("selected_top10", "sum"),
        ).reset_index()
        grouped["missed_count"] = grouped["positive_count"] - grouped["captured_count"]
        grouped["capture_rate_pct"] = grouped["captured_count"] / grouped["positive_count"] * 100
        grouped["dimension"] = dimension
        grouped["evidence_flag"] = np.where(
            grouped["positive_count"] >= 10, "descriptive_10plus_positives", "sparse_under_10_positives"
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)[[
        "dimension", "segment_value", "positive_count", "captured_count",
        "missed_count", "capture_rate_pct", "evidence_flag",
    ]].sort_values(["dimension", "missed_count"], ascending=[True, False])


def capacity_frontier(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "global_test":
        eligible = frame.sort_values(
            [SCORE, "customer_unique_id"], ascending=[False, True]
        ).copy()
    elif scope == "q3q4_test":
        eligible = frame.loc[frame["order_value_band"].isin(["Q3", "Q4_high"])].copy()
        eligible = eligible.sort_values(
            ["order_value_band", SCORE, "customer_unique_id"],
            ascending=[True, False, True],
        )
        eligible["band_rank"] = eligible.groupby("order_value_band").cumcount() + 1
        eligible["band_count"] = eligible.groupby("order_value_band")["customer_unique_id"].transform("size")
    else:
        raise ValueError(scope)
    base = eligible[TARGET].mean()
    total_positive = int(eligible[TARGET].sum())
    rows = []
    previous_ids: set[str] = set()
    for fraction in FRACTIONS:
        if scope == "global_test":
            n = int(np.ceil(len(eligible) * fraction))
            selected = eligible.iloc[:n]
        else:
            selected = eligible.loc[
                eligible["band_rank"] <= np.ceil(eligible["band_count"] * fraction)
            ]
            n = len(selected)
        selected_ids = set(selected["customer_unique_id"])
        tranche = eligible.loc[eligible["customer_unique_id"].isin(selected_ids - previous_ids)]
        captured = int(selected[TARGET].sum())
        tranche_n = len(tranche)
        tranche_positive = int(tranche[TARGET].sum())
        rows.append({
            "scope": scope,
            "selected_fraction": fraction,
            "eligible_customer_count": len(eligible),
            "total_positive_count": total_positive,
            "selected_customer_count": n,
            "captured_positive_count": captured,
            "precision_pct": captured / n * 100,
            "recall_pct": captured / total_positive * 100,
            "lift": (captured / n) / base,
            "contacts_per_observed_repurchase": n / captured if captured else np.nan,
            "marginal_tranche_customer_count": tranche_n,
            "marginal_tranche_positive_count": tranche_positive,
            "marginal_tranche_positive_rate_pct": tranche_positive / tranche_n * 100 if tranche_n else np.nan,
        })
        previous_ids = selected_ids
    return pd.DataFrame(rows)


def cost_sensitivity() -> pd.DataFrame:
    rows = []
    for contact_cost in [0.10, 0.50, 1.00, 2.00]:
        for contribution_margin in [20.0, 50.0, 100.0]:
            rows.append({
                "contact_cost_brl": contact_cost,
                "incremental_contribution_per_conversion_brl": contribution_margin,
                "break_even_incremental_conversion_rate_pct_point": contact_cost / contribution_margin * 100,
                "break_even_incremental_conversions_per_1000_contacts": contact_cost / contribution_margin * 1000,
                "allowed_interpretation": "minimum causal uplift required to cover contact cost",
                "prohibited_interpretation": "expected uplift, expected profit, or observed campaign ROI",
            })
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame = add_top10_error_group(load_frame())
    confusion = confusion_summary(frame)
    numeric = numeric_error_profile(frame)
    missed = missed_positive_profile(frame)
    capacity = pd.concat(
        [capacity_frontier(frame, "global_test"), capacity_frontier(frame, "q3q4_test")],
        ignore_index=True,
    )
    costs = cost_sensitivity()

    count = confusion.set_index("metric")["value"]
    top10 = capacity.query("scope == 'global_test' and selected_fraction == 0.10").iloc[0]
    checks = pd.DataFrame([
        ("test_rows", len(frame), 17355, len(frame) == 17355),
        ("unique_customers", frame["customer_unique_id"].nunique(), 17355, frame["customer_unique_id"].nunique() == 17355),
        ("positive_count", int(frame[TARGET].sum()), 108, int(frame[TARGET].sum()) == 108),
        ("tp_count", int(count["TP"]), 19, int(count["TP"]) == 19),
        ("fn_count", int(count["FN"]), 89, int(count["FN"]) == 89),
        ("top10_selected", int(top10["selected_customer_count"]), 1736, int(top10["selected_customer_count"]) == 1736),
        ("top10_lift", float(top10["lift"]), 1.758753, abs(float(top10["lift"]) - 1.758753) < 0.000001),
        ("capacity_rows", len(capacity), 18, len(capacity) == 18),
        ("cost_scenario_rows", len(costs), 12, len(costs) == 12),
        ("missing_scores", int(frame[SCORE].isna().sum()), 0, frame[SCORE].notna().all()),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    confusion.to_csv(TABLES / "olist_model_top10_error_summary.csv", index=False, encoding="utf-8-sig")
    numeric.to_csv(TABLES / "olist_model_error_numeric_profile.csv", index=False, encoding="utf-8-sig")
    missed.to_csv(TABLES / "olist_model_missed_positive_profile.csv", index=False, encoding="utf-8-sig")
    capacity.to_csv(TABLES / "olist_crm_capacity_frontier.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(TABLES / "olist_crm_cost_break_even_sensitivity.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(TABLES / "olist_model_error_capacity_build_summary.csv", index=False, encoding="utf-8-sig")

    print("Top10 error summary")
    print(confusion.to_string(index=False))
    print("\nCapacity frontier")
    print(capacity.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

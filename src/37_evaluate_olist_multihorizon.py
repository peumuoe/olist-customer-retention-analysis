"""Evaluate the frozen 30-day Olist score across 30/60/90-day outcomes.

This is a research appendix for CRM horizon choice. It does not retrain or
select a new model, and 60/90-day results are not final model claims.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = PROCESSED / "olist_ml_test_predictions.csv"
RETENTION = PROCESSED / "customer_retention_metrics.csv"
MART = PROCESSED / "olist_first_purchase_customer_mart.csv"
HORIZONS = [30, 60, 90]
SCORE = "score_logistic_raw"


def top_fraction_metrics(y: np.ndarray, score: np.ndarray, fraction: float = 0.10):
    selected_count = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-score, kind="stable")
    selected = order[:selected_count]
    positives = int(y.sum())
    captured = int(y[selected].sum())
    selected_rate = captured / selected_count
    base_rate = positives / len(y)
    return {
        "selected_count": selected_count,
        "captured_positive_count": captured,
        "recall_at_10pct": captured / positives if positives else np.nan,
        "lift_at_10pct": selected_rate / base_rate if base_rate else np.nan,
    }


def prepare_frame() -> tuple[pd.DataFrame, pd.Timestamp]:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["order_purchase_timestamp"])
    retention = pd.read_csv(
        RETENTION,
        usecols=[
            "customer_unique_id", "first_purchase_timestamp",
            "days_to_second_purchase",
        ],
        parse_dates=["first_purchase_timestamp"],
    )
    value = pd.read_csv(
        MART,
        usecols=["customer_unique_id", "order_value_band"],
        low_memory=False,
    )
    frame = (
        predictions.merge(retention, on="customer_unique_id", how="left", validate="one_to_one")
        .merge(value, on="customer_unique_id", how="left", validate="one_to_one")
    )
    observation_cutoff = pd.Timestamp("2018-08-29 15:00:37")
    for horizon in HORIZONS:
        frame[f"eligible_{horizon}d"] = (
            frame["first_purchase_timestamp"]
            <= observation_cutoff - pd.Timedelta(days=horizon)
        )
        # Frozen operational definition: the second purchase itself must occur
        # over 24 hours after the first and no later than the horizon.
        frame[f"target_over24h_{horizon}d"] = (
            frame["days_to_second_purchase"].gt(1)
            & frame["days_to_second_purchase"].le(horizon)
        )
    return frame, observation_cutoff


def evaluate_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        eligible = frame.loc[frame[f"eligible_{horizon}d"]].copy()
        y = eligible[f"target_over24h_{horizon}d"].astype(int).to_numpy()
        score = eligible[SCORE].to_numpy(float)
        rank = top_fraction_metrics(y, score)
        rows.append(
            {
                "horizon_days": horizon,
                "evaluation_scope": "frozen_independent_test_subset",
                "eligible_customer_count": len(eligible),
                "positive_count": int(y.sum()),
                "positive_rate_pct": y.mean() * 100,
                "mean_frozen_30d_score_pct": score.mean() * 100,
                "score_mean_to_outcome_ratio": score.mean() / y.mean() if y.mean() else np.nan,
                "pr_auc": average_precision_score(y, score),
                "pr_auc_over_prevalence": average_precision_score(y, score) / y.mean(),
                "roc_auc": roc_auc_score(y, score),
                "brier_score_using_30d_score": brier_score_loss(y, score),
                **rank,
            }
        )
    return pd.DataFrame(rows)


def evaluate_common_90d_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold customers fixed so only the outcome horizon changes."""
    common = frame.loc[frame["eligible_90d"]].copy()
    rows = []
    for horizon in HORIZONS:
        y = common[f"target_over24h_{horizon}d"].astype(int).to_numpy()
        score = common[SCORE].to_numpy(float)
        rows.append(
            {
                "horizon_days": horizon,
                "common_customer_count": len(common),
                "positive_count": int(y.sum()),
                "positive_rate_pct": y.mean() * 100,
                "pr_auc": average_precision_score(y, score),
                "pr_auc_over_prevalence": average_precision_score(y, score) / y.mean(),
                "roc_auc": roc_auc_score(y, score),
                **top_fraction_metrics(y, score),
            }
        )
    return pd.DataFrame(rows)


def build_q3q4_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    rows = []
    selected_sets = {}
    for horizon in HORIZONS:
        eligible = frame.loc[
            frame[f"eligible_{horizon}d"]
            & frame["order_value_band"].isin(["Q3", "Q4_high"])
        ].copy()
        eligible = eligible.sort_values(
            ["order_value_band", SCORE, "customer_unique_id"],
            ascending=[True, False, True],
        )
        eligible["band_rank"] = eligible.groupby("order_value_band").cumcount() + 1
        eligible["band_count"] = eligible.groupby("order_value_band")["customer_unique_id"].transform("size")
        eligible["selected"] = eligible["band_rank"] <= np.ceil(eligible["band_count"] * 0.10)
        target = f"target_over24h_{horizon}d"
        selected = eligible.loc[eligible["selected"]]
        selected_sets[horizon] = set(selected["customer_unique_id"])
        total_positive = int(eligible[target].sum())
        captured = int(selected[target].sum())
        rows.append(
            {
                "horizon_days": horizon,
                "eligible_q3q4_customer_count": len(eligible),
                "eligible_q3q4_positive_count": total_positive,
                "selected_customer_count": len(selected),
                "selected_positive_count": captured,
                "selected_positive_rate_pct": selected[target].mean() * 100,
                "eligible_q3q4_positive_rate_pct": eligible[target].mean() * 100,
                "lift_within_q3q4": (
                    selected[target].mean() / eligible[target].mean()
                    if eligible[target].mean() else np.nan
                ),
                "recall_within_q3q4": captured / total_positive if total_positive else np.nan,
            }
        )
    return pd.DataFrame(rows), selected_sets


def build_overlap(selected_sets: dict[int, set[str]]) -> pd.DataFrame:
    rows = []
    for left, right in [(30, 60), (30, 90), (60, 90)]:
        intersection = selected_sets[left] & selected_sets[right]
        union = selected_sets[left] | selected_sets[right]
        rows.append(
            {
                "left_horizon_days": left,
                "right_horizon_days": right,
                "left_selected_count": len(selected_sets[left]),
                "right_selected_count": len(selected_sets[right]),
                "intersection_count": len(intersection),
                "jaccard_overlap": len(intersection) / len(union) if union else np.nan,
                "right_contained_in_left_pct": (
                    len(intersection) / len(selected_sets[right]) * 100
                    if selected_sets[right] else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame, observation_cutoff = prepare_frame()
    summary = evaluate_horizons(frame)
    common = evaluate_common_90d_cohort(frame)
    q3q4, selected_sets = build_q3q4_policy(frame)
    overlap = build_overlap(selected_sets)

    row30 = summary.set_index("horizon_days").loc[30]
    checks = pd.DataFrame(
        [
            ("test_row_count", len(frame), 17355, len(frame) == 17355),
            ("unique_test_customers", frame["customer_unique_id"].nunique(), 17355, frame["customer_unique_id"].nunique() == 17355),
            ("observation_cutoff", str(observation_cutoff), "2018-08-29 15:00:37", observation_cutoff == pd.Timestamp("2018-08-29 15:00:37")),
            ("30d_eligible_count", int(summary.set_index("horizon_days").loc[30, "eligible_customer_count"]), 17355, int(row30["eligible_customer_count"]) == 17355),
            ("30d_positive_count_matches_frozen_test", int(row30["positive_count"]), 108, int(row30["positive_count"]) == 108),
            ("30d_lift_matches_frozen_test", float(row30["lift_at_10pct"]), 1.758753, abs(float(row30["lift_at_10pct"]) - 1.758753) < 0.000001),
            ("eligibility_non_increasing", int(summary["eligible_customer_count"].is_monotonic_decreasing), 1, summary["eligible_customer_count"].is_monotonic_decreasing),
            ("positive_horizon_non_decreasing_on_common_90d_subset", int(all(
                frame.loc[frame["eligible_90d"], f"target_over24h_{left}d"].sum()
                <= frame.loc[frame["eligible_90d"], f"target_over24h_{right}d"].sum()
                for left, right in [(30, 60), (60, 90)]
            )), 1, all(
                frame.loc[frame["eligible_90d"], f"target_over24h_{left}d"].sum()
                <= frame.loc[frame["eligible_90d"], f"target_over24h_{right}d"].sum()
                for left, right in [(30, 60), (60, 90)]
            )),
            ("missing_scores", int(frame[SCORE].isna().sum()), 0, frame[SCORE].notna().all()),
            ("q3q4_policy_only", int((~frame.loc[frame["order_value_band"].isin(["Q3", "Q4_high"]), "order_value_band"].isin(["Q3", "Q4_high"])).sum()), 0, True),
        ],
        columns=["check_name", "actual_value", "expected_value", "passed"],
    )
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    summary.to_csv(TABLES / "olist_multihorizon_frozen_score_summary.csv", index=False, encoding="utf-8-sig")
    common.to_csv(TABLES / "olist_multihorizon_common_cohort_summary.csv", index=False, encoding="utf-8-sig")
    q3q4.to_csv(TABLES / "olist_multihorizon_q3q4_policy_summary.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(TABLES / "olist_multihorizon_candidate_overlap.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(TABLES / "olist_multihorizon_build_summary.csv", index=False, encoding="utf-8-sig")
    print("Frozen score by horizon")
    print(summary.to_string(index=False))
    print("\nCommon 90-day-mature cohort")
    print(common.to_string(index=False))
    print("\nQ3/Q4 policy")
    print(q3q4.to_string(index=False))
    print("\nCandidate overlap")
    print(overlap.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

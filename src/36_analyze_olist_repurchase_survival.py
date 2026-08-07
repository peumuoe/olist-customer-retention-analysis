"""Describe time to Olist repurchase with audited Kaplan-Meier estimates.

This is a descriptive research appendix. It does not replace the frozen
30-day KPI or the final ML model, and it does not estimate CRM causal effects.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "reports" / "figures"
ORDERS_PATH = PROCESSED / "retention_orders.csv"
CUTOFFS = [1, 7, 14, 30, 60, 90, 180, 365]


def build_customer_survival(orders: pd.DataFrame, minimum_gap_days: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    orders = orders.sort_values(
        ["customer_unique_id", "order_purchase_timestamp", "order_id"]
    ).copy()
    cutoff = orders["order_purchase_timestamp"].max()
    first = (
        orders.groupby("customer_unique_id", as_index=False)
        .first()[["customer_unique_id", "order_purchase_timestamp"]]
        .rename(columns={"order_purchase_timestamp": "first_purchase_timestamp"})
    )
    candidates = orders.merge(first, on="customer_unique_id", how="left")
    candidates["_order_rank"] = (
        candidates.groupby("customer_unique_id").cumcount().add(1)
    )
    candidates["gap_days"] = (
        candidates["order_purchase_timestamp"]
        - candidates["first_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    qualifying_mask = candidates["_order_rank"].ge(2)
    if minimum_gap_days > 0:
        # Keep the frozen operational target definition: the second purchase
        # itself must occur after 24 hours. A later third order does not
        # retrospectively turn a <=24h second order into a positive target.
        qualifying_mask &= (
            candidates["_order_rank"].eq(2)
            & candidates["gap_days"].gt(minimum_gap_days)
        )
    qualifying = (
        candidates.loc[qualifying_mask]
        .groupby("customer_unique_id", as_index=False)
        .first()[["customer_unique_id", "order_purchase_timestamp", "gap_days"]]
        .rename(columns={"order_purchase_timestamp": "event_timestamp"})
    )
    customer = first.merge(qualifying, on="customer_unique_id", how="left")
    customer["event_observed"] = customer["event_timestamp"].notna()
    customer["censor_days"] = (
        cutoff - customer["first_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    customer["duration_days"] = np.where(
        customer["event_observed"], customer["gap_days"], customer["censor_days"]
    )
    customer["first_purchase_cohort"] = np.select(
        [
            customer["first_purchase_timestamp"] < pd.Timestamp("2017-07-01"),
            customer["first_purchase_timestamp"] < pd.Timestamp("2018-01-01"),
            customer["first_purchase_timestamp"] < pd.Timestamp("2018-07-01"),
        ],
        ["through_2017H1", "2017H2", "2018H1"],
        default="2018H2_partial",
    )
    return customer, cutoff


def km_at_horizons(customer: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    duration = customer["duration_days"].to_numpy(float)
    event = customer["event_observed"].to_numpy(bool)
    event_times = np.sort(np.unique(duration[event]))
    survival = 1.0
    greenwood = 0.0
    rows = []
    event_index = 0

    for horizon in horizons:
        while event_index < len(event_times) and event_times[event_index] <= horizon:
            time = event_times[event_index]
            at_risk = int(np.sum(duration >= time))
            events = int(np.sum(event & np.isclose(duration, time)))
            if at_risk > 0:
                survival *= 1 - events / at_risk
                if at_risk > events:
                    greenwood += events / (at_risk * (at_risk - events))
            event_index += 1

        if 0 < survival < 1 and greenwood > 0:
            log_survival = math.log(survival)
            se_loglog = math.sqrt(greenwood) / abs(log_survival)
            lower_s = math.exp(-math.exp(math.log(-log_survival) + 1.96 * se_loglog))
            upper_s = math.exp(-math.exp(math.log(-log_survival) - 1.96 * se_loglog))
        else:
            lower_s = upper_s = survival
        rows.append(
            {
                "horizon_days": horizon,
                "at_risk_after_horizon": int(np.sum(duration > horizon)),
                "cumulative_repurchase_pct": (1 - survival) * 100,
                "ci95_lower_pct": (1 - upper_s) * 100,
                "ci95_upper_pct": (1 - lower_s) * 100,
            }
        )
    return pd.DataFrame(rows)


def fixed_window_rate(customer: pd.DataFrame, horizon: int) -> tuple[int, int, float]:
    eligible = customer["censor_days"] >= horizon
    events = eligible & customer["event_observed"] & (customer["duration_days"] <= horizon)
    n = int(eligible.sum())
    e = int(events.sum())
    return n, e, e / n * 100 if n else np.nan


def timing_distribution(customer: pd.DataFrame) -> pd.DataFrame:
    observed = customer.loc[customer["event_observed"], "duration_days"]
    labels = [
        "0_to_1d", "over_1_to_7d", "over_7_to_14d", "over_14_to_30d",
        "over_30_to_60d", "over_60_to_90d", "over_90_to_180d",
        "over_180_to_365d", "over_365d",
    ]
    buckets = pd.cut(
        observed,
        bins=[-np.inf, 1, 7, 14, 30, 60, 90, 180, 365, np.inf],
        labels=labels,
        right=True,
    )
    result = buckets.value_counts(sort=False).rename_axis("timing_bucket").reset_index(name="event_count")
    result["share_of_observed_events_pct"] = result["event_count"] / len(observed) * 100
    return result


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(ORDERS_PATH, parse_dates=["order_purchase_timestamp"])

    all_outputs = []
    customer_outputs = {}
    for definition, minimum_gap in [("official_any_later_order", 0.0), ("operational_over_24h", 1.0)]:
        customer, cutoff = build_customer_survival(orders, minimum_gap)
        customer_outputs[definition] = customer
        km = km_at_horizons(customer, CUTOFFS)
        km.insert(0, "definition", definition)
        fixed = [fixed_window_rate(customer, h) for h in CUTOFFS]
        km["fixed_eligible_customer_count"] = [x[0] for x in fixed]
        km["fixed_repeat_customer_count"] = [x[1] for x in fixed]
        km["fixed_window_repurchase_pct"] = [x[2] for x in fixed]
        km["km_minus_fixed_pct_point"] = (
            km["cumulative_repurchase_pct"] - km["fixed_window_repurchase_pct"]
        )
        all_outputs.append(km)

    survival_summary = pd.concat(all_outputs, ignore_index=True)

    cohort_rows = []
    for definition, customer in customer_outputs.items():
        for cohort, group in customer.groupby("first_purchase_cohort", observed=True):
            estimates = km_at_horizons(group, [30, 90, 180])
            for row in estimates.to_dict("records"):
                cohort_rows.append(
                    {
                        "definition": definition,
                        "first_purchase_cohort": cohort,
                        "customer_count": len(group),
                        "observed_event_count": int(group["event_observed"].sum()),
                        **row,
                    }
                )
    cohort_summary = pd.DataFrame(cohort_rows)
    timing = timing_distribution(customer_outputs["official_any_later_order"])

    official_30 = survival_summary.loc[
        (survival_summary["definition"] == "official_any_later_order")
        & (survival_summary["horizon_days"] == 30)
    ].iloc[0]
    operational_30 = survival_summary.loc[
        (survival_summary["definition"] == "operational_over_24h")
        & (survival_summary["horizon_days"] == 30)
    ].iloc[0]
    within_day = timing.loc[timing["timing_bucket"].astype(str) == "0_to_1d"].iloc[0]

    checks = [
        ("input_order_rows", len(orders), len(orders) == 96478),
        ("unique_customers", orders["customer_unique_id"].nunique(), orders["customer_unique_id"].nunique() == 93358),
        ("observation_cutoff", str(cutoff), cutoff == pd.Timestamp("2018-08-29 15:00:37")),
        ("official_events", int(customer_outputs["official_any_later_order"]["event_observed"].sum()), True),
        ("operational_over24h_events", int(customer_outputs["operational_over_24h"]["event_observed"].sum()), True),
        ("negative_duration_rows", int(sum((x["duration_days"] < 0).sum() for x in customer_outputs.values())), all((x["duration_days"] >= 0).all() for x in customer_outputs.values())),
        ("official_30d_fixed_rate_matches_kpi", float(official_30["fixed_window_repurchase_pct"]), abs(float(official_30["fixed_window_repurchase_pct"]) - 1.5823) < 0.00006),
        ("survival_probability_monotonic", 1, all(g["cumulative_repurchase_pct"].is_monotonic_increasing for _, g in survival_summary.groupby("definition"))),
        ("operational_below_official_30d", float(operational_30["cumulative_repurchase_pct"]), operational_30["cumulative_repurchase_pct"] < official_30["cumulative_repurchase_pct"]),
    ]
    build_summary = pd.DataFrame(checks, columns=["check_name", "value", "passed"])
    if not build_summary["passed"].all():
        raise ValueError(build_summary.loc[~build_summary["passed"]].to_string(index=False))

    survival_summary.to_csv(TABLES / "olist_repurchase_survival_summary.csv", index=False, encoding="utf-8-sig")
    cohort_summary.to_csv(TABLES / "olist_repurchase_survival_cohort_summary.csv", index=False, encoding="utf-8-sig")
    timing.to_csv(TABLES / "olist_repurchase_timing_distribution.csv", index=False, encoding="utf-8-sig")
    build_summary.to_csv(TABLES / "olist_repurchase_survival_build_summary.csv", index=False, encoding="utf-8-sig")

    print(survival_summary.to_string(index=False))
    print("\nObserved-event timing:")
    print(timing.to_string(index=False))
    print("\nChecks:")
    print(build_summary.to_string(index=False))
    print(
        f"\n30d official KM={official_30['cumulative_repurchase_pct']:.4f}%, "
        f"fixed={official_30['fixed_window_repurchase_pct']:.4f}%; "
        f"operational KM={operational_30['cumulative_repurchase_pct']:.4f}%; "
        f"within-day observed events={int(within_day['event_count'])}."
    )


if __name__ == "__main__":
    main()

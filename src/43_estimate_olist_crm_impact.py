"""Estimate the expected impact of the Q3·Q4 CRM message and its break-even point.

Inputs come from earlier steps (src/06, src/30). Message cost and contribution
margin are NOT in the data: they are explicit planning assumptions, varied as
scenarios so the decision does not hinge on one guess.
"""

from math import sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

# From src/30 (olist_crm_experiment_power.csv): control rate observed on 905 comparable customers.
BASE_RATE = 16 / 905
BASE_N, BASE_POS = 905, 16
MDE_RELATIVE = 0.7087

# Planning assumptions (not observed in the data).
RELATIVE_UPLIFTS = [0.10, 0.20, 0.30, 0.50, MDE_RELATIVE]
MARGINS = {"margin_10pct": 0.10, "margin_20pct": 0.20}
MESSAGE_COSTS_BRL = {"email_0.05": 0.05, "sms_0.30": 0.30}


def wilson_interval(pos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = pos / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def second_order_value() -> float:
    """Mean payment of >24h-to-30d second orders placed by Q3·Q4 first-order customers."""
    orders = pd.read_csv(PROCESSED / "retention_orders.csv", parse_dates=["order_purchase_timestamp"])
    pay = pd.read_csv(RAW / "olist_order_payments_dataset.csv").groupby("order_id")["payment_value"].sum()
    orders["payment_value"] = orders["order_id"].map(pay)
    first = orders.loc[orders["customer_order_number"].eq(1)].set_index("customer_unique_id")
    second = orders.loc[orders["customer_order_number"].eq(2)].set_index("customer_unique_id")
    joined = first.join(second, lsuffix="_1", rsuffix="_2", how="inner")
    gap = (joined["order_purchase_timestamp_2"] - joined["order_purchase_timestamp_1"]).dt.total_seconds() / 86400
    q50 = first["payment_value"].median()
    mask = gap.gt(1) & gap.le(30) & joined["payment_value_1"].ge(q50)
    return float(joined.loc[mask, "payment_value_2"].mean())


def annual_candidates() -> int:
    """Candidates whose first order falls in the last 12 months of the candidate period."""
    cand = pd.read_csv(PROCESSED / "olist_crm_experiment_assignment_dry_run.csv",
                       parse_dates=["order_purchase_timestamp"])
    end = cand["order_purchase_timestamp"].max()
    return int(cand["order_purchase_timestamp"].gt(end - pd.Timedelta(days=365)).sum())


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aov = second_order_value()
    yearly = annual_candidates()
    low, high = wilson_interval(BASE_POS, BASE_N)

    rows = []
    for uplift in RELATIVE_UPLIFTS:
        for m_name, margin in MARGINS.items():
            for c_name, cost in MESSAGE_COSTS_BRL.items():
                extra = yearly * BASE_RATE * uplift
                contribution = extra * aov * margin
                spend = yearly * cost
                rows.append({
                    "relative_uplift_pct": round(uplift * 100, 1),
                    "margin_assumption": m_name,
                    "message_cost_assumption": c_name,
                    "customers_messaged_per_year": yearly,
                    "incremental_repurchasers": round(extra, 1),
                    "incremental_revenue_brl": round(extra * aov),
                    "incremental_contribution_brl": round(contribution),
                    "message_cost_brl": round(spend),
                    "net_brl": round(contribution - spend),
                    "roi": round((contribution - spend) / spend, 2),
                })
    scenarios = pd.DataFrame(rows)

    # Relative uplift at which contribution exactly covers message cost.
    breakeven = pd.DataFrame([
        {"margin_assumption": m_name, "message_cost_assumption": c_name,
         "breakeven_relative_uplift_pct": round(100 * cost / (BASE_RATE * aov * margin), 1),
         "breakeven_at_base_low_pct": round(100 * cost / (low * aov * margin), 1),
         "breakeven_at_base_high_pct": round(100 * cost / (high * aov * margin), 1)}
        for m_name, margin in MARGINS.items()
        for c_name, cost in MESSAGE_COSTS_BRL.items()
    ])
    facts = {"base_rate": BASE_RATE, "base_rate_95ci": (low, high),
             "second_order_value_brl": aov, "annual_candidates": yearly}
    return scenarios, breakeven, facts


def main() -> None:
    scenarios, breakeven, facts = build()
    scenarios.to_csv(TABLES / "olist_crm_impact_scenarios.csv", index=False, encoding="utf-8-sig")
    breakeven.to_csv(TABLES / "olist_crm_impact_breakeven.csv", index=False, encoding="utf-8-sig")
    print(facts)
    print(scenarios.loc[scenarios["margin_assumption"].eq("margin_20pct")
                        & scenarios["message_cost_assumption"].eq("email_0.05")].to_string(index=False))
    print(breakeven.to_string(index=False))


if __name__ == "__main__":
    main()

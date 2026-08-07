"""Design a reproducible, customer-level randomized CRM experiment dry run."""

from pathlib import Path
from statistics import NormalDist
from math import ceil, log, sqrt
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "data" / "processed" / "olist_repurchase_production_scores.csv"
SUBGROUP_PATH = ROOT / "outputs" / "tables" / "olist_ml_subgroup_performance.csv"
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"

ASSIGNMENT_PATH = PROCESSED / "olist_crm_experiment_assignment_dry_run.csv"
BALANCE_PATH = TABLES / "olist_crm_experiment_randomization_balance.csv"
POWER_PATH = TABLES / "olist_crm_experiment_power_scenarios.csv"
KPI_PATH = TABLES / "olist_crm_experiment_kpi_registry.csv"
PROTOCOL_PATH = TABLES / "olist_crm_experiment_protocol.csv"
SUMMARY_PATH = TABLES / "olist_crm_experiment_design_build_summary.csv"

EXPERIMENT_ID = "olist_crm_q3q4_top10_pilot_v1_20260724"
ALPHA = 0.05
POWER = 0.80


def stable_hash(value):
    return hashlib.sha256(f"{EXPERIMENT_ID}|{value}".encode("utf-8")).hexdigest()


def state_bucket(series):
    return series.where(series.isin(["SP", "RJ", "MG"]), "Other")


def assign_arms(candidates):
    frame = candidates.copy()
    frame["state_bucket"] = state_bucket(frame["customer_state"])
    frame["score_quintile"] = pd.qcut(
        frame["repurchase_score"].rank(method="first"),
        5,
        labels=["Q1_low_score", "Q2", "Q3", "Q4", "Q5_high_score"],
    )
    frame["randomization_stratum"] = (
        frame["order_value_band"].astype(str)
        + "|" + frame["state_bucket"].astype(str)
        + "|" + frame["score_quintile"].astype(str)
    )
    frame["randomization_hash"] = frame["customer_unique_id"].map(stable_hash)
    frame = frame.sort_values(
        ["randomization_stratum", "randomization_hash", "customer_unique_id"]
    ).reset_index(drop=True)
    frame["stratum_sequence"] = frame.groupby("randomization_stratum").cumcount()
    frame["stratum_start_offset"] = frame["randomization_stratum"].map(
        lambda x: int(stable_hash(x)[0], 16) % 2
    )
    frame["experiment_arm"] = np.where(
        (frame["stratum_sequence"] + frame["stratum_start_offset"]) % 2 == 0,
        "treatment",
        "control",
    )

    # Resolve any accumulated one-row imbalance from odd-sized strata while
    # preserving deterministic assignment and no more than one switch per stratum.
    counts = frame["experiment_arm"].value_counts()
    target_treatment = len(frame) // 2
    excess = int(counts.get("treatment", 0) - target_treatment)
    if excess != 0:
        from_arm = "treatment" if excess > 0 else "control"
        to_arm = "control" if excess > 0 else "treatment"
        switch_count = abs(excess)
        odd_strata = (
            frame.groupby("randomization_stratum").size()
            .loc[lambda x: x.mod(2).eq(1)]
            .index
        )
        switchable = frame.loc[
            frame["randomization_stratum"].isin(odd_strata)
            & frame["experiment_arm"].eq(from_arm)
        ].sort_values("randomization_hash").groupby(
            "randomization_stratum", as_index=False
        ).head(1).head(switch_count)
        frame.loc[switchable.index, "experiment_arm"] = to_arm

    frame["assignment_status"] = "dry_run_not_sent"
    frame["randomization_unit"] = "customer_unique_id"
    frame["planned_trigger"] = "24_to_48h_after_first_order"
    frame["analysis_population"] = "intention_to_treat"
    return frame


def categorical_balance(frame, column):
    rows = []
    shares = pd.crosstab(
        frame[column], frame["experiment_arm"], normalize="columns"
    )
    counts = pd.crosstab(frame[column], frame["experiment_arm"])
    for level in shares.index:
        treatment_share = shares.loc[level].get("treatment", 0.0)
        control_share = shares.loc[level].get("control", 0.0)
        rows.append({
            "balance_type": "categorical",
            "variable": column,
            "level": str(level),
            "treatment_count": int(counts.loc[level].get("treatment", 0)),
            "control_count": int(counts.loc[level].get("control", 0)),
            "treatment_value": treatment_share,
            "control_value": control_share,
            "absolute_difference": abs(treatment_share - control_share),
            "standardized_mean_difference": np.nan,
        })
    return rows


def numeric_balance(frame, column):
    treatment = frame.loc[frame["experiment_arm"].eq("treatment"), column]
    control = frame.loc[frame["experiment_arm"].eq("control"), column]
    pooled_sd = sqrt((treatment.var(ddof=1) + control.var(ddof=1)) / 2)
    smd = (treatment.mean() - control.mean()) / pooled_sd if pooled_sd else 0.0
    return {
        "balance_type": "numeric",
        "variable": column,
        "level": "mean",
        "treatment_count": len(treatment),
        "control_count": len(control),
        "treatment_value": treatment.mean(),
        "control_value": control.mean(),
        "absolute_difference": abs(treatment.mean() - control.mean()),
        "standardized_mean_difference": smd,
    }


def build_balance(frame):
    rows = [numeric_balance(frame, "repurchase_score")]
    for column in [
        "order_value_band", "state_bucket", "score_quintile", "primary_category"
    ]:
        rows.extend(categorical_balance(frame, column))
    return pd.DataFrame(rows)


def normal_power(n_per_arm, p_control, p_treatment):
    z_alpha = NormalDist().inv_cdf(1 - ALPHA / 2)
    standard_error = sqrt(
        p_control * (1 - p_control) / n_per_arm
        + p_treatment * (1 - p_treatment) / n_per_arm
    )
    z_effect = abs(p_treatment - p_control) / standard_error
    return (
        NormalDist().cdf(z_effect - z_alpha)
        + NormalDist().cdf(-z_effect - z_alpha)
    )


def required_n_per_arm(p_control, p_treatment):
    z_alpha = NormalDist().inv_cdf(1 - ALPHA / 2)
    z_power = NormalDist().inv_cdf(POWER)
    pooled = (p_control + p_treatment) / 2
    numerator = (
        z_alpha * sqrt(2 * pooled * (1 - pooled))
        + z_power * sqrt(
            p_control * (1 - p_control)
            + p_treatment * (1 - p_treatment)
        )
    ) ** 2
    return ceil(numerator / (p_treatment - p_control) ** 2)


def detectable_rate(n_per_arm, baseline):
    low, high = baseline, min(0.50, baseline + 0.20)
    while normal_power(n_per_arm, baseline, high) < POWER and high < 0.99:
        high = min(0.99, high * 1.5)
    for _ in range(100):
        middle = (low + high) / 2
        if normal_power(n_per_arm, baseline, middle) < POWER:
            low = middle
        else:
            high = middle
    return high


def validated_baseline():
    subgroups = pd.read_csv(SUBGROUP_PATH)
    rows = subgroups.loc[
        subgroups["subgroup_dimension"].eq("order_value_band")
        & subgroups["subgroup_value"].isin(["Q3", "Q4_high"])
    ]
    selected = int(rows["within_group_top10_selected_count"].sum())
    positives = int(rows["within_group_top10_captured_positive_count"].sum())
    return positives / selected, positives, selected


def build_power(frame):
    baseline, baseline_positives, baseline_n = validated_baseline()
    n_per_arm = min(frame["experiment_arm"].value_counts())
    rows = []
    for relative_effect in [0.20, 0.30, 0.50, 0.75, 1.00]:
        treatment_rate = baseline * (1 + relative_effect)
        required = required_n_per_arm(baseline, treatment_rate)
        rows.append({
            "scenario": f"relative_uplift_{int(relative_effect * 100)}pct",
            "baseline_rate": baseline,
            "treatment_rate": treatment_rate,
            "absolute_effect_pct_point": (treatment_rate - baseline) * 100,
            "relative_effect": relative_effect,
            "available_n_per_arm": n_per_arm,
            "required_n_per_arm": required,
            "required_total_n": required * 2,
            "achieved_power_with_available_n": normal_power(
                n_per_arm, baseline, treatment_rate
            ),
            "baseline_source_positive_count": baseline_positives,
            "baseline_source_selected_count": baseline_n,
            "planning_status": (
                "adequately_powered" if required <= n_per_arm
                else "underpowered"
            ),
        })
    mde_rate = detectable_rate(n_per_arm, baseline)
    rows.append({
        "scenario": "available_sample_mde",
        "baseline_rate": baseline,
        "treatment_rate": mde_rate,
        "absolute_effect_pct_point": (mde_rate - baseline) * 100,
        "relative_effect": mde_rate / baseline - 1,
        "available_n_per_arm": n_per_arm,
        "required_n_per_arm": n_per_arm,
        "required_total_n": n_per_arm * 2,
        "achieved_power_with_available_n": POWER,
        "baseline_source_positive_count": baseline_positives,
        "baseline_source_selected_count": baseline_n,
        "planning_status": "minimum_detectable_effect",
    })
    return pd.DataFrame(rows)


def build_kpi_registry():
    return pd.DataFrame([
        {
            "kpi_role": "primary",
            "kpi_name": "30d_repeat_purchase_rate_over_24h",
            "unit": "customer_rate",
            "definition": "customers with >=1 delivered repeat order >24h and <=30d / randomized customers",
            "analysis": "intention_to_treat risk_difference and risk_ratio with 95% CI",
            "required_source": "orders",
            "decision_rule": "report estimate and CI; no success claim from p-value alone",
        },
        {
            "kpi_role": "secondary",
            "kpi_name": "30d_repeat_order_count_per_customer",
            "unit": "orders_per_randomized_customer",
            "definition": "paid repeat orders >24h and <=30d / randomized customers",
            "analysis": "mean difference with bootstrap CI",
            "required_source": "orders",
            "decision_rule": "interpret only with primary KPI",
        },
        {
            "kpi_role": "secondary",
            "kpi_name": "30d_observed_repeat_revenue_per_customer",
            "unit": "currency_per_randomized_customer",
            "definition": "observed repeat payment value within outcome window / randomized customers",
            "analysis": "mean difference with bootstrap CI",
            "required_source": "orders_and_payments",
            "decision_rule": "not future CLV",
        },
        {
            "kpi_role": "guardrail",
            "kpi_name": "unsubscribe_rate",
            "unit": "customer_rate",
            "definition": "customers unsubscribing within 7d after planned send / delivered customers",
            "analysis": "risk difference",
            "required_source": "crm_delivery_and_consent_log",
            "decision_rule": "must be populated before launch",
        },
        {
            "kpi_role": "guardrail",
            "kpi_name": "complaint_rate",
            "unit": "customer_rate",
            "definition": "spam or customer complaints / delivered customers",
            "analysis": "risk difference",
            "required_source": "crm_complaint_log",
            "decision_rule": "must be populated before launch",
        },
        {
            "kpi_role": "guardrail",
            "kpi_name": "incremental_contribution_after_contact_cost",
            "unit": "currency_per_randomized_customer",
            "definition": "incremental contribution margin minus incentive and contact cost",
            "analysis": "treatment-control difference",
            "required_source": "margin_incentive_and_contact_cost",
            "decision_rule": "cannot calculate from current Olist data alone",
        },
    ])


def build_protocol(frame, power):
    mde = power.loc[power["scenario"].eq("available_sample_mde")].iloc[0]
    return pd.DataFrame([
        ("experiment_id", EXPERIMENT_ID),
        ("design_status", "dry_run_design_not_launched"),
        ("objective", "estimate incremental effect of one CRM contact on 30d repeat purchase"),
        ("randomization_unit", "customer_unique_id"),
        ("eligible_population", "Q3 or Q4_high and within-value-band model-score top 10%"),
        ("assignment_ratio", "1:1 treatment vs control"),
        ("stratification", "order_value_band x state_bucket x score_quintile"),
        ("treatment", "one non-discount related-product/category discovery message; channel must be fixed before launch"),
        ("control", "no incremental campaign contact; normal transactional communication continues"),
        ("planned_trigger", "24 to 48 hours after first order"),
        ("outcome_window", ">24 hours through 30 days after first order"),
        ("primary_analysis", "intention-to-treat"),
        ("multiple_testing", "one primary KPI; secondary and subgroup results are exploratory"),
        ("available_candidate_count", str(len(frame))),
        ("available_n_per_arm", str(int(mde["available_n_per_arm"]))),
        ("planning_baseline_rate", f"{mde['baseline_rate']:.8f}"),
        ("available_sample_mde_absolute_pct_point", f"{mde['absolute_effect_pct_point']:.6f}"),
        ("available_sample_mde_relative", f"{mde['relative_effect']:.6f}"),
        ("launch_gate_1", "confirm legal consent, contact channel, frequency cap, and suppression rules"),
        ("launch_gate_2", "freeze message, benefit, trigger, sample-size target, and analysis code"),
        ("launch_gate_3", "verify CRM delivery, unsubscribe, complaint, cost, and order joins"),
        ("stopping_rule", "do not stop early for observed conversion unless a pre-specified sequential design is adopted"),
        ("interpretation", "model lift is not campaign incrementality; only randomized comparison estimates effect"),
    ], columns=["protocol_field", "protocol_value"])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SCORE_PATH, low_memory=False)
    candidates = scores.loc[scores["recommended_pilot_flag"].eq(True)].copy()
    assignment = assign_arms(candidates)
    balance = build_balance(assignment)
    power = build_power(assignment)
    kpis = build_kpi_registry()
    protocol = build_protocol(assignment, power)

    arm_counts = assignment["experiment_arm"].value_counts()
    max_cat_diff = balance.loc[
        balance["balance_type"].eq("categorical"), "absolute_difference"
    ].max()
    score_smd = balance.loc[
        balance["variable"].eq("repurchase_score"),
        "standardized_mean_difference",
    ].abs().iloc[0]
    checks = pd.DataFrame([
        ("candidate_count", len(assignment), 4667, len(assignment) == 4667),
        ("unique_customer_count", assignment["customer_unique_id"].nunique(), len(assignment), assignment["customer_unique_id"].nunique() == len(assignment)),
        ("arm_count_difference", abs(int(arm_counts["treatment"]) - int(arm_counts["control"])), 1, abs(int(arm_counts["treatment"]) - int(arm_counts["control"])) <= 1),
        ("non_q3_q4_candidates", int((~assignment["order_value_band"].isin(["Q3", "Q4_high"])).sum()), 0, assignment["order_value_band"].isin(["Q3", "Q4_high"]).all()),
        ("missing_assignments", int(assignment["experiment_arm"].isna().sum()), 0, assignment["experiment_arm"].notna().all()),
        ("score_smd_abs_below_0_10", float(score_smd), 0.10, score_smd < 0.10),
        ("max_categorical_share_diff_below_0_05", float(max_cat_diff), 0.05, max_cat_diff < 0.05),
        ("primary_kpi_count", int(kpis["kpi_role"].eq("primary").sum()), 1, kpis["kpi_role"].eq("primary").sum() == 1),
    ], columns=["check_name", "actual_value", "expected_or_limit", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    assignment.to_csv(ASSIGNMENT_PATH, index=False)
    balance.to_csv(BALANCE_PATH, index=False)
    power.to_csv(POWER_PATH, index=False)
    kpis.to_csv(KPI_PATH, index=False)
    protocol.to_csv(PROTOCOL_PATH, index=False)
    checks.to_csv(SUMMARY_PATH, index=False)

    print("\nARM COUNTS\n", arm_counts.to_string())
    print("\nPOWER\n", power.to_string(index=False))
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

"""Build the final decision-ready Olist CRM playbook from audited artifacts."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
EXECUTION = PROCESSED / "olist_crm_customer_execution_plan_dry_run.csv"
SCORES = PROCESSED / "olist_repurchase_production_scores.csv"


def build_population_summary(execution: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    candidate = scores.loc[scores["recommended_pilot_flag"].astype(bool)].copy()
    rows = [
        {
            "population_id": "P0_all_historical_candidates",
            "population_rule": "Q3 or Q4_high and within-value-band score top 10%",
            "historical_customer_count": len(candidate),
            "treatment_count_dry_run": int((execution["experiment_arm"] == "treatment").sum()),
            "control_count_dry_run": int((execution["experiment_arm"] == "control").sum()),
            "current_live_contact_authorized_count": int(execution["actual_send_authorized"].astype(bool).sum()),
        }
    ]
    for strategy, group in execution.groupby("content_strategy"):
        rows.append({
            "population_id": f"content_{strategy}",
            "population_rule": strategy,
            "historical_customer_count": len(group),
            "treatment_count_dry_run": int((group["experiment_arm"] == "treatment").sum()),
            "control_count_dry_run": int((group["experiment_arm"] == "control").sum()),
            "current_live_contact_authorized_count": int(group["actual_send_authorized"].astype(bool).sum()),
        })
    monitor = scores.loc[
        scores["value_band_top_10pct_flag"].astype(bool)
        & scores["order_value_band"].isin(["Q1_low", "Q2"])
    ]
    rows.append({
        "population_id": "monitor_q1_q2_top10",
        "population_rule": "Q1_low or Q2 and within-value-band score top 10%",
        "historical_customer_count": len(monitor),
        "treatment_count_dry_run": 0,
        "control_count_dry_run": 0,
        "current_live_contact_authorized_count": 0,
    })
    return pd.DataFrame(rows)


def build_playbook(population: pd.DataFrame) -> pd.DataFrame:
    counts = population.set_index("population_id")["historical_customer_count"]
    common = {
        "benefit": "none",
        "frequency": "maximum 1 experimental contact in 30 days",
        "primary_kpi": "ITT delivered repeat-order customer rate >24h and <=30d",
        "secondary_kpi": "60d and 90d cumulative repeat; repeat order value; delivery/click diagnostics",
        "guardrails": "unsubscribe; complaint; cancellation/refund; incremental contribution after contact cost",
        "suppression": "invalid consent; unsubscribe/DNC; no owned channel; cancelled first order; fraud/test; duplicate experiment; frequency cap",
        "causal_status": "hypothesis requiring randomized validation",
    }
    rows = [
        {
            "playbook_id": "PB-00",
            "priority": 0,
            "audience": "all historical pilot candidates",
            "historical_customer_count": int(counts["P0_all_historical_candidates"]),
            "role": "launch gate",
            "trigger": "before any experimental send",
            "message": "none",
            "decision": "BLOCK while consent, channel, frequency-cap and event/cost joins are unavailable",
            "execution_status": "blocked_not_executable_with_current_olist_data",
            **common,
        },
        {
            "playbook_id": "PB-01",
            "priority": 1,
            "audience": "Q3/Q4_high within-band top10; watches_gifts/auto/toys; treatment assignment",
            "historical_customer_count": int(counts["content_same_seller_and_category_exploration"]),
            "role": "treatment content branch",
            "trigger": "24 to 48 hours after first order after runtime eligibility checks",
            "message": "non-discount same-seller and same-category exploration",
            "decision": "randomize 1:1; send only if all launch gates pass",
            "execution_status": "designed_historical_dry_run_only",
            **common,
        },
        {
            "playbook_id": "PB-02",
            "priority": 1,
            "audience": "Q3/Q4_high within-band top10; other categories; treatment assignment",
            "historical_customer_count": int(counts["content_related_product_and_category_discovery"]),
            "role": "treatment content branch",
            "trigger": "24 to 48 hours after first order after runtime eligibility checks",
            "message": "non-discount related-product and category discovery",
            "decision": "randomize 1:1; send only if all launch gates pass",
            "execution_status": "designed_historical_dry_run_only",
            **common,
        },
        {
            "playbook_id": "PB-03",
            "priority": 1,
            "audience": "same Q3/Q4_high top10 randomized candidate population; control assignment",
            "historical_customer_count": int(population.iloc[0]["control_count_dry_run"]),
            "role": "randomized holdout",
            "trigger": "no experimental send",
            "message": "none; normal transactional communication continues",
            "decision": "retain assignment and full outcome tracking under ITT",
            "execution_status": "designed_historical_dry_run_only",
            **common,
        },
        {
            "playbook_id": "PB-04",
            "priority": 2,
            "audience": "Q1_low/Q2 within-band top10",
            "historical_customer_count": int(counts["monitor_q1_q2_top10"]),
            "role": "monitor only",
            "trigger": "none in first pilot",
            "message": "none",
            "decision": "do not include in first pilot; subgroup ranking was not sufficiently stable",
            "execution_status": "not_selected_for_initial_pilot",
            **common,
        },
        {
            "playbook_id": "PB-05",
            "priority": 3,
            "audience": "delivery-complete delayed/ontime descriptive segments",
            "historical_customer_count": 0,
            "role": "separate future hypothesis only",
            "trigger": "not part of the 24-48h order-time experiment",
            "message": "service-recovery or post-delivery discovery must be tested separately",
            "decision": "do not mix with first treatment because delivery experience is future information at order time",
            "execution_status": "excluded_from_initial_experiment",
            **common,
        },
    ]
    return pd.DataFrame(rows)


def build_decision_card(gates: pd.DataFrame) -> pd.DataFrame:
    status = gates["status"].value_counts().to_dict()
    return pd.DataFrame([
        ("current_project_decision", "DO_NOT_LAUNCH", "current Olist data contains no live consent, channel, send, complaint, or cost state"),
        ("recommended_business_next_action", "CONDITIONAL_RANDOMIZED_PILOT", "organizational owner must satisfy every blocked and pending launch gate first"),
        ("pilot_population", "Q3/Q4_high within-value-band score top 10%", "historical candidate count 4,667; not a live contact list"),
        ("treatment", "one non-discount discovery message", "seller/category module selected by first-order category"),
        ("primary_decision_window", ">24h to 30d", "60d and 90d remain secondary and cannot replace a failed primary"),
        ("scale_rule", "GO only if primary CI lower >0, contribution nonnegative, guardrails and data quality pass", "otherwise ITERATE, STOP, or NO DECISION"),
        ("ready_gate_count", str(status.get("ready", 0)), "historical design evidence only"),
        ("pending_gate_count", str(status.get("pending", 0)), "requires organizational decision"),
        ("blocked_gate_count", str(status.get("blocked", 0)), "cannot be resolved from current public Olist data"),
    ], columns=["decision_field", "decision_value", "interpretation"])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    execution = pd.read_csv(EXECUTION, low_memory=False)
    scores = pd.read_csv(SCORES, low_memory=False)
    gates = pd.read_csv(TABLES / "olist_crm_launch_gate_checklist.csv")
    population = build_population_summary(execution, scores)
    playbook = build_playbook(population)
    decision = build_decision_card(gates)

    p0 = population.set_index("population_id").loc["P0_all_historical_candidates"]
    checks = pd.DataFrame([
        ("historical_candidate_count", int(p0["historical_customer_count"]), 4667, int(p0["historical_customer_count"]) == 4667),
        ("dry_run_assignment_count", len(execution), 4667, len(execution) == 4667),
        ("treatment_plus_control", int(p0["treatment_count_dry_run"] + p0["control_count_dry_run"]), 4667, int(p0["treatment_count_dry_run"] + p0["control_count_dry_run"]) == 4667),
        ("live_authorized_count", int(p0["current_live_contact_authorized_count"]), 0, int(p0["current_live_contact_authorized_count"]) == 0),
        ("playbook_row_count", len(playbook), 6, len(playbook) == 6),
        ("discount_treatment_rows", int(playbook["benefit"].ne("none").sum()), 0, playbook["benefit"].eq("none").all()),
        ("primary_kpi_missing_rows", int(playbook["primary_kpi"].isna().sum()), 0, playbook["primary_kpi"].notna().all()),
        ("causal_warning_missing_rows", int((~playbook["causal_status"].str.contains("randomized")).sum()), 0, playbook["causal_status"].str.contains("randomized").all()),
        ("blocked_gate_count", int((gates["status"] == "blocked").sum()), 4, int((gates["status"] == "blocked").sum()) == 4),
        ("current_decision_do_not_launch", int(decision.iloc[0]["decision_value"] == "DO_NOT_LAUNCH"), 1, decision.iloc[0]["decision_value"] == "DO_NOT_LAUNCH"),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    population.to_csv(TABLES / "olist_final_crm_population_summary.csv", index=False, encoding="utf-8-sig")
    playbook.to_csv(TABLES / "olist_final_integrated_crm_playbook.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(TABLES / "olist_final_crm_decision_card.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(TABLES / "olist_final_crm_playbook_build_summary.csv", index=False, encoding="utf-8-sig")
    print("Population")
    print(population.to_string(index=False))
    print("\nPlaybook")
    print(playbook[["playbook_id", "audience", "historical_customer_count", "role", "execution_status"]].to_string(index=False))
    print("\nDecision")
    print(decision.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

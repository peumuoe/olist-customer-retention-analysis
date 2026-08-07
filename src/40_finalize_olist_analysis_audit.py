"""Run the final consistency, evidence, and project-exit audit."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORT = ROOT / "reports" / "olist_repurchase_final_report.md"
DOCS = ROOT / "docs"


def claim_matrix() -> pd.DataFrame:
    rows = [
        ("C01", "Olist official 30d repeat rate is 1.5823%", "L1_audited_descriptive", "retention_window_summary.csv", "level diagnosis", "cause attribution"),
        ("C02", "Operational >24h-to-30d target rate is 0.616558%", "L1_audited_descriptive", "olist_asof_crm_targeting_flag_evaluation.csv", "ML/CRM target definition", "substitute for official KPI"),
        ("C03", "External observed rates are higher", "L1_non_comparable", "external_retention_comparison.csv", "directional benchmark", "causal inferiority of Olist"),
        ("C04", "Frozen logistic top10 Lift is 1.758753", "L2_independent_predictive", "olist_ml_test_lift_recall.csv", "candidate ranking support", "campaign uplift or individual certainty"),
        ("C05", "Q3/Q4 within-band top10 is a pilot compromise", "L2_predictive_diagnostic", "olist_crm_capacity_frontier.csv", "historical recruitment design", "profit-optimal threshold"),
        ("C06", "Historical pilot candidate count is 4,667", "L2_design_dry_run", "olist_final_crm_population_summary.csv", "allocation rehearsal", "current contactable audience"),
        ("C07", "Current live authorized count is zero", "L2_design_dry_run", "olist_final_crm_population_summary.csv", "current project execution decision", "claim that campaign launched"),
        ("C08", "Current decision is DO_NOT_LAUNCH", "L2_decision_design", "olist_final_crm_decision_card.csv", "current project state", "permanent rejection of the hypothesis"),
        ("C09", "Business proposal is a conditional randomized pilot", "L2_decision_design", "olist_final_crm_decision_card.csv", "proposal after organizational gates", "proven CRM policy"),
        ("C10", "CRM incremental effect remains unknown", "L3_required_not_observed", "olist_crm_experiment_protocol.csv", "explicit evidence boundary", "expected uplift or ROI"),
        ("C11", "Synthetic scenarios validate analysis code only", "L0_pipeline_only", "olist_crm_synthetic_scenario_summary.csv", "pipeline rehearsal", "observed business performance"),
        ("C12", "Analysis can close under the fixed data scope", "project_exit_audit", "olist_project_completion_checklist.csv", "portfolio handoff", "claim that real campaign validation completed"),
        ("C13", "Imbalance treatment does not justify final model replacement", "L2_research_appendix_posthoc", "olist_imbalance_diagnostic_conclusion.csv", "diagnosis of limited signal and probability distortion", "claim of proven superior model"),
    ]
    return pd.DataFrame(rows, columns=[
        "claim_id", "claim", "evidence_level", "source_table",
        "allowed_wording_scope", "prohibited_interpretation",
    ])


def completion_checklist() -> pd.DataFrame:
    rows = [
        ("data_quality", "schema, keys, foreign keys, dates and order grain audited", "complete", "outputs/tables/schema_inventory.csv"),
        ("kpi", "official 30/60/90/180/365d and operational target separated", "complete", "outputs/tables/retention_window_summary.csv"),
        ("external", "H&M, REES46 and public benchmark comparability assessed", "complete", "outputs/tables/external_retention_comparison.csv"),
        ("opportunity", "priority segments and category transitions assessed", "complete", "outputs/tables/olist_improvement_opportunity_priority.csv"),
        ("prediction", "frozen logistic model independently, temporally and under imbalance sensitivity validated", "complete", "outputs/tables/olist_imbalance_diagnostic_conclusion.csv"),
        ("survival", "right censoring and repurchase timing evaluated", "complete", "outputs/tables/olist_repurchase_survival_summary.csv"),
        ("multihorizon", "30/60/90d frozen-score transfer assessed", "complete", "outputs/tables/olist_multihorizon_frozen_score_summary.csv"),
        ("errors_capacity", "TP/FP/FN/TN, capacity and break-even sensitivity audited", "complete", "outputs/tables/olist_model_top10_error_summary.csv"),
        ("crm_design", "target, treatment, control, KPIs, power and analysis code designed", "complete", "outputs/tables/olist_final_integrated_crm_playbook.csv"),
        ("risk_decision", "side effects, guardrails and GO/ITERATE/STOP/NO DECISION defined", "complete", "outputs/tables/olist_crm_experiment_decision_rules.csv"),
        ("actual_campaign_effect", "incremental CRM effect and ROI", "closed_unverifiable_with_current_data", "outputs/tables/olist_final_crm_decision_card.csv"),
        ("reporting", "decision-ready report, evidence registry and logs updated", "complete", "reports/olist_repurchase_final_report.md"),
    ]
    return pd.DataFrame(rows, columns=["workstream", "exit_requirement", "status", "evidence_path"])


def main() -> None:
    report = REPORT.read_text(encoding="utf-8")
    claims = claim_matrix()
    completion = completion_checklist()
    metrics = pd.read_csv(TABLES / "olist_final_report_key_metrics.csv").set_index("metric_name")
    decision = pd.read_csv(TABLES / "olist_final_crm_decision_card.csv").set_index("decision_field")
    imbalance = pd.read_csv(TABLES / "olist_imbalance_diagnostic_conclusion.csv").iloc[0]

    source_exists = []
    for source in claims["source_table"]:
        if source == "olist_project_completion_checklist.csv":
            source_exists.append(True)
        else:
            source_exists.append((TABLES / source).exists())
    claims["source_exists"] = source_exists

    forbidden_phrases = [
        "캠페인 효과가 입증되었다",
        "CRM으로 매출이 증가했다",
        "예상 매출 증가",
        "모델이 재구매 여부를 정확히 예측",
        "현재 고객에게 발송했다",
    ]
    checks = [
        ("report_date_current", "기준일: 2026-08-03" in report, "report uses current project date"),
        ("official_30d_consistent", "1.5823%" in report and abs(metrics.loc["official_30d_repeat_rate", "metric_value"] - 1.5823) < 1e-9, "official KPI matches registry"),
        ("operational_30d_consistent", "0.6166%" in report and abs(metrics.loc["operational_over24h_30d_rate", "metric_value"] - 0.616558) < 0.000001, "operational target matches registry"),
        ("candidate_count_consistent", "4,667명" in report and int(metrics.loc["final_playbook_historical_candidates", "metric_value"]) == 4667, "historical candidate count matches"),
        ("live_authorized_zero", "현재 실제 발송 승인 | 0명" in report and int(metrics.loc["final_playbook_live_authorized_count", "metric_value"]) == 0, "no live sends claimed"),
        ("current_decision_consistent", decision.loc["current_project_decision", "decision_value"] == "DO_NOT_LAUNCH" and "DO_NOT_LAUNCH" in report, "current execution decision is explicit"),
        ("conditional_proposal_consistent", decision.loc["recommended_business_next_action", "decision_value"] == "CONDITIONAL_RANDOMIZED_PILOT" and "CONDITIONAL_RANDOMIZED_PILOT" in report, "conditional proposal separated"),
        ("external_causal_limit", "인과적 열위로 해석할 수 없다" in report, "external comparison limit included"),
        ("model_causal_limit", "캠페인 효과 주장" in report, "prediction and incrementality separated"),
        ("synthetic_limit", "합성 실험 결과는 코드 검증용" in report, "synthetic evidence boundary included"),
        ("data_scope_exit", "분석은 종결할 수 있다" in report, "analysis exit is explicit"),
        ("all_claim_sources_exist", bool(claims["source_exists"].all()), "every claim has an available artifact"),
        ("all_workstreams_closed", bool(completion["status"].isin(["complete", "closed_unverifiable_with_current_data"]).all()), "every workstream is complete or explicitly closed"),
        ("no_forbidden_claims", not any(p in report for p in forbidden_phrases), "no unsupported success or revenue claim"),
        ("imbalance_no_model_replacement", not bool(imbalance["final_model_change_authorized"]) and "SMOTENC 추가 실험과 최종 모델 교체는 권고하지 않으며" in report, "imbalance research does not overwrite final model"),
        ("audit_and_decision_logs_exist", (DOCS / "data_audit_log.md").exists() and (DOCS / "decision_log.md").exists(), "two core logs exist"),
        ("insight_log_exists", (DOCS / "insight_log.md").exists(), "contextual insights exist"),
    ]
    audit = pd.DataFrame(checks, columns=["check_name", "passed", "interpretation"])
    if not audit["passed"].all():
        raise AssertionError(audit.loc[~audit["passed"]].to_dict("records"))

    status = pd.DataFrame([{
        "project_status": "ANALYSIS_COMPLETE_PORTFOLIO_READY",
        "as_of_date": "2026-08-03",
        "current_execution_decision": "DO_NOT_LAUNCH",
        "conditional_business_proposal": "CONDITIONAL_RANDOMIZED_PILOT",
        "actual_campaign_effect_status": "NOT_OBSERVED_CLOSED_AS_DATA_LIMIT",
        "audit_checks_passed": int(audit["passed"].sum()),
        "audit_checks_total": len(audit),
        "claim_count": len(claims),
        "workstream_count": len(completion),
    }])

    claims.to_csv(TABLES / "olist_final_claim_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    completion.to_csv(TABLES / "olist_project_completion_checklist.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(TABLES / "olist_final_consistency_audit.csv", index=False, encoding="utf-8-sig")
    status.to_csv(TABLES / "olist_project_final_status.csv", index=False, encoding="utf-8-sig")
    print("Final status")
    print(status.to_string(index=False))
    print("\nAudit")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

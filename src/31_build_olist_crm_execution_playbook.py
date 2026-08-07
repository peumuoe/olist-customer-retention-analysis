"""Build a non-sending CRM execution playbook and instrumentation dry run."""

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_PATH = ROOT / "data" / "processed" / "olist_crm_experiment_assignment_dry_run.csv"
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"

CUSTOMER_PLAN_PATH = PROCESSED / "olist_crm_customer_execution_plan_dry_run.csv"
PLAYBOOK_PATH = TABLES / "olist_crm_playbook.csv"
SUPPRESSION_PATH = TABLES / "olist_crm_suppression_and_frequency_rules.csv"
TRACKING_PATH = TABLES / "olist_crm_event_tracking_schema.csv"
LAUNCH_PATH = TABLES / "olist_crm_launch_gate_checklist.csv"
KPI_TREE_PATH = TABLES / "olist_crm_kpi_tree.csv"
SUMMARY_PATH = TABLES / "olist_crm_playbook_build_summary.csv"

EXPERIMENT_ID = "olist_crm_q3q4_top10_pilot_v1_20260724"
SELLER_EXPLORATION_CATEGORIES = {"watches_gifts", "auto", "toys"}


def build_customer_plan(assignment):
    plan = assignment.copy()
    plan["content_strategy"] = np.where(
        plan["primary_category"].isin(SELLER_EXPLORATION_CATEGORIES),
        "same_seller_and_category_exploration",
        "related_product_and_category_discovery",
    )
    plan["message_template_version"] = np.where(
        plan["experiment_arm"].eq("treatment"),
        "non_discount_discovery_v1",
        "holdout_no_experimental_message",
    )
    plan["planned_experimental_contact_count"] = np.where(
        plan["experiment_arm"].eq("treatment"), 1, 0
    )
    plan["planned_send_window"] = np.where(
        plan["experiment_arm"].eq("treatment"),
        "first_order_plus_24_to_48h",
        "not_applicable",
    )
    plan["benefit_type"] = "none"
    plan["recommended_channel"] = np.where(
        plan["experiment_arm"].eq("treatment"),
        "single_consented_owned_channel_to_be_frozen",
        "none",
    )
    plan["actual_send_authorized"] = False
    plan["execution_status"] = "historical_dry_run_not_contactable"
    plan["frequency_cap"] = "max_1_experimental_contact_in_30d"
    plan["transactional_messages_continue_flag"] = True
    plan["analysis_population"] = "intention_to_treat"
    return plan


def build_playbook():
    return pd.DataFrame([
        {
            "arm": "treatment",
            "target": "Q3/Q4_high within-value-band score top 10%",
            "timing": "24 to 48 hours after first order",
            "message_objective": "support discovery of relevant follow-up products without claiming proven effectiveness",
            "content_rule": "watches_gifts/auto/toys: same seller/category exploration; otherwise related product/category discovery",
            "benefit": "none",
            "contact_policy": "one experimental message in 30 days",
            "primary_kpi": "delivered repeat-order customer rate >24h and <=30d",
            "guardrails": "unsubscribe, complaint, cancellation/refund, contribution after contact cost",
            "causal_status": "hypothesis_to_be_tested",
        },
        {
            "arm": "control",
            "target": "same randomized candidate population",
            "timing": "no experimental send",
            "message_objective": "estimate counterfactual natural repeat purchase",
            "content_rule": "no experimental content",
            "benefit": "none",
            "contact_policy": "normal transactional communications only",
            "primary_kpi": "delivered repeat-order customer rate >24h and <=30d",
            "guardrails": "same outcome and customer-experience tracking where applicable",
            "causal_status": "randomized_holdout",
        },
    ])


def build_suppression_rules():
    return pd.DataFrame([
        ("marketing_consent_invalid", "pre_send", "suppress", "retain original assignment for ITT", "CRM consent source"),
        ("global_unsubscribe_or_do_not_contact", "pre_send", "suppress", "retain original assignment for ITT", "suppression list"),
        ("no_contactable_owned_channel", "pre_send", "suppress", "retain original assignment for ITT", "CRM profile"),
        ("first_order_cancelled_before_trigger", "pre_send", "suppress", "retain original assignment for ITT", "orders"),
        ("fraud_employee_test_account", "pre_send", "suppress", "retain original assignment for ITT", "account controls"),
        ("already_contacted_for_same_experiment", "pre_send", "suppress duplicate", "retain original assignment for ITT", "send log"),
        ("frequency_cap_reached", "pre_send", "suppress", "retain original assignment for ITT", "contact history"),
        ("message_delivery_failure", "post_assignment", "do not resend outside protocol", "retain in treatment for ITT", "delivery log"),
        ("message_not_opened", "post_assignment", "no exclusion", "retain in treatment for ITT", "engagement log"),
    ], columns=[
        "condition", "evaluation_time", "execution_action",
        "analysis_action", "required_source",
    ]).assign(
        frequency_cap_definition="maximum 1 experimental contact per customer in 30 days"
    )


def build_tracking_schema():
    rows = [
        ("assignment", "experiment_id", "string", True, "stable experiment identifier"),
        ("assignment", "customer_unique_id", "string", True, "customer randomization unit"),
        ("assignment", "experiment_arm", "string", True, "treatment or control"),
        ("assignment", "assignment_timestamp", "datetime_utc", True, "time of immutable assignment"),
        ("assignment", "randomization_stratum", "string", True, "pre-treatment stratum"),
        ("assignment", "model_version", "string", True, "model used for eligibility"),
        ("assignment", "repurchase_score", "float", True, "pre-treatment ranking score"),
        ("eligibility", "consent_status_at_trigger", "string", True, "marketing consent snapshot"),
        ("eligibility", "suppression_reason", "string", True, "null or predeclared reason"),
        ("send", "message_id", "string", True, "unique send attempt identifier"),
        ("send", "template_version", "string", True, "frozen treatment version"),
        ("send", "content_strategy", "string", True, "content module selected before send"),
        ("send", "channel", "string", True, "single frozen owned channel"),
        ("send", "scheduled_timestamp", "datetime_utc", True, "planned trigger time"),
        ("send", "sent_timestamp", "datetime_utc", True, "actual send time or null"),
        ("delivery", "delivery_status", "string", True, "delivered, bounced, failed"),
        ("delivery", "delivery_timestamp", "datetime_utc", False, "actual delivery time"),
        ("engagement", "open_timestamp", "datetime_utc", False, "secondary engagement event"),
        ("engagement", "click_timestamp", "datetime_utc", False, "secondary engagement event"),
        ("guardrail", "unsubscribe_timestamp", "datetime_utc", True, "customer opt-out event"),
        ("guardrail", "complaint_timestamp", "datetime_utc", True, "spam or complaint event"),
        ("outcome", "repeat_order_id", "string", True, "delivered repeat order identifier"),
        ("outcome", "repeat_order_purchase_timestamp", "datetime_utc", True, "outcome-window check"),
        ("outcome", "repeat_order_delivered_flag", "boolean", True, "primary outcome status"),
        ("outcome", "repeat_payment_value", "decimal", False, "observed secondary revenue"),
        ("cost", "contact_cost", "decimal", True, "channel delivery cost"),
        ("cost", "incentive_cost", "decimal", True, "zero for v1 no-discount treatment"),
        ("cost", "contribution_margin", "decimal", True, "required for profit guardrail"),
    ]
    return pd.DataFrame(rows, columns=[
        "event_domain", "field_name", "data_type", "required_for_launch", "purpose",
    ])


def build_launch_checklist():
    return pd.DataFrame([
        ("legal_and_consent", "approved marketing-consent basis and current suppression list", "business/legal", "blocked", "not available in Olist data"),
        ("channel", "one owned channel selected and frozen for both protocol and cost tracking", "CRM owner", "blocked", "channel not selected"),
        ("message", "non-discount discovery copy and category modules approved", "CRM/content", "pending", "semantic treatment defined; final copy absent"),
        ("frequency", "30-day one-contact cap implemented and tested", "CRM engineering", "blocked", "no CRM platform connection"),
        ("randomization", "assignment table immutable and customer uniqueness verified", "data science", "ready", "dry-run checks passed"),
        ("instrumentation", "assignment, send, delivery, consent, guardrail, order and cost joins tested", "data engineering", "blocked", "future CRM data required"),
        ("sample_size", "pilot purpose and stopping rule approved", "experiment owner", "pending", "available sample detects only large effects"),
        ("analysis", "ITT analysis code and 30-day maturity rule frozen before launch", "data science", "ready", "script 32 and 12 quality checks passed"),
        ("operations", "transactional messages remain identical across arms", "CRM operations", "pending", "must be verified in live system"),
    ], columns=["gate", "requirement", "owner", "status", "current_evidence"])


def build_kpi_tree():
    return pd.DataFrame([
        ("decision", "incremental_30d_repeat_purchase", "primary", "treatment minus control delivered repeat-customer rate", "orders", "success estimate; report 95% CI"),
        ("commercial", "observed_repeat_revenue_per_randomized_customer", "secondary", "repeat payment value / randomized customers", "orders/payments", "not future CLV"),
        ("commercial", "incremental_contribution_after_contact_cost", "guardrail", "incremental contribution - contact - incentive cost", "finance/CRM", "must not be materially negative"),
        ("customer_experience", "unsubscribe_rate", "guardrail", "unsubscribe customers / delivered customers", "CRM", "review any material increase"),
        ("customer_experience", "complaint_rate", "guardrail", "complaint customers / delivered customers", "CRM", "review any material increase"),
        ("operations", "delivery_rate", "diagnostic", "delivered messages / treatment-assigned customers", "CRM", "do not exclude failures from ITT"),
        ("engagement", "click_rate", "secondary", "clicking customers / delivered customers", "CRM", "not a substitute for repeat purchase"),
        ("model", "score_band_effect_heterogeneity", "exploratory", "treatment effect by predeclared score quintile", "assignment/orders", "hypothesis generation only"),
    ], columns=["kpi_branch", "kpi_name", "kpi_role", "definition", "source", "decision_use"])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(ASSIGNMENT_PATH, low_memory=False)
    plan = build_customer_plan(assignment)
    playbook = build_playbook()
    suppression = build_suppression_rules()
    tracking = build_tracking_schema()
    launch = build_launch_checklist()
    kpi_tree = build_kpi_tree()

    treatment = plan["experiment_arm"].eq("treatment")
    control = plan["experiment_arm"].eq("control")
    checks = pd.DataFrame([
        ("customer_plan_row_count", len(plan), 4667, len(plan) == 4667),
        ("unique_customer_count", plan["customer_unique_id"].nunique(), len(plan), plan["customer_unique_id"].nunique() == len(plan)),
        ("treatment_count", int(treatment.sum()), 2333, treatment.sum() == 2333),
        ("control_count", int(control.sum()), 2334, control.sum() == 2334),
        ("treatment_contact_count_rule_violations", int((plan.loc[treatment, "planned_experimental_contact_count"] != 1).sum()), 0, (plan.loc[treatment, "planned_experimental_contact_count"] == 1).all()),
        ("control_contact_count_rule_violations", int((plan.loc[control, "planned_experimental_contact_count"] != 0).sum()), 0, (plan.loc[control, "planned_experimental_contact_count"] == 0).all()),
        ("benefit_non_none_count", int(plan["benefit_type"].ne("none").sum()), 0, plan["benefit_type"].eq("none").all()),
        ("actual_send_authorized_count", int(plan["actual_send_authorized"].sum()), 0, (~plan["actual_send_authorized"]).all()),
        ("tracking_required_field_count", int(tracking["required_for_launch"].sum()), 24, tracking["required_for_launch"].sum() == 24),
        ("ready_launch_gate_count", int(launch["status"].eq("ready").sum()), 2, launch["status"].eq("ready").sum() == 2),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    plan.to_csv(CUSTOMER_PLAN_PATH, index=False)
    playbook.to_csv(PLAYBOOK_PATH, index=False)
    suppression.to_csv(SUPPRESSION_PATH, index=False)
    tracking.to_csv(TRACKING_PATH, index=False)
    launch.to_csv(LAUNCH_PATH, index=False)
    kpi_tree.to_csv(KPI_TREE_PATH, index=False)
    checks.to_csv(SUMMARY_PATH, index=False)

    print("\nCONTENT STRATEGY\n", pd.crosstab(
        plan["content_strategy"], plan["experiment_arm"]
    ).to_string())
    print("\nLAUNCH GATES\n", launch["status"].value_counts().to_string())
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

"""Build Olist decision actions, risk register, and experiment decision rules."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

ACTION_PATH = TABLES / "olist_decision_action_priority.csv"
RISK_PATH = TABLES / "olist_crm_risk_register.csv"
DECISION_RULE_PATH = TABLES / "olist_crm_experiment_decision_rules.csv"
SIMULATION_PATH = TABLES / "olist_decision_simulation_evidence.csv"
SUMMARY_PATH = TABLES / "olist_decision_support_build_summary.csv"


def build_actions():
    rows = [
        {
            "action_id": "A1",
            "action": "30일 공식 KPI와 24시간 제외 운영 타깃 분리 운영",
            "decision_status": "즉시 실행",
            "business_impact_1_to_5": 4,
            "evidence_strength_1_to_5": 5,
            "urgency_1_to_5": 5,
            "effort_1_to_5": 1,
            "risk_1_to_5": 1,
            "reason": "사업 진단과 CRM 모델 타깃 혼용 방지",
            "owner": "analytics/product",
            "required_resource": "KPI dictionary and dashboard definition",
        },
        {
            "action_id": "A2",
            "action": "Q3·Q4 모델 상위군 대상 1:1 무작위 CRM 파일럿",
            "decision_status": "설계 완료·실행 조건부",
            "business_impact_1_to_5": 4,
            "evidence_strength_1_to_5": 3,
            "urgency_1_to_5": 4,
            "effort_1_to_5": 3,
            "risk_1_to_5": 3,
            "reason": "제한적 예측 신호를 캠페인 인과효과와 분리해 검증",
            "owner": "CRM/product/analytics",
            "required_resource": "consent, channel, frequency cap, experiment logging",
        },
        {
            "action_id": "A3",
            "action": "CRM 배정·발송·도달·Guardrail·주문·비용 계측 연결",
            "decision_status": "실행 전 필수",
            "business_impact_1_to_5": 5,
            "evidence_strength_1_to_5": 4,
            "urgency_1_to_5": 5,
            "effort_1_to_5": 4,
            "risk_1_to_5": 2,
            "reason": "효과·고객경험·수익성을 동일 실험 단위로 판단하기 위한 전제",
            "owner": "data engineering/CRM/finance",
            "required_resource": "experiment_id and customer-level event contract",
        },
        {
            "action_id": "A4",
            "action": "할인 없는 탐색 메시지 1회로 첫 처치 단순화",
            "decision_status": "권고 실험 처치",
            "business_impact_1_to_5": 3,
            "evidence_strength_1_to_5": 2,
            "urgency_1_to_5": 3,
            "effort_1_to_5": 2,
            "risk_1_to_5": 2,
            "reason": "메시지 효과와 가격 인센티브 효과의 혼합 방지",
            "owner": "CRM/content",
            "required_resource": "one frozen template and one consented channel",
        },
        {
            "action_id": "A5",
            "action": "할인·쿠폰 제공 실험",
            "decision_status": "후순위",
            "business_impact_1_to_5": 3,
            "evidence_strength_1_to_5": 1,
            "urgency_1_to_5": 2,
            "effort_1_to_5": 3,
            "risk_1_to_5": 4,
            "reason": "마진 훼손·자연 구매 잠식·처치 해석 혼합 위험",
            "owner": "CRM/finance",
            "required_resource": "margin and incentive cost data",
        },
        {
            "action_id": "A6",
            "action": "전 고객 대상 모델 자동 타기팅 확대",
            "decision_status": "실행 금지",
            "business_impact_1_to_5": 3,
            "evidence_strength_1_to_5": 1,
            "urgency_1_to_5": 1,
            "effort_1_to_5": 4,
            "risk_1_to_5": 5,
            "reason": "Recall 17.59%, Q1·Q2 Lift 약 0.95, 지역·카테고리 편중",
            "owner": "CRM/data science",
            "required_resource": "not recommended under current evidence",
        },
        {
            "action_id": "A7",
            "action": "추가 복잡 모델을 최종 운영모델로 교체",
            "decision_status": "연구 부록",
            "business_impact_1_to_5": 2,
            "evidence_strength_1_to_5": 2,
            "urgency_1_to_5": 1,
            "effort_1_to_5": 4,
            "risk_1_to_5": 3,
            "reason": "새로운 미사용 검증데이터 없이 우월성 확정 불가",
            "owner": "data science",
            "required_resource": "current data only; no fair new holdout",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["priority_score"] = (
        0.30 * frame["business_impact_1_to_5"]
        + 0.25 * frame["evidence_strength_1_to_5"]
        + 0.20 * frame["urgency_1_to_5"]
        + 0.15 * (6 - frame["effort_1_to_5"])
        + 0.10 * (6 - frame["risk_1_to_5"])
    )
    status_order = {
        "즉시 실행": 1,
        "실행 전 필수": 2,
        "설계 완료·실행 조건부": 3,
        "권고 실험 처치": 4,
        "후순위": 5,
        "연구 부록": 6,
        "실행 금지": 7,
    }
    frame["decision_order"] = frame["decision_status"].map(status_order)
    return frame.sort_values(
        ["decision_order", "priority_score"], ascending=[True, False]
    ).reset_index(drop=True)


def build_risks():
    rows = [
        ("customer_fatigue", "추가 메시지가 피로·수신거부·불만을 증가", "unsubscribe and complaint rate", "사전 사업 비열등 한계 필요", "30일 1회 빈도 제한; 거래성 메시지 동일 유지", "한계 초과 시 중단"),
        ("natural_purchase_cannibalization", "원래 구매할 고객에게 불필요한 비용·혜택 사용", "ITT incremental repeat and contribution", "incremental contribution >= 0", "첫 실험 무할인; 대조군 유지", "공헌이익 음수면 확대 금지"),
        ("selection_concentration", "SP·bed_bath_table에 대상이 과도하게 집중", "group representation ratio", "0.5~2.0 운영 경계", "가치구간 층화; 구성비 모니터링", "2.0 초과 시 모집정책 재검토"),
        ("false_negative_low_power", "실제 중간 효과가 있어도 유의하지 않음", "effect CI width and event counts", "MDE 약 상대 70.9%", "효과크기와 구간 보고; 비유의=무효과 금지", "확증 판단 보류"),
        ("delivery_timing_conflict", "상품 수령 전 교차판매 메시지가 고객경험 악화", "cancellation, complaint, delivery status", "사업 임계값 사전 정의", "발송 전 주문 취소 억제; 24~48h 가설 검토", "Guardrail 악화 시 시점 변경"),
        ("randomization_contamination", "대조군에 다른 실험성 메시지가 노출", "contact history by arm", "대조군 실험성 연락 0회", "실험 ID·빈도 제한·캠페인 제외목록", "오염률 보고 및 실험 무효 검토"),
        ("tracking_failure", "배정·발송·주문·비용 연결 누락", "join rate and missing required fields", "필수 필드 완전성 사전 기준", "런치 전 end-to-end 테스트", "기준 미달 시 런치 금지"),
        ("margin_erosion", "할인 또는 비용이 재구매 매출보다 큼", "incremental contribution after cost", "0 이상", "V1 무할인; 비용 별도 기록", "음수면 중단"),
        ("model_drift_or_miscalibration", "과거 모델 순위가 현재 고객에서 재현되지 않음", "Lift@10, calibration, PSI", "Lift<1 중단; PSI 0.10/0.25 점검", "파일럿 한정 사용", "Lift<1이면 점수 타기팅 중단"),
    ]
    return pd.DataFrame(rows, columns=[
        "risk_id", "side_effect_or_risk", "leading_indicator",
        "guardrail_or_threshold", "mitigation", "decision_trigger",
    ])


def build_decision_rules():
    rows = [
        {
            "decision": "GO_확대_또는_확증실험",
            "primary_condition": "30일 ITT 위험차이 95% 구간 하한 > 0",
            "commercial_condition": "접촉비 차감 후 증분 공헌이익 >= 0",
            "guardrail_condition": "사전 정의 수신거부·불만·취소 한계 미초과",
            "data_quality_condition": "randomization and required event joins pass",
            "interpretation": "양의 효과가 지지됨; 검증한 모집단과 처치 범위 안에서만 확대 검토",
        },
        {
            "decision": "ITERATE_판단보류",
            "primary_condition": "점추정치 > 0이지만 95% 구간이 0 포함",
            "commercial_condition": "명확한 음의 공헌이익 신호 없음",
            "guardrail_condition": "중대한 위해 신호 없음",
            "data_quality_condition": "analysis valid but precision insufficient",
            "interpretation": "성공·실패로 단정하지 않고 용량·시점 재검토 또는 동등 코호트 누적",
        },
        {
            "decision": "STOP_중단_또는_재설계",
            "primary_condition": "효과 추정치가 음수이거나 구간이 사업상 중대한 손실 포함",
            "commercial_condition": "증분 공헌이익 < 0",
            "guardrail_condition": "사전 정의 위해 한계 중 하나라도 초과",
            "data_quality_condition": "if joins fail, stop analysis claim and repair instrumentation",
            "interpretation": "확대하지 않고 메시지·시점·타기팅·측정 실패를 구분",
        },
        {
            "decision": "NO_DECISION_실험무효",
            "primary_condition": "무작위화 훼손·대조군 오염·결과 성숙 미완료",
            "commercial_condition": "비용 데이터 누락",
            "guardrail_condition": "필수 안전지표 누락",
            "data_quality_condition": "critical launch/analysis checks fail",
            "interpretation": "해당 실험으로 사업 의사결정을 지원할 수 없음",
        },
    ]
    return pd.DataFrame(rows)


def build_simulation_evidence():
    temporal = pd.read_csv(TABLES / "olist_ml_temporal_stability_summary.csv").iloc[0]
    power = pd.read_csv(TABLES / "olist_crm_experiment_power_scenarios.csv")
    synthetic = pd.read_csv(TABLES / "olist_crm_synthetic_scenario_summary.csv")
    uplift30 = power.loc[power["scenario"].eq("relative_uplift_30pct")].iloc[0]
    uplift75 = synthetic.loc[
        synthetic["scenario"].eq("synthetic_relative_uplift_75pct")
    ].iloc[0]
    uplift30_synthetic = synthetic.loc[
        synthetic["scenario"].eq("synthetic_relative_uplift_30pct")
    ].iloc[0]
    rows = [
        ("과거_모델_백테스트", "연속 미래기간 4개", f"Lift@10% {temporal['min_lift_at_10pct']:.3f}~{temporal['max_lift_at_10pct']:.3f}", "예측 순위의 시간 안정성 확인", "캠페인 효과 추정 금지"),
        ("타기팅_정책_드라이런", "Q3·Q4 가치구간 내부 상위 10%", "후보 4,667명; 전체 상위 10%와 85.58% 중복", "운영 모집정책 비교", "현재 연락 가능한 고객명단으로 해석 금지"),
        ("검정력_상대30pct", "상대 +30% 효과 가정", f"필요 총표본 {int(uplift30['required_total_n']):,}명; 현재 검정력 {uplift30['achieved_power_with_available_n'] * 100:.1f}%", "표본수 계획", "기대효과 예측으로 해석 금지"),
        ("합성_상대30pct", "합성 사건 54명 대 41명", f"95% 구간 {uplift30_synthetic['absolute_effect_ci95_lower'] * 100:.3f}~{uplift30_synthetic['absolute_effect_ci95_upper'] * 100:.3f}%p", "현재 표본에서 판단보류 가능성 설명", "실제 관찰 성과로 해석 금지"),
        ("합성_상대75pct", "합성 사건 72명 대 41명", f"95% 구간 {uplift75['absolute_effect_ci95_lower'] * 100:.3f}~{uplift75['absolute_effect_ci95_upper'] * 100:.3f}%p", "큰 효과 탐지 코드 검증", "실제 관찰 성과로 해석 금지"),
    ]
    return pd.DataFrame(rows, columns=[
        "evidence_item", "scenario_or_scope", "result",
        "allowed_decision_use", "prohibited_interpretation",
    ])


def main():
    actions = build_actions()
    risks = build_risks()
    rules = build_decision_rules()
    simulation = build_simulation_evidence()
    checks = pd.DataFrame([
        ("action_count", len(actions), 7, len(actions) == 7),
        ("risk_count", len(risks), 9, len(risks) == 9),
        ("decision_rule_count", len(rules), 4, len(rules) == 4),
        ("simulation_evidence_count", len(simulation), 5, len(simulation) == 5),
        ("do_now_count", int(actions["decision_status"].eq("즉시 실행").sum()), 1, actions["decision_status"].eq("즉시 실행").sum() == 1),
        ("do_not_do_count", int(actions["decision_status"].eq("실행 금지").sum()), 1, actions["decision_status"].eq("실행 금지").sum() == 1),
        ("all_risks_have_mitigation", int(risks["mitigation"].notna().sum()), len(risks), risks["mitigation"].notna().all()),
        ("all_rules_have_data_quality_condition", int(rules["data_quality_condition"].notna().sum()), len(rules), rules["data_quality_condition"].notna().all()),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    actions.to_csv(ACTION_PATH, index=False)
    risks.to_csv(RISK_PATH, index=False)
    rules.to_csv(DECISION_RULE_PATH, index=False)
    simulation.to_csv(SIMULATION_PATH, index=False)
    checks.to_csv(SUMMARY_PATH, index=False)
    print("\nACTIONS\n", actions[[
        "action_id", "decision_status", "priority_score", "action"
    ]].to_string(index=False))
    print("\nCHECKS\n", checks.to_string(index=False))


if __name__ == "__main__":
    main()

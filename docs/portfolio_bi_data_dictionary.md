# BI 데이터 사전

기준일: 2026-08-07

상세 컬럼 단위 사전은 `data_dictionary.csv`를 사용합니다.

## 데이터 모델 원칙

- 모든 테이블은 집계 Grain이며 고객·주문 식별자를 포함하지 않습니다.
- 테이블 간 강제 관계보다 각 분석 페이지의 목적별 Fact로 사용합니다.
- 관찰기간 필터는 `window_days`, 선택비율 필터는 `top_fraction` 또는 `selected_fraction`을 사용합니다.
- 공식 KPI와 운영 타깃은 `metric_type`으로 반드시 구분합니다.
- 외부 비교와 CRM 관찰값을 인과효과로 표현하지 않습니다.

## fact_retention.csv

- Grain: 지표 정의 × 관찰기간 1행
- 행 수: 13
- 원천: `retention_window_summary.csv + olist_repurchase_survival_summary.csv`

## fact_external_benchmark.csv

- Grain: 플랫폼 × 관찰기간 1행
- 행 수: 15
- 원천: `external_retention_comparison.csv`

## fact_segment_priority.csv

- Grain: 고객군 차원 × 고객군 값 1행
- 행 수: 39
- 원천: `olist_improvement_opportunity_priority.csv`

## fact_model_performance.csv

- Grain: 모델 × 상위 선택비율 1행
- 행 수: 20
- 원천: `olist_ml_test_model_summary.csv + olist_ml_test_lift_recall.csv`

## fact_crm_capacity.csv

- Grain: 평가범위 × 선택비율 1행
- 행 수: 18
- 원천: `olist_crm_capacity_frontier.csv`

## fact_action_priority.csv

- Grain: Action Item 1행
- 행 수: 7
- 원천: `olist_decision_action_priority.csv`

## fact_crm_risk.csv

- Grain: 리스크 1행
- 행 수: 9
- 원천: `olist_crm_risk_register.csv`

## fact_launch_gate.csv

- Grain: 런치 게이트 1행
- 행 수: 9
- 원천: `olist_crm_launch_gate_checklist.csv`

## fact_claim_evidence.csv

- Grain: 최종 주장 1행
- 행 수: 13
- 원천: `olist_final_claim_evidence_matrix.csv`

# Olist Portfolio BI Data Mart

기준일: 2026-08-07

Tableau, Power BI, Looker Studio에서 공통으로 사용할 수 있도록 만든 공개용 집계 데이터입니다.

## 공개 원칙

- 고객·주문·판매자·리뷰 식별자를 포함하지 않습니다.
- Olist/H&M/REES46 대용량 원본과 고객 단위 모델 점수는 포함하지 않습니다.
- 외부 비교는 인과 비교가 아니며 `causal_interpretation_allowed=False`입니다.
- CRM 수용량의 재구매는 과거 자연 재구매이며 캠페인 증분효과가 아닙니다.
- 역사적 후보를 현재 연락 가능한 고객으로 해석하지 않습니다.

## 파일

| 파일 | Grain | 사용 목적 |
|---|---|---|
| `fact_retention.csv` | 지표 정의 × 관찰기간 | 공식 KPI와 운영 타깃 |
| `fact_external_benchmark.csv` | 플랫폼 × 관찰기간 | 외부 관찰값 비교 |
| `fact_segment_priority.csv` | 고객군 차원 × 값 | 개선기회 우선순위 |
| `fact_model_performance.csv` | 모델 × 선택비율 | PR-AUC·Lift·Recall |
| `fact_crm_capacity.csv` | 범위 × 선택비율 | 발송 수용량 |
| `fact_action_priority.csv` | Action Item | 자원 우선순위 |
| `fact_crm_risk.csv` | 리스크 | Side Effect·Guardrail |
| `fact_launch_gate.csv` | 런치 조건 | 실행 준비상태 |
| `fact_claim_evidence.csv` | 최종 주장 | 주장–근거 추적 |
| `data_dictionary.csv` | 데이터셋 × 컬럼 | 데이터 사전 |
| `data_manifest.csv` | 데이터셋 | 행·열·해시·출처 |

각 CSV는 UTF-8 with BOM으로 저장되어 Windows BI 도구에서 한글을 읽을 수 있습니다.

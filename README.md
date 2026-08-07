# Olist Customer Retention Analysis

브라질 Olist 전자상거래 공개 데이터를 이용해 재구매율을 진단하고, 외부 데이터 비교·개선기회 탐색·재구매 가능성 예측·CRM 실험 설계까지 연결한 의사결정 지원 프로젝트입니다.

기준일: 2026-08-07  
프로젝트 상태: `ANALYSIS_COMPLETE_PORTFOLIO_READY`  
현재 실행 결정: `DO_NOT_LAUNCH`

## Executive Summary

- 완전한 관찰기간을 확보한 Olist 고객의 누적 재구매율은 30일 **1.5823%**, 60일 **1.9507%**, 90일 **2.2756%**, 180일 **3.1052%**, 365일 **4.5331%**입니다.
- H&M·REES46 공개 데이터의 관찰 재구매율은 더 높지만 구매 이벤트, 상품 구성, 채널, 관찰기간과 사업모델이 달라 Olist 정책의 인과적 열위로 해석하지 않습니다.
- 첫 주문 정보 기반 L2 로지스틱 회귀는 독립 테스트 상위 10%에서 **Lift 1.7588**, **Recall 17.59%**를 기록했습니다. 무작위보다 양성을 농축하지만 자동 의사결정이 아닌 CRM 후보 압축 보조 도구로 제한합니다.
- Q3·Q4 가치구간 내부 상위 10%인 4,667명을 역사적 파일럿 후보로 정의했으나, 동의·연락채널·비용·결과 계측 데이터가 없어 현재 캠페인은 실행하지 않습니다.
- 제안은 효과가 입증된 정책이 아니라 운영 조건을 충족한 뒤 무작위 대조실험으로 검증할 `CONDITIONAL_RANDOMIZED_PILOT`입니다.

## Business Questions

1. Olist의 30·60·90·180·365일 재구매율은 어느 수준인가?
2. 외부 이커머스 데이터와 비교할 때 관찰 격차는 어느 정도인가?
3. 고객 규모·재구매율 격차·관찰 고객가치·개입 가능성을 고려하면 어디를 먼저 살펴봐야 하는가?
4. 첫 주문 시점 정보로 재구매 가능 고객을 얼마나 압축할 수 있는가?
5. 모델 점수를 CRM 실험으로 어떻게 연결하고 증분효과를 어떻게 검증해야 하는가?

## Analysis Design

| 구분 | 정의 |
|---|---|
| 고객 식별자 | `customer_unique_id` |
| 주문 식별자 | `order_id` |
| 재구매 이벤트 | 서로 다른 배송 완료 주문 |
| 공식 30일 KPI | 첫 주문 후 30일 이내 두 번째 배송 완료 주문 |
| 운영 ML 타깃 | 첫 주문 24시간 초과~30일 이내 두 번째 배송 완료 주문 |
| 예측 시점 | 첫 주문 시점 |
| 모델 평가 | PR-AUC·Calibration·Lift·Recall 중심 |
| 캠페인 효과 | 향후 무작위 실험군·대조군으로만 검증 |

공식 KPI와 운영 ML 타깃은 목적과 사건 정의가 다르므로 혼합하지 않습니다.

## Key Results

### Retention

![Olist 기간별 누적 재구매율](reports/figures/01_olist_retention_curve.png)

### External benchmark

![Olist H&M REES46 30일 비교](reports/figures/02_external_30d_comparison.png)

### Opportunity priority

![개선기회 우선순위 지도](reports/figures/03_opportunity_priority_map.png)

### Model capacity

![로지스틱 모델 Lift와 Recall](reports/figures/04_model_lift_recall.png)

## Decision Recommendation

현재 상태는 `DO_NOT_LAUNCH`입니다. 실제 Olist 운영 환경에서 다음 조건을 충족한 경우에만 제한된 무작위 파일럿으로 전환합니다.

- 마케팅 동의와 연락 가능 채널 확보
- 중복 발송 및 빈도 제한 규칙 구현
- CRM 노출·주문·비용·불만 데이터 연결
- Primary KPI와 Guardrail KPI 사전 등록
- 실험군·대조군 무작위 배정과 ITT 분석
- 효과·수익성·고객 위해 기준을 모두 통과할 때만 확대

## Repository Guide

| 경로 | 내용 |
|---|---|
| `src/` | 감사·마트·분석·모델·실험 설계 스크립트 42개 |
| `data/portfolio_data/` | BI 도구용 공개 집계 데이터와 데이터 사전 |
| `docs/` | 지표 정의·감사 로그·의사결정 로그·대시보드 설계 |
| `reports/` | 최종 분석 보고서와 핵심 그림 |
| `outputs/tables/` | 단계별 검증·분석 결과표 |
| `models/` | 공개 가능한 모델 메타데이터 |

주요 문서:

- [최종 분석 보고서](reports/olist_repurchase_final_report.md)
- [대시보드 설계서](docs/dashboard_design_spec.md)
- [BI 데이터 사전](docs/portfolio_bi_data_dictionary.md)
- [지표 정의](docs/metric_definition.md)
- [데이터 감사 기록](docs/data_audit_log.md)
- [분석 의사결정 기록](docs/decision_log.md)
- [핵심 인사이트 기록](docs/insight_log.md)

## Reproduction

Python 3.13 환경을 기준으로 작성했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\src\42_build_portfolio_bi_data_mart.py
```

마지막 명령은 기존 분석 산출물로 공개 BI 마트를 다시 만들고 9개 품질검사를 수행합니다. 전체 원천 분석 재현에는 아래 원본 데이터가 별도로 필요합니다.

- Brazilian E-Commerce Public Dataset by Olist
- H&M Personalized Fashion Recommendations
- REES46 Retail Rocket/eCommerce behavior dataset 사용본

원본·대용량 외부 데이터, 고객 단위 가공 마트와 직렬화 모델은 저장소에 포함하지 않습니다. 공개 BI 패키지는 고객·주문·판매자·리뷰 식별자를 포함하지 않습니다.

## Limitations

- Olist 관찰기간 이전·이후 구매와 플랫폼 밖 구매는 알 수 없습니다.
- 외부 데이터 비교는 지표와 사업 구조가 달라 인과 비교가 아닙니다.
- 첫 주문 결제금액은 관찰 가치이며 미래 CLV가 아닙니다.
- 예측모델의 순위 성능은 캠페인의 증분효과를 의미하지 않습니다.
- 실제 캠페인 데이터가 없어 제안의 효과·수익성·부작용은 아직 검증되지 않았습니다.

## Tech Stack

Python, pandas, NumPy, scikit-learn, SciPy, XGBoost, Matplotlib 및 BI 도구용 집계 CSV를 사용했습니다.

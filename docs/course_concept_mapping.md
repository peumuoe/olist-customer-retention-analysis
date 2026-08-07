# 강의 개념과 프로젝트 적용 매핑

| 분석 단계 | 적용한 강의 개념 | 프로젝트 적용 내용 | 생성 파일 | 학습 포인트 |
|---|---|---|---|---|
| REES46 원본 구조 감사 | 정형 데이터·Tidy Data | CSV 1행을 사용자·상품·시각별 이벤트 1건으로 확인하고 9개 컬럼을 검증 | `rees46_file_audit_summary.csv` | 파일 형식이 같아도 주문행과 이벤트행의 의미는 다르다. |
| 대용량 전수 집계 | GIGO·재현 가능성 | 100만 행 청크로 두 파일 109,950,743행을 전수 처리하고 두 번 실행해 행 수를 재현 | `13_rees46_data_audit.py` | 메모리에 모두 올리지 않아도 전수 품질 검사가 가능하다. |
| 이벤트 분포 검증 | 분석 단위·데이터 계보 | view·cart·purchase 합계를 원본 총행 수와 대조 | `rees46_event_type_summary.csv` | 집계 결과는 원본 행 수로 역추적할 수 있어야 한다. |
| 결측·고유값 감사 | 결측과 중복·GIGO | 식별자 결측과 카테고리·브랜드 결측을 전체 및 purchase 기준으로 분리 | `rees46_column_summary.csv`, `rees46_purchase_data_audit_summary.csv` | 결측은 일괄 삭제하지 않고 분석 목적에 미치는 영향을 판단한다. |
| 완전 중복 감사 | 기본키 후보·중복 | 청크 내부와 경계의 중복 하한만 확인하고 정상 반복 이벤트 가능성을 보존 | `rees46_file_audit_summary.csv` | 중복처럼 보인다는 이유만으로 삭제하면 실제 구매 수량을 잃을 수 있다. |
| 관찰기간 감사 | 분자·분모·관찰기간 | 두 달 범위에서 30·60일 가능, 90일 이상 불가를 구분 | `rees46_date_range_summary.csv` | 재구매율 분모에는 충분한 관찰기회를 가진 고객만 포함해야 한다. |
| 외부 비교 후보 평가 | 정보 리터러시·외부 타당성 | Olist와 구조 유사성 및 주문번호·기간 차이를 함께 기록 | `data_audit_log.md` | 같은 지표명도 데이터 생성 과정이 다르면 직접 비교가 제한된다. |

현재 문서는 REES46 원본 감사와 구매 이벤트 대체키 감사까지 기록한다.
재구매율은 후속 검증 전까지 학습 완료로 표시하지 않는다.

| 분석 단계 | 적용한 강의 개념 | 프로젝트 적용 내용 | 생성 파일 | 학습 포인트 |
|---|---|---|---|---|
| REES46 구매 이벤트 대체키 감사 | 분석 단위·기본키 후보·GIGO | 상품 purchase 행을 구매 횟수로 사용하지 않고 `user_id + user_session`으로 집계 | `rees46_purchase_key_candidate_comparison.csv` | 원본 행의 의미와 비즈니스 사건의 의미는 다를 수 있다. |
| 세션 품질 검증 | 기본키·함수 종속 | 모든 세션이 한 고객에게만 귀속되는지 전수 확인 | `rees46_purchase_session_audit_summary.csv` | 대체키는 고유성뿐 아니라 고객 귀속과 생성 구조를 설명해야 한다. |
| 장기 세션 민감도 | 민감도 분석·측정 오차 | 30분·60분 간격 분리 결과를 주 정의와 비교 | `rees46_purchase_session_duration_summary.csv` | 근거 없는 시간 기준은 확정값이 아니라 민감도 범위로 사용한다. |
| 완전 중복 해석 | 중복·데이터 맥락 | 85개 추가행을 수량 가능성 때문에 보존하고 세션 집계로 영향 통제 | `rees46_purchase_session_audit_summary.csv` | 중복처럼 보이는 행을 업무 맥락 없이 삭제하면 실제 사건을 잃을 수 있다. |
| REES46 구매 이벤트 마트 | Tidy Data·Grain 변환 | 상품 purchase 행을 고객·세션당 1행으로 집계 | `rees46_purchase_events.csv` | 원본 관찰 단위와 분석 목적의 관찰 단위를 구분한다. |
| 마트 무결성 검증 | 데이터 계보·기본키 | 마트의 상품행 수 합계를 원본 purchase 행 수와 대조하고 복합키 중복을 검사 | `rees46_purchase_event_mart_summary.csv` | 파일 생성이 아니라 원본에서 결과까지 추적 가능해야 완료다. |
| 품질 플래그 | 민감도 분석·결측 해석 | 월 경계·장기 세션·설명정보 결측을 삭제하지 않고 플래그로 보존 | `rees46_purchase_event_quality_flag_summary.csv` | 불확실한 데이터를 보존하면서 분석 영향은 별도로 검증할 수 있다. |
| 구매 구조·가격 분포 | 기술통계·분포 | 다상품 구매 비중과 가격 합계의 중앙값·분위수·극단값을 확인 | `rees46_purchase_event_distribution_summary.csv` | 치우친 분포에서는 평균만으로 대표값을 판단하면 안 된다. |
| 고객별 구매 순서 | 정렬·윈도 함수·Grain | 고객별 구매시각을 정렬하고 `cumcount()`로 첫·두 번째 관찰 구매를 추출 | `rees46_customer_purchase_audit_metrics.csv` | 구매 순서는 행의 저장 순서가 아니라 명시적인 시각 정렬로 정한다. |
| 관찰 종료일 | 시간 범위·GIGO | 전체 이벤트가 아닌 마지막 purchase 시각을 종료일로 선택 | `rees46_customer_purchase_order_audit_summary.csv` | 분석 대상 사건의 실제 관찰 범위를 사용해야 한다. |
| 기간별 적격 고객 | 오른쪽 검열·분모 정의 | 첫 관찰 구매 후 N일 이상 관찰 가능한 고객만 분모 후보로 표시 | `rees46_observation_window_eligibility_summary.csv` | 아직 관찰하지 못한 고객을 미재구매로 처리하면 비율이 왜곡된다. |
| 첫 관찰 구매 해석 | 좌측 검열 | 데이터 시작 전 구매 이력을 알 수 없으므로 생애 첫 구매 표현을 금지 | `data_audit_log.md` | 데이터의 시작이 고객 생애의 시작은 아니다. |
| 전체 기간 반복구매 | 기술통계와 KPI 구분 | 2회 이상 고객 비중을 고정 기간 재구매율과 분리 | `rees46_customer_purchase_count_distribution.csv` | 관찰기회가 다른 전체 기간 비율은 직접 벤치마크로 쓰기 어렵다. |
| REES46 30일 재구매율 | 분자·분모·조건부 비율 | N일 관찰 가능한 고객만 분모로 삼고 N일 내 두 번째 구매 고객을 분자로 계산 | `rees46_retention_window_summary.csv` | 비율을 해석하려면 숫자보다 먼저 분모의 자격 조건을 확인해야 한다. |
| 오른쪽 검열 처리 | 생존분석 기초·결측의 의미 | 관찰기간이 부족한 고객의 재구매 플래그를 `False`가 아닌 결측값으로 보존 | `rees46_customer_retention_metrics.csv` | 아직 결과를 볼 수 없는 사례와 실제 미발생 사례는 다르다. |
| 세션 플래그 민감도 | 민감도 분석·강건성 | 장기·월 경계 세션 포함 여부별 동일 KPI를 다시 계산해 변화폭을 %p로 비교 | `rees46_retention_sensitivity_summary.csv` | 민감도 분석은 어느 정의가 참인지 증명하기보다 처리 선택이 결론을 바꾸는지 측정한다. |
| 비율 차이 표현 | 퍼센트와 퍼센트포인트 | 시나리오별 재구매율 차이를 상대 변화율이 아닌 %p로 제시 | `rees46_retention_sensitivity_summary.csv` | 두 비율을 직접 뺀 값은 퍼센트포인트로 표현한다. |
| 외부 재구매율 통합 비교 | 비교 가능성·동일 공식 | 세 데이터의 분자·분모 구조를 맞추고 구매 이벤트 정의 차이를 별도 표로 기록 | `external_retention_comparison.csv` | 공식이 같아도 데이터 생성 구조가 다르면 완전히 동일한 KPI는 아니다. |
| 비교 가능성 등급 | 직접 비교·제한적 참고·비교 불가 | H&M 보수 비교, REES46 30일 주 비교·60일 참고·90일 이상 불가로 구분 | `external_retention_comparison_methodology.csv` | 숫자가 존재한다는 이유만으로 대표성이 확보되는 것은 아니다. |
| 외부 대비 격차 | 퍼센트포인트·관찰 관계 | 외부 재구매율에서 Olist 재구매율을 빼 격차를 계산 | `external_retention_comparison.csv` | 격차는 개선 탐색의 출발점이며 격차의 원인을 증명하지 않는다. |
| 공개 벤치마크 감사 | 지표 동명이의·정의 동등성 | 동일한 재구매 명칭의 분석 단위·기간창·분자·분모를 원문에서 분해 | `public_retention_benchmark_audit.csv` | 지표 이름이 같다고 계산식도 같은 것은 아니다. |
| 벤치마크 증거 등급 | 직접 비교·보조 비교·참고선·제외 | 정의 일치 정도와 출처 추적성에 따라 사용 범위를 구분 | `public_retention_benchmark_audit.csv` | 비교 가능성은 숫자의 크기가 아니라 정의와 증거 수준으로 판단한다. |
| 출처 추적성 | 데이터 계보·재현 가능성 | 발행기관·연도·URL·표본·공식·채택 판단을 한 행에 기록 | `public_retention_benchmark_audit.csv` | 원자료로 돌아갈 수 없는 벤치마크는 의사결정 근거가 약하다. |
| 선택 편향 점검 | 표본 대표성 | 솔루션 고객사·자가 응답·방법론 미공개 여부를 비교 제한으로 기록 | `data_audit_log.md` | 공개된 평균도 전체 시장을 대표한다고 자동으로 가정할 수 없다. |
| Olist 재구매 시간 분석 | Kaplan–Meier·우측 중도절단 | 첫 관찰 구매부터 두 번째 구매 또는 데이터 종료까지의 시간을 고객별로 구성 | `olist_repurchase_survival_summary.csv` | 미발생과 아직 관찰되지 않음을 구분해야 한다. |
| 생존분석 KPI 비교 | 독립 중도절단 가정·코호트 효과 | Kaplan–Meier 값과 완전 관찰 고객 고정창 값을 함께 보고 달력 코호트별 추정치를 점검 | `olist_repurchase_survival_cohort_summary.csv` | 중도절단이 달력 시점과 연결되면 전체 생존곡선도 구성 변화의 영향을 받는다. |
| 30·60·90일 결과창 비교 | 멀티호라이즌·공통 코호트 | 기간별 성숙 표본과 동일 90일 성숙 고객의 누적 양성을 병렬 평가 | `olist_multihorizon_common_cohort_summary.csv` | 기간을 늘릴 때 표본 구성 변화와 사건 누적을 분리해야 한다. |
| 장기 결과로의 점수 전이 | 시간창 불일치·Lift·Recall | 동결된 30일 점수를 60·90일 결과에 적용하되 재학습·재선택하지 않음 | `olist_multihorizon_frozen_score_summary.csv` | 다른 Horizon의 점수 평균과 실제율 차이는 정식 Calibration 비교가 아니다. |
| 상위 10% 오류 감사 | 혼동행렬·Precision·Recall·희소 사건 | 독립 테스트의 TP·FP·FN·TN과 선택·누락 규모를 함께 계산 | `olist_model_top10_error_summary.csv` | 낮은 기준율에서는 유효한 순위모델도 FP가 많다. |
| CRM 발송량 선택 | Capacity frontier·한계 수익 | Q3·Q4 내부 선택 비율별 누적·추가 포착과 접촉량 비교 | `olist_crm_capacity_frontier.csv` | 과거 재구매 농축은 캠페인 증분전환이 아니다. |
| 비용 가정 감사 | 손익분기 민감도 | 접촉비와 증분 공헌이익 가정별 최소 필요 증분전환율 계산 | `olist_crm_cost_break_even_sensitivity.csv` | 비용 가정표를 기대 ROI로 표현하지 않는다. |
| 최종 CRM 통합 | 실행형 세그먼트·Treatment·Control | 고객군·점수·콘텐츠·시점·KPI·제외조건·판정규칙을 한 플레이북으로 연결 | `olist_final_integrated_crm_playbook.csv` | 세그먼트는 설명이 아니라 서로 다른 실행 규칙에 연결돼야 한다. |
| 현재 실행 판단 | Launch gate·조건부 의사결정 | 실제 승인 0명과 차단 게이트를 근거로 DO_NOT_LAUNCH와 조건부 파일럿을 분리 | `olist_final_crm_decision_card.csv` | 분석 설계 완료와 운영 실행 가능은 다른 상태다. |
| 최종 주장 감사 | Claim–evidence mapping·Decision boundary | 주장별 근거 수준·원천·허용·금지 표현 연결 | `olist_final_claim_evidence_matrix.csv` | 모든 결론은 근거가 허용하는 범위 안에서만 표현해야 한다. |
| 프로젝트 종료 판정 | Exit criteria·일관성 감사 | 12개 작업영역과 16개 보고서 일관성 검사를 통해 포트폴리오 전환 상태 확정 | `olist_project_final_status.csv` | 검증 불가능한 영역을 명시적으로 닫는 것도 분석 완성의 일부다. |
| 희소 양성 진단 | Class Weight·Random Oversampling | 시간순 학습 데이터만 가중·복제하고 원본 검증에서 PR-AUC·Lift·Calibration 비교 | `olist_imbalance_validation_candidates.csv` | 양성 행의 복제는 새로운 행동 신호를 만들지 않는다. |
| 불균형 모델 판정 | 순위 안정성·사후 테스트·확률 왜곡 | 상위 고객 중첩과 보정 전후 확률을 확인하고 모델 교체를 보류 | `olist_imbalance_diagnostic_conclusion.csv` | 이미 본 테스트의 작은 개선은 새로운 최종 성능이 아니다. |
| 공개 BI 데이터 설계 | 데이터 마트·Fact Grain·Semantic layer | 검증 산출물을 9개 집계 Fact와 96개 컬럼 사전으로 재구성 | `data/portfolio_data/data_dictionary.csv` | BI 마트는 원본 복사가 아니라 의사결정 의미계층이다. |
| GitHub 데이터 공개 | 데이터 최소화·Lineage·Manifest | 식별자 제거, 대용량 원본 제외, 파일별 해시와 원천 기록 | `data/portfolio_data/data_manifest.csv` | 공개 재현성과 원본 전체 공개는 같은 요구가 아니다. |
| 첫 관찰 구매 마트 | 인덱스 구매·고객 Grain | 고객별 첫 배송 완료 주문 특성과 30일 재구매 여부를 1행으로 결합 | `olist_first_purchase_customer_mart.csv` | 결과변수와 설명 특성의 기준 시점을 같은 주문에 맞춰야 한다. |
| 특성 가용 시점 | 데이터 누수·예측 시점 | 주문 시점과 경험 완료 후 특성을 분리 표시 | `olist_first_purchase_segment_retention.csv` | 미래에 알게 되는 정보를 과거 시점 의사결정에 사용할 수 없다. |
| 단변량 세그먼트 프로파일링 | 조건부 비율·구성 효과 | 9개 차원별 고객 수·재구매율·전체 대비 격차를 계산 | `olist_first_purchase_segment_retention.csv` | 그룹 차이는 연관성이며 다른 특성의 구성 차이를 포함할 수 있다. |
| Wilson 신뢰구간 | 비율의 통계적 불확실성 | 낮은 30일 사건률에 대해 그룹별 95% 구간 계산 | `olist_first_purchase_segment_retention.csv` | 점추정치만으로 순위를 확정하지 않고 불확실성 범위를 함께 본다. |
| 다중 비교 문제 | 탐색적 분석·우연한 극단값 | 133개 결과를 동시에 탐색하고 최소 표본과 신뢰구간으로 과해석을 제한 | `data_audit_log.md` | 그룹을 많이 볼수록 우연한 최고·최저값이 나올 가능성이 커진다. |
| 다변량 조건부 연관성 | 교란·로지스틱 회귀 | 지역·금액·결제·월 코호트를 함께 고려해 첫 구매 특성의 방향을 재평가 | `olist_multivariable_coefficients.csv` | 조정 후 연관성도 관찰되지 않은 교란을 제거하지 못한다. |
| 기준 범주와 오즈비 | 범주형 회귀 해석 | 상품 1개·단일 판매자·bed_bath_table 등을 기준으로 오즈비 산출 | `olist_multivariable_coefficients.csv` | 오즈비는 확률의 비율이 아니며 기준 범주가 바뀌면 표현도 바뀐다. |
| L2 정규화 | 계수 축소·안정화 | 희소 사건과 다수 원-핫 변수의 극단 계수를 억제 | `olist_multivariable_model_summary.csv` | 정규화는 과적합을 줄이지만 변수의 인과성을 보장하지 않는다. |
| 희소 범주 통합 | 범주 안정화 | 표본 500명 미만 수준을 other_rare로 통합 | `olist_multivariable_adjusted_probabilities.csv` | 작은 범주의 극단값을 그대로 학습하면 불안정한 결론이 생길 수 있다. |
| 조정 예측확률 | 표준화·조건부 비교 | 다른 특성 분포를 유지한 모델 기반 평균 확률과 기준 대비 격차 계산 | `olist_multivariable_adjusted_probabilities.csv` | 조정 확률 차이는 실험으로 확인한 증분효과가 아니다. |
| 월 코호트 통제 | 시기 교란 | 첫 구매 월을 회귀에 포함해 시기별 구성 변화를 부분 통제 | `21_olist_multivariable_association.py` | 달력 시점에 따른 시장·상품 구성 변화가 특성 효과처럼 보일 수 있다. |
| 개선 기회 우선순위 | 다기준 의사결정·MCDA | 규모·격차·첫 구매 가치·개입 가능성을 네 축으로 결합 | `olist_improvement_opportunity_priority.csv` | 하나의 낮은 비율만으로 투자 우선순위를 정하지 않는다. |
| 순위 점수 정규화 | 백분위 점수 | 단위가 다른 고객 수·%p·금액을 후보 내 0~1 상대점수로 변환 | `22_build_olist_opportunity_priority.py` | 상대점수는 원래 단위의 경제적 의미를 대체하지 않는다. |
| 가중치 시나리오 | 의사결정 민감도 | 균형형·규모 중심·실행 중심 가중치별 순위를 비교 | `olist_improvement_opportunity_weight_scenarios.csv` | 가중치는 데이터의 객관적 정답이 아니라 의사결정 선호다. |
| 순위 강건성 | 순위 안정성 | 세 시나리오의 상위 10위 반복 횟수와 평균 순위를 계산 | `olist_improvement_opportunity_priority.csv` | 가중치에 안정적이라는 사실이 정책 효과를 증명하지는 않는다. |
| 중첩 세그먼트 | 집합 교집합·Jaccard | 상위 후보 간 고객 교집합·합집합·중첩률 계산 | `olist_improvement_opportunity_top10_overlap.csv` | 중첩 후보의 고객 수를 합하면 동일 고객을 여러 번 계산한다. |
| 개입 가능성 루브릭 | 운영 판단·추적성 | 차원별 1~5점과 판단 이유를 별도 표로 기록 | `olist_improvement_opportunity_actionability_rubric.csv` | 실행 가능성은 모델 산출값이 아니라 검토 가능한 비즈니스 가정이다. |
| 카테고리 전이행렬 | 조건부 분포·행렬 집계 | 첫 주문 카테고리를 행, 두 번째 주문 카테고리를 열로 두고 고객 수와 구성비를 계산 | `olist_30d_category_transition_pairs.csv` | 전이 구성비의 분모는 전체 고객이 아니라 이미 재구매한 해당 첫 카테고리 고객이다. |
| 동일·교차 카테고리 전이 | 행동 경로 분해 | 첫·두 번째 대표 카테고리 일치 여부로 반복 경로를 분리 | `olist_30d_category_transition_summary.csv` | 카테고리 내 재구매와 교차판매 가능성을 같은 현상으로 보지 않는다. |
| 전이 리프트 | 기준분포 대비 상대 집중도 | 특정 첫 카테고리의 목적지 구성비를 전체 목적지 구성비로 나눔 | `olist_30d_category_transition_pairs.csv` | 희소 건수에서는 높은 리프트가 쉽게 발생하므로 건수와 함께 해석한다. |
| 시간 간격 민감도 | 운영 정의 민감도·데이터 생성과정 | 동일 시각·1시간·24시간·7일 경계로 두 번째 주문을 분해 | `olist_30d_category_transition_timing_sensitivity.csv` | 근접 주문이 별도 구매인지 주문 분할인지 데이터 생성 과정을 먼저 점검한다. |
| 상품·판매자 집합 비교 | 집합·교집합·합집합 | 첫·두 번째 주문의 고유 product_id와 seller_id 집합을 비교 | `olist_priority_category_over_24h_followup_detail.csv` | 주문 품목 행을 그대로 조인하면 다대다 중복이 생기므로 주문별 집합으로 먼저 변환한다. |
| Jaccard 중첩률 | 집합 유사도 | 상품·판매자 교집합 크기를 합집합 크기로 나눔 | `olist_priority_category_over_24h_followup_detail.csv` | 완전 일치뿐 아니라 부분적으로 겹치는 주문 구성을 측정한다. |
| 후속 구매 행동 분해 | 규칙 기반 행동 유형화 | 동일 상품 구성·부분 중첩·같은 카테고리 다른 상품·교차 카테고리로 분리 | `olist_priority_category_followup_product_patterns.csv` | 군집분석 없이 CRM 질문에 직접 연결되는 규칙으로 행동을 구분한다. |
| 가격 이동 분석 | 쌍체 비교·비율 | 고객별 두 번째 주문 품목금액을 첫 주문과 비교 | `olist_priority_category_followup_price_movement.csv` | 주문 구성과 수량이 달라 가격비를 순수한 지불의향 변화로 해석할 수 없다. |
| As-of 변수 설계 | 시점 일관성·데이터 누수 방지 | 변수별 최초 가용 시점과 기준점별 입력 허용 여부를 표로 관리 | `olist_asof_feature_availability_registry.csv` | 같은 변수도 기준점에 아직 생성되지 않았다면 입력할 수 없다. |
| 기준점별 타깃 재정의 | 랜드마크 설계·예측 창 | 주문 후 24시간 초과~30일과 배송 완료 후 30일 타깃을 분리 | `olist_asof_crm_segment_customer_mart.csv` | 기준점이 바뀌면 미래 결과를 측정하는 시작점도 함께 바뀌어야 한다. |
| 규칙형 실행 세그먼트 | 상호 배타적 규칙·커버리지 | 주문·배송 시점에 사용 가능한 변수만으로 고객을 한 번씩 분류 | `olist_asof_crm_segment_rulebook.csv` | 결과 행동 세그먼트와 사전 타기팅 세그먼트를 구분한다. |
| 양성 포착률과 타기팅 리프트 | 불균형 타기팅 평가 | 선택 고객 비중과 선택된 양성 비중, 전체 대비 양성 농도를 비교 | `olist_asof_crm_targeting_flag_evaluation.csv` | 메시지 관련성이 있어도 양성 포착률이 커버리지와 같으면 타깃 압축 효과는 없다. |
| 확장형 시간 백테스트 | 시계열 외부검증·드리프트 | 과거 누적 학습창으로 바로 다음 미래창을 네 번 예측 | `olist_ml_rolling_time_backtest.csv` | 한 번의 시간 분할 성능이 다른 미래기간에도 반복되는지 확인한다. |
| 부트스트랩 신뢰구간 | 재표본추출·표본 불확실성 | 테스트 고객을 2,000회 복원추출해 지표 분포를 생성 | `olist_ml_test_bootstrap_confidence_intervals.csv` | 점추정치 하나가 아니라 가능한 변동 범위를 함께 판단한다. |
| 세부 고객군 성능 | 이질성·조건부 평가 | 세그먼트·가치·지역·카테고리별 PR-AUC·Lift·Calibration 계산 | `olist_ml_subgroup_performance.csv` | 전체 평균 성능이 모든 고객군에 동일하다고 가정하지 않는다. |
| 선택 대표성 비율 | 분포 비교·모델 편중 진단 | 상위 10% 구성비를 테스트 기준 구성비와 비교 | `olist_ml_top10_subgroup_selection.csv` | 상위군이 특정 시장 구성에 과도하게 의존하는지 확인한다. |
| 모델 동결과 운영 재학습 | 머신러닝 운영·재현성 | 검증된 `C=0.01` 설정을 고정하고 적격 이력 전체로 재학습 | `olist_repurchase_logistic_production.joblib` | 테스트 과적합을 막고 반복 가능한 점수 산출 모델을 만든다. |
| 가치구간 내부 순위 | 층화 타기팅·용량 정책 | Q3·Q4 각각에서 점수 상위 10%를 선택 | `olist_ml_production_score_thresholds.csv` | 전체 상위군의 저가 고객 과다선택 위험을 분리한다. |
| PSI | 모델 모니터링·데이터 드리프트 | 기준 점수·특성 분포를 미래 배치와 비교 | `olist_ml_monitoring_reference.csv` | 입력 고객 구성이 학습 시점과 달라지는지 조기에 감지한다. |
| 증분효과와 ITT | 인과추론·실험분석 | 고객을 무작위 배정하고 원래 배정군 기준으로 30일 재구매율 비교 | `olist_crm_experiment_protocol.csv` | 모델 예측력과 CRM 접촉의 인과효과를 구분한다. |
| 층화 무작위화 | 실험설계·표본 균형 | 가치구간·지역·점수 5분위 안에서 고정 해시로 1:1 배정 | `olist_crm_experiment_assignment_dry_run.csv` | 중요한 사전 특성의 우연한 군 간 불균형을 줄인다. |
| 검정력과 MDE | 통계적 의사결정·표본수 설계 | 기준율 1.77%에서 효과 시나리오별 필요 표본 계산 | `olist_crm_experiment_power_scenarios.csv` | 효과가 없다는 결론과 표본이 부족한 상황을 구분한다. |
| Guardrail KPI | 실험 운영·의사결정 | 수신거부·불만·접촉비 차감 후 증분 공헌이익 감시 | `olist_crm_experiment_kpi_registry.csv` | 재구매율 개선이 고객 경험이나 수익성 악화와 교환되지 않게 한다. |
| 처치 충실도 | 실험 운영·인과추론 | 실험군 1회·대조군 0회와 템플릿·시점·혜택을 고정 | `olist_crm_playbook.csv` | 계획한 처치와 실제 집행이 다른 문제를 감지한다. |
| 발송 제외와 빈도 제한 | CRM 운영·고객 보호 | 동의·수신거부·취소·중복·30일 1회 상한 적용 | `olist_crm_suppression_and_frequency_rules.csv` | 고객 보호와 실험 오염 방지를 함께 관리한다. |
| 이벤트 계측 명세 | 데이터 엔지니어링·실험 분석 | 배정부터 비용까지 필수 키와 시점을 정의 | `olist_crm_event_tracking_schema.csv` | CRM·주문·재무 데이터가 같은 실험 단위로 연결되게 한다. |
| Newcombe-Wilson 신뢰구간 | 이항 추론·희소 사건 분석 | 실험군과 대조군 재구매율 차이의 95% 구간 계산 | `olist_crm_synthetic_itt_analysis_results.csv` | 낮은 재구매율에서 효과 크기의 불확실성을 안정적으로 제시한다. |
| 결과 관찰 성숙도 | 코호트 분석·실험 운영 | 배정 후 30일을 완전히 관찰한 고객만 결과 분석 | `olist_crm_experiment_analysis_spec.csv` | 아직 발생할 수 있는 미래 재구매를 음성으로 잘못 처리하지 않는다. |
| 합성 시나리오 검증 | 분석 엔지니어링·테스트 | 무효과·+30%·+75% 사건 수로 분석 코드 점검 | `olist_crm_synthetic_scenario_summary.csv` | 실제 결과 전에 계산과 해석 규칙의 오류를 발견한다. |
| 근거 수준 | 분석 커뮤니케이션·의사결정 | 기술통계·예측검증·인과실험·합성검증을 계층화 | `olist_final_report_evidence_registry.csv` | 분석방법보다 강한 결론을 내리는 오류를 방지한다. |
| 분석 추적성 | 분석 엔지니어링·재현성 | 보고서 핵심 수치를 원천 CSV와 생성 코드에 연결 | `olist_final_report_key_metrics.csv` | 수동 복사 오류를 줄이고 결과 갱신을 재현 가능하게 한다. |
| Executive Summary | 비즈니스 분석 커뮤니케이션 | 수준·모델·실험·제약을 핵심 의사결정으로 압축 | `olist_repurchase_final_report.md` | 상세 분석을 의사결정자가 빠르게 이해하도록 한다. |
| 범위 동결 | 분석 거버넌스·프로젝트 관리 | 추가 데이터 불가 조건에서 최종 질문과 산출물 확정 | `olist_repurchase_final_report.md` | 실현 불가능한 로드맵과 검증 가능한 결론을 구분한다. |
| 분석 종료선 | 비판적 분석·근거 커뮤니케이션 | 관찰·예측·인과 근거의 최종 경계를 명시 | `olist_final_report_evidence_registry.csv` | 데이터가 답하지 못하는 질문을 추측으로 채우지 않는다. |
| 보고서 시각화 | 데이터 시각화·의사결정 지원 | 재구매·외부·기회·모델·실험 6개 차트 생성 | `reports/figures/` | 핵심 관계를 빠르게 이해하되 한계를 함께 전달한다. |
| Action Item 우선순위 | 의사결정 분석·자원 배분 | 영향·근거·긴급성·노력·위험 루브릭 평가 | `olist_decision_action_priority.csv` | 분석 제언을 제한된 자원에서의 실행 순서로 전환한다. |
| 리스크 레지스터 | 실험 운영·리스크 관리 | Side Effect별 지표·완화책·중단 조건 연결 | `olist_crm_risk_register.csv` | Primary 개선과 함께 발생할 부작용을 사전에 통제한다. |
| 사후 의사결정 규칙 | 실험분석·거버넌스 | GO·ITERATE·STOP·NO DECISION 사전 정의 | `olist_crm_experiment_decision_rules.csv` | 결과를 본 뒤 기준을 바꾸는 사후 선택을 방지한다. |

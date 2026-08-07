"""Build leakage-safe as-of feature registry and rule-based CRM segment candidates."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

MART_PATH = PROCESSED / "olist_first_purchase_customer_mart.csv"
ORDERS_PATH = PROCESSED / "retention_orders.csv"
TRANSITION_PATH = PROCESSED / "olist_customer_30d_category_transition.csv"

CUSTOMER_PATH = PROCESSED / "olist_asof_crm_segment_customer_mart.csv"
FEATURE_PATH = TABLES / "olist_asof_feature_availability_registry.csv"
SUMMARY_PATH = TABLES / "olist_asof_crm_segment_summary.csv"
RULEBOOK_PATH = TABLES / "olist_asof_crm_segment_rulebook.csv"
FLAG_PATH = TABLES / "olist_asof_crm_targeting_flag_evaluation.csv"
CHECK_PATH = TABLES / "olist_asof_crm_segment_build_summary.csv"

EXPECTED_ORDER_ELIGIBLE = 86_772
EXPECTED_ORDER_REPEAT = 1_373
EXPECTED_OVER_24H_REPEAT = 535
SELLER_EXPLORATION_CATEGORIES = {"watches_gifts", "auto", "toys"}

ORDER_TIME_INPUTS = {
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "customer_state", "customer_city", "item_count", "product_count",
    "seller_count", "category_count", "item_price_sum", "freight_value_sum",
    "primary_category", "payment_value_sum", "payment_installments_max",
    "payment_type_count", "primary_payment_type", "freight_share",
    "order_value_band", "item_count_group", "seller_count_group",
    "installment_group",
}
DELIVERY_INPUTS = {
    "order_delivered_customer_date", "delivery_days", "delivery_delay_days",
    "delivery_experience_group",
}
REVIEW_INPUTS = {
    "review_count", "review_score_mean", "review_score_min",
    "review_comment_present", "review_present_flag", "review_score_group",
}
IDENTIFIERS = {"order_id", "customer_id", "customer_unique_id"}
QA_ONLY = {"first_purchase_timestamp", "first_timestamp_match_flag"}
LABEL_OR_ELIGIBILITY = {"eligible_30d", "repeat_within_30d"}


def build_feature_registry(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column in ORDER_TIME_INPUTS:
            availability, role = "order_time", "candidate_input"
            order_allowed, delivery_allowed = True, True
            reason = "첫 주문 확정 시 생성되거나 계산 가능한 정보"
        elif column in DELIVERY_INPUTS:
            availability, role = "delivery_complete", "candidate_input"
            order_allowed, delivery_allowed = False, True
            reason = "실제 배송 완료 이후에만 확정되는 정보"
        elif column in REVIEW_INPUTS:
            availability, role = "review_event_or_later", "post_delivery_input"
            order_allowed, delivery_allowed = False, False
            reason = "배송 완료 시점에는 리뷰가 아직 작성되지 않았을 수 있음"
        elif column in IDENTIFIERS:
            availability, role = "order_time", "identifier_not_model_input"
            order_allowed, delivery_allowed = False, False
            reason = "행 연결용 식별자이며 일반화 가능한 설명변수가 아님"
        elif column in QA_ONLY:
            availability, role = "dataset_qa", "qa_not_model_input"
            order_allowed, delivery_allowed = False, False
            reason = "마트 구축 검증용 변수"
        elif column in LABEL_OR_ELIGIBILITY:
            availability, role = "future_observation", "target_or_eligibility"
            order_allowed, delivery_allowed = False, False
            reason = "예측 기준점 이후의 결과 또는 충분한 관찰기간 조건"
        else:
            availability, role = "manual_review", "unclassified"
            order_allowed, delivery_allowed = False, False
            reason = "명시적 가용성 검토 필요"
        rows.append({
            "variable": column,
            "earliest_availability": availability,
            "variable_role": role,
            "allowed_as_order_time_input": order_allowed,
            "allowed_as_delivery_complete_input": delivery_allowed,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def add_delivery_target(mart: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    orders = pd.read_csv(ORDERS_PATH, low_memory=False)
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="raise"
    )
    observation_end = orders["order_purchase_timestamp"].max()
    later = orders.loc[
        orders["customer_order_number"].gt(1),
        ["customer_unique_id", "order_purchase_timestamp"],
    ].rename(columns={"order_purchase_timestamp": "later_purchase_timestamp"})
    delivery = mart[[
        "customer_unique_id", "order_delivered_customer_date"
    ]].copy()
    delivery["order_delivered_customer_date"] = pd.to_datetime(
        delivery["order_delivered_customer_date"], errors="coerce"
    )
    delivery["eligible_30d_after_delivery"] = (
        delivery["order_delivered_customer_date"].notna()
        & delivery["order_delivered_customer_date"].le(observation_end - pd.Timedelta(days=30))
    )
    candidates = delivery[[
        "customer_unique_id", "order_delivered_customer_date"
    ]].merge(later, on="customer_unique_id", how="left")
    candidates["days_after_delivery"] = (
        candidates["later_purchase_timestamp"]
        - candidates["order_delivered_customer_date"]
    ).dt.total_seconds() / 86_400
    positives = candidates.loc[
        candidates["days_after_delivery"].gt(0)
        & candidates["days_after_delivery"].le(30),
        "customer_unique_id",
    ].drop_duplicates()
    delivery["repeat_within_30d_after_delivery"] = delivery[
        "customer_unique_id"
    ].isin(positives)
    mart = mart.merge(
        delivery[[
            "customer_unique_id", "eligible_30d_after_delivery",
            "repeat_within_30d_after_delivery",
        ]],
        on="customer_unique_id", how="left", validate="one_to_one",
    )
    return mart, observation_end


def build_customer_mart() -> tuple[pd.DataFrame, pd.Timestamp]:
    mart = pd.read_csv(MART_PATH, low_memory=False)
    transitions = pd.read_csv(
        TRANSITION_PATH,
        usecols=["customer_unique_id", "days_to_second_purchase"],
    )
    over_24h_customers = set(transitions.loc[
        transitions["days_to_second_purchase"].gt(1), "customer_unique_id"
    ])
    mart["target_repeat_over_24h_within_30d"] = mart[
        "customer_unique_id"
    ].isin(over_24h_customers)
    mart["seller_exploration_category_flag"] = mart["primary_category"].isin(
        SELLER_EXPLORATION_CATEGORIES
    )
    mart["high_first_order_value_flag"] = mart["order_value_band"].eq("Q4_high")
    mart["delayed_delivery_flag"] = mart["delivery_experience_group"].isin(
        ["late_1_3d", "late_4_7d", "late_8d_plus"]
    )
    mart["order_time_segment"] = np.select(
        [
            mart["seller_exploration_category_flag"] & mart["high_first_order_value_flag"],
            mart["seller_exploration_category_flag"],
            mart["high_first_order_value_flag"],
        ],
        [
            "seller_category_high_value_candidate",
            "seller_category_standard_value_candidate",
            "other_category_high_value_candidate",
        ],
        default="general_order_time_candidate",
    )
    mart["delivery_complete_segment"] = np.select(
        [
            mart["delayed_delivery_flag"] & mart["seller_exploration_category_flag"],
            mart["delayed_delivery_flag"],
            mart["delivery_experience_group"].eq("on_time_or_early")
            & mart["seller_exploration_category_flag"],
            mart["delivery_experience_group"].eq("on_time_or_early"),
        ],
        [
            "delayed_seller_category_candidate",
            "delayed_other_category_candidate",
            "ontime_seller_category_candidate",
            "ontime_other_category_candidate",
        ],
        default="delivery_status_unknown_candidate",
    )
    mart, observation_end = add_delivery_target(mart)
    return mart, observation_end


def metric_row(frame, point, segment, strategy, primary_kpi, evidence_note):
    if point == "order_time":
        eligible = frame.loc[frame["eligible_30d"].eq(True)]
        official_positive = int(eligible["repeat_within_30d"].sum())
        operational_positive = int(eligible["target_repeat_over_24h_within_30d"].sum())
        return {
            "prediction_point": point,
            "segment_name": segment,
            "eligible_customer_count": len(eligible),
            "customer_share_pct": len(eligible) / EXPECTED_ORDER_ELIGIBLE * 100,
            "observed_repeat_count": official_positive,
            "observed_repeat_rate_pct": official_positive / len(eligible) * 100,
            "operational_over_24h_repeat_count": operational_positive,
            "operational_over_24h_repeat_rate_pct": operational_positive / len(eligible) * 100,
            "median_first_payment_value": eligible["payment_value_sum"].median(),
            "strategy_hypothesis": strategy,
            "primary_kpi_candidate": primary_kpi,
            "evidence_note": evidence_note,
        }
    eligible = frame.loc[frame["eligible_30d_after_delivery"].eq(True)]
    positive = int(eligible["repeat_within_30d_after_delivery"].sum())
    return {
        "prediction_point": point,
        "segment_name": segment,
        "eligible_customer_count": len(eligible),
        "customer_share_pct": np.nan,
        "observed_repeat_count": positive,
        "observed_repeat_rate_pct": (
            positive / len(eligible) * 100 if len(eligible) else np.nan
        ),
        "operational_over_24h_repeat_count": np.nan,
        "operational_over_24h_repeat_rate_pct": np.nan,
        "median_first_payment_value": (
            eligible["payment_value_sum"].median() if len(eligible) else np.nan
        ),
        "strategy_hypothesis": strategy,
        "primary_kpi_candidate": primary_kpi,
        "evidence_note": evidence_note,
    }


def build_summary(mart: pd.DataFrame) -> pd.DataFrame:
    order_rules = {
        "seller_category_high_value_candidate": (
            "동일 판매자 내 카테고리 상품 탐색; 혜택 비용 별도 통제",
            "30일 24시간 초과 재구매율", "관찰 행동 기반 가설; 효과 미검증",
        ),
        "seller_category_standard_value_candidate": (
            "동일 판매자 내 카테고리 상품 탐색",
            "30일 24시간 초과 재구매율", "관찰 행동 기반 가설; 효과 미검증",
        ),
        "other_category_high_value_candidate": (
            "고가 첫 주문 고객의 일반 카테고리 재방문 탐색",
            "30일 24시간 초과 재구매율", "가치 보존용 운영 분리; 추천 경로 근거 없음",
        ),
        "general_order_time_candidate": (
            "향후 예측 점수 기반 우선순위 적용",
            "30일 24시간 초과 재구매율", "현재 규칙만으로 세부 메시지 미확정",
        ),
    }
    delivery_rules = {
        "delayed_seller_category_candidate": (
            "배송 지연 사과·회복 후 카테고리 탐색",
            "배송 완료 후 30일 재구매율", "서비스 회복 가설; 기존 분석에서 인과효과 미확인",
        ),
        "delayed_other_category_candidate": (
            "배송 지연 사과·서비스 회복",
            "배송 완료 후 30일 재구매율", "서비스 회복 가설; 기존 분석에서 인과효과 미확인",
        ),
        "ontime_seller_category_candidate": (
            "배송 완료 확인 후 동일 판매자 카테고리 탐색",
            "배송 완료 후 30일 재구매율", "관찰 행동 기반 가설; 효과 미검증",
        ),
        "ontime_other_category_candidate": (
            "배송 완료 확인 후 일반 재방문",
            "배송 완료 후 30일 재구매율", "향후 예측 점수로 대상 압축 필요",
        ),
        "delivery_status_unknown_candidate": (
            "발송 보류 및 데이터 상태 점검",
            "데이터 완전성", "배송 정보 미확정 또는 누락",
        ),
    }
    rows = []
    for segment, values in order_rules.items():
        rows.append(metric_row(
            mart.loc[mart["order_time_segment"].eq(segment)],
            "order_time", segment, *values,
        ))
    delivery_denominator = int(mart["eligible_30d_after_delivery"].sum())
    for segment, values in delivery_rules.items():
        row = metric_row(
            mart.loc[mart["delivery_complete_segment"].eq(segment)],
            "delivery_complete", segment, *values,
        )
        row["customer_share_pct"] = (
            row["eligible_customer_count"] / delivery_denominator * 100
            if delivery_denominator else np.nan
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(6)
    return result


def build_rulebook() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "prediction_point": "order_time",
            "rule_order": 1,
            "segment_name": "seller_category_high_value_candidate",
            "selection_rule": "primary_category in {watches_gifts, auto, toys} AND order_value_band = Q4_high",
            "available_inputs_only": True,
            "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "order_time", "rule_order": 2,
            "segment_name": "seller_category_standard_value_candidate",
            "selection_rule": "primary_category in {watches_gifts, auto, toys} AND order_value_band != Q4_high",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "order_time", "rule_order": 3,
            "segment_name": "other_category_high_value_candidate",
            "selection_rule": "primary_category not in {watches_gifts, auto, toys} AND order_value_band = Q4_high",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "order_time", "rule_order": 4,
            "segment_name": "general_order_time_candidate",
            "selection_rule": "remaining order-time eligible customers",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "delivery_complete", "rule_order": 1,
            "segment_name": "delayed_seller_category_candidate",
            "selection_rule": "delivery is late AND primary_category in {watches_gifts, auto, toys}",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "delivery_complete", "rule_order": 2,
            "segment_name": "delayed_other_category_candidate",
            "selection_rule": "delivery is late AND primary_category not in {watches_gifts, auto, toys}",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "delivery_complete", "rule_order": 3,
            "segment_name": "ontime_seller_category_candidate",
            "selection_rule": "delivery on time or early AND primary_category in {watches_gifts, auto, toys}",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "delivery_complete", "rule_order": 4,
            "segment_name": "ontime_other_category_candidate",
            "selection_rule": "delivery on time or early AND primary_category not in {watches_gifts, auto, toys}",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
        {
            "prediction_point": "delivery_complete", "rule_order": 5,
            "segment_name": "delivery_status_unknown_candidate",
            "selection_rule": "delivery experience unknown",
            "available_inputs_only": True, "future_behavior_used_as_input": False,
        },
    ])


def build_flag_evaluation(mart: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = mart.loc[mart["eligible_30d"].eq(True)]
    order_positive = int(order["target_repeat_over_24h_within_30d"].sum())
    order_rate = order_positive / len(order)
    for name, mask in {
        "seller_exploration_category_flag": order["seller_exploration_category_flag"],
        "high_first_order_value_flag": order["high_first_order_value_flag"],
        "seller_category_and_high_value_flag": (
            order["seller_exploration_category_flag"] & order["high_first_order_value_flag"]
        ),
    }.items():
        selected = order.loc[mask]
        positives = int(selected["target_repeat_over_24h_within_30d"].sum())
        rate = positives / len(selected)
        rows.append({
            "prediction_point": "order_time",
            "target_definition": "over_24h_to_30d_after_first_order",
            "targeting_flag": name,
            "eligible_customer_count": len(order),
            "selected_customer_count": len(selected),
            "customer_coverage_pct": len(selected) / len(order) * 100,
            "positive_count": positives,
            "positive_capture_pct": positives / order_positive * 100,
            "selected_positive_rate_pct": rate * 100,
            "overall_positive_rate_pct": order_rate * 100,
            "lift_vs_overall": rate / order_rate,
        })
    delivery = mart.loc[mart["eligible_30d_after_delivery"].eq(True)]
    delivery_positive = int(delivery["repeat_within_30d_after_delivery"].sum())
    delivery_rate = delivery_positive / len(delivery)
    for name, mask in {
        "seller_exploration_category_flag": delivery["seller_exploration_category_flag"],
        "delayed_delivery_flag": delivery["delayed_delivery_flag"],
        "delayed_seller_category_flag": (
            delivery["delayed_delivery_flag"] & delivery["seller_exploration_category_flag"]
        ),
    }.items():
        selected = delivery.loc[mask]
        positives = int(selected["repeat_within_30d_after_delivery"].sum())
        rate = positives / len(selected)
        rows.append({
            "prediction_point": "delivery_complete",
            "target_definition": "over_0h_to_30d_after_delivery_complete",
            "targeting_flag": name,
            "eligible_customer_count": len(delivery),
            "selected_customer_count": len(selected),
            "customer_coverage_pct": len(selected) / len(delivery) * 100,
            "positive_count": positives,
            "positive_capture_pct": positives / delivery_positive * 100,
            "selected_positive_rate_pct": rate * 100,
            "overall_positive_rate_pct": delivery_rate * 100,
            "lift_vs_overall": rate / delivery_rate,
        })
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(6)
    return result


def validate(mart, registry, summary, rulebook, flags):
    order_eligible = mart.loc[mart["eligible_30d"].eq(True)]
    delivery_eligible = mart.loc[mart["eligible_30d_after_delivery"].eq(True)]
    order_summary = summary.loc[summary["prediction_point"].eq("order_time")]
    delivery_summary = summary.loc[summary["prediction_point"].eq("delivery_complete")]
    checks = [
        ("customer_row_count", len(mart), 93_358, len(mart) == 93_358),
        ("unique_customer_count", mart["customer_unique_id"].nunique(), len(mart), mart["customer_unique_id"].is_unique),
        ("order_eligible_count", len(order_eligible), EXPECTED_ORDER_ELIGIBLE, len(order_eligible) == EXPECTED_ORDER_ELIGIBLE),
        ("order_repeat_count", int(order_eligible["repeat_within_30d"].sum()), EXPECTED_ORDER_REPEAT, order_eligible["repeat_within_30d"].sum() == EXPECTED_ORDER_REPEAT),
        ("over_24h_repeat_count", int(order_eligible["target_repeat_over_24h_within_30d"].sum()), EXPECTED_OVER_24H_REPEAT, order_eligible["target_repeat_over_24h_within_30d"].sum() == EXPECTED_OVER_24H_REPEAT),
        ("order_segment_count_sum", int(order_summary["eligible_customer_count"].sum()), len(order_eligible), order_summary["eligible_customer_count"].sum() == len(order_eligible)),
        ("delivery_segment_count_sum", int(delivery_summary["eligible_customer_count"].sum()), len(delivery_eligible), delivery_summary["eligible_customer_count"].sum() == len(delivery_eligible)),
        ("unclassified_feature_count", int(registry["variable_role"].eq("unclassified").sum()), 0, ~registry["variable_role"].eq("unclassified").any()),
        ("review_allowed_at_delivery_count", int(registry.loc[registry["variable"].isin(REVIEW_INPUTS), "allowed_as_delivery_complete_input"].sum()), 0, ~registry.loc[registry["variable"].isin(REVIEW_INPUTS), "allowed_as_delivery_complete_input"].any()),
        ("future_behavior_rule_count", int(rulebook["future_behavior_used_as_input"].sum()), 0, ~rulebook["future_behavior_used_as_input"].any()),
        ("segment_rule_count", len(rulebook), 9, len(rulebook) == 9),
        ("targeting_flag_row_count", len(flags), 6, len(flags) == 6),
    ]
    return pd.DataFrame(checks, columns=["check_name", "actual_value", "expected_value", "passed"])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mart, observation_end = build_customer_mart()
    registry = build_feature_registry(list(pd.read_csv(MART_PATH, nrows=0).columns))
    summary = build_summary(mart)
    rulebook = build_rulebook()
    flags = build_flag_evaluation(mart)
    checks = validate(mart, registry, summary, rulebook, flags)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    mart.to_csv(CUSTOMER_PATH, index=False)
    registry.to_csv(FEATURE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    rulebook.to_csv(RULEBOOK_PATH, index=False)
    flags.to_csv(FLAG_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    print(f"observation_end={observation_end}")
    print(checks.to_string(index=False))
    print("\nSegment summary")
    print(summary.to_string(index=False))
    print("\nFeature availability")
    print(registry.groupby(["earliest_availability", "variable_role"]).size().to_string())
    print("\nTargeting flag evaluation")
    print(flags.to_string(index=False))


if __name__ == "__main__":
    main()

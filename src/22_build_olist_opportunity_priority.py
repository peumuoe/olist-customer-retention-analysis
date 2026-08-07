"""Olist 개선 기회 후보를 다기준·가중치 시나리오로 우선순위화한다."""

from itertools import combinations_with_replacement
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "olist_first_purchase_customer_mart.csv"
SEGMENT_PATH = ROOT / "outputs" / "tables" / "olist_first_purchase_segment_retention.csv"
TABLES = ROOT / "outputs" / "tables"
PRIORITY_PATH = TABLES / "olist_improvement_opportunity_priority.csv"
SCENARIO_PATH = TABLES / "olist_improvement_opportunity_weight_scenarios.csv"
RUBRIC_PATH = TABLES / "olist_improvement_opportunity_actionability_rubric.csv"
OVERLAP_PATH = TABLES / "olist_improvement_opportunity_top10_overlap.csv"
CHECK_PATH = TABLES / "olist_improvement_opportunity_build_summary.csv"

OVERALL_RATE = 1.5823076568478311
MIN_CUSTOMERS = 500
EXPECTED_ELIGIBLE = 86_772

DIMENSIONS = [
    "primary_category", "customer_state", "order_value_band",
    "item_count_group", "seller_count_group", "primary_payment_type",
    "installment_group", "delivery_experience_group", "review_score_group",
]

ACTIONABILITY = {
    "primary_category": (4, "카테고리별 보완상품·재구매 주기 메시지를 설계할 수 있음"),
    "customer_state": (2, "지역 타깃은 가능하지만 관찰 격차에 직접 대응하는 메시지가 불명확함"),
    "order_value_band": (3, "금액 기반 혜택 차등은 가능하나 가격 민감도와 수익성 검토가 필요함"),
    "item_count_group": (3, "장바구니·번들 전략과 연결 가능하지만 고객 구매의향의 대용치일 수 있음"),
    "seller_count_group": (2, "플랫폼이 식별 가능하지만 판매자 구성 자체를 CRM으로 바꾸기 어려움"),
    "primary_payment_type": (2, "결제수단별 메시지는 가능하나 재구매 제안과의 직접 연결성이 낮음"),
    "installment_group": (2, "할부 기반 혜택은 가능하지만 금융·마진·정책 제약 검토가 필요함"),
    "delivery_experience_group": (4, "배송 후 서비스 회복 메시지를 설계할 수 있으나 주문 직후에는 사용 불가"),
    "review_score_group": (4, "리뷰 기반 서비스 회복·감사 메시지가 가능하나 리뷰 미작성 선택편향이 있음"),
}

SCENARIOS = {
    "balanced": {
        "size_score": 0.25, "gap_score": 0.25,
        "first_value_score": 0.25, "actionability_score": 0.25,
    },
    "scale_focused": {
        "size_score": 0.40, "gap_score": 0.30,
        "first_value_score": 0.15, "actionability_score": 0.15,
    },
    "action_focused": {
        "size_score": 0.25, "gap_score": 0.25,
        "first_value_score": 0.15, "actionability_score": 0.35,
    },
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """첫 구매 고객 마트와 기존 세그먼트 결과를 읽는다."""
    for path in [MART_PATH, SEGMENT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    mart = pd.read_csv(MART_PATH, low_memory=False)
    mart = mart.loc[mart["eligible_30d"].eq(True)].copy()
    segments = pd.read_csv(SEGMENT_PATH)
    return mart, segments


def build_rubric() -> pd.DataFrame:
    """주관적 개입 가능성 판단을 추적 가능한 표로 만든다."""
    return pd.DataFrame([
        {
            "segment_dimension": dimension,
            "actionability_raw_1_to_5": score,
            "actionability_reason": reason,
            "score_owner": "분석 설계 가정; 비즈니스 검토 필요",
        }
        for dimension, (score, reason) in ACTIONABILITY.items()
    ])


def add_first_value(
    mart: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """후보별 첫 주문 결제금액 중앙값·평균을 결합한다."""
    rows = []
    for candidate in candidates.itertuples(index=False):
        values = mart.loc[
            mart[candidate.segment_dimension].astype("string").fillna("unknown")
            .eq(str(candidate.segment_value)),
            "payment_value_sum",
        ].dropna()
        rows.append({
            "segment_dimension": candidate.segment_dimension,
            "segment_value": str(candidate.segment_value),
            "first_payment_valid_count": len(values),
            "median_first_payment_value": float(values.median()),
            "mean_first_payment_value": float(values.mean()),
        })
    value = pd.DataFrame(rows)
    candidates = candidates.copy()
    candidates["segment_value"] = candidates["segment_value"].astype(str)
    return candidates.merge(
        value, on=["segment_dimension", "segment_value"],
        how="left", validate="one_to_one",
    )


def build_candidates(
    mart: pd.DataFrame, segments: pd.DataFrame, rubric: pd.DataFrame
) -> pd.DataFrame:
    """규모와 음의 격차 기준을 충족하는 후보와 네 평가축을 만든다."""
    candidates = segments.loc[
        segments["segment_dimension"].isin(DIMENSIONS)
        & segments["eligible_customer_count"].ge(MIN_CUSTOMERS)
        & segments["retention_rate_pct"].lt(OVERALL_RATE)
    ].copy()
    candidates["retention_gap_below_overall_pct_point"] = (
        OVERALL_RATE - candidates["retention_rate_pct"]
    )
    candidates = add_first_value(mart, candidates)
    candidates = candidates.merge(
        rubric[["segment_dimension", "actionability_raw_1_to_5"]],
        on="segment_dimension", how="left", validate="many_to_one",
    )
    # 각 축은 후보 집합 안에서의 상대적 백분위다. 값이 클수록 우선순위가 높다.
    candidates["size_score"] = candidates["eligible_customer_count"].rank(
        pct=True, method="average"
    )
    candidates["gap_score"] = candidates[
        "retention_gap_below_overall_pct_point"
    ].rank(pct=True, method="average")
    candidates["first_value_score"] = candidates[
        "median_first_payment_value"
    ].rank(pct=True, method="average")
    candidates["actionability_score"] = (
        candidates["actionability_raw_1_to_5"] - 1
    ) / 4
    return candidates


def score_scenarios(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """세 가중치 시나리오 점수·순위와 순위 강건성을 계산한다."""
    scenario_rows = []
    for scenario, weights in SCENARIOS.items():
        current = candidates[["segment_dimension", "segment_value"]].copy()
        current["scenario"] = scenario
        current["priority_score"] = sum(
            candidates[column] * weight for column, weight in weights.items()
        )
        current["priority_rank"] = current["priority_score"].rank(
            ascending=False, method="min"
        ).astype(int)
        for column, weight in weights.items():
            current[f"weight_{column}"] = weight
        scenario_rows.append(current)
    scenarios = pd.concat(scenario_rows, ignore_index=True)
    stability = scenarios.groupby(
        ["segment_dimension", "segment_value"], as_index=False
    ).agg(
        mean_priority_rank=("priority_rank", "mean"),
        best_priority_rank=("priority_rank", "min"),
        worst_priority_rank=("priority_rank", "max"),
        top10_scenario_count=("priority_rank", lambda x: int(x.le(10).sum())),
    )
    balanced = scenarios.loc[
        scenarios["scenario"].eq("balanced"),
        ["segment_dimension", "segment_value", "priority_score", "priority_rank"],
    ].rename(columns={
        "priority_score": "balanced_priority_score",
        "priority_rank": "balanced_priority_rank",
    })
    priority = candidates.merge(
        balanced, on=["segment_dimension", "segment_value"],
        how="left", validate="one_to_one",
    ).merge(
        stability, on=["segment_dimension", "segment_value"],
        how="left", validate="one_to_one",
    )
    priority["rank_robustness"] = priority["top10_scenario_count"].map({
        3: "top10_all_scenarios", 2: "top10_two_scenarios",
        1: "top10_one_scenario", 0: "outside_top10",
    })
    numeric = [
        "retention_gap_below_overall_pct_point", "median_first_payment_value",
        "mean_first_payment_value", "size_score", "gap_score",
        "first_value_score", "actionability_score", "balanced_priority_score",
        "mean_priority_rank",
    ]
    priority[numeric] = priority[numeric].round(6)
    scenarios["priority_score"] = scenarios["priority_score"].round(6)
    return priority.sort_values("balanced_priority_rank"), scenarios


def membership_mask(
    mart: pd.DataFrame, dimension: str, value: str
) -> pd.Series:
    return mart[dimension].astype("string").fillna("unknown").eq(str(value))


def build_overlap(mart: pd.DataFrame, priority: pd.DataFrame) -> pd.DataFrame:
    """균형형 상위 10개 후보의 교집합과 Jaccard 중첩률을 계산한다."""
    top = priority.nsmallest(10, "balanced_priority_rank")
    candidate_keys = list(top[["segment_dimension", "segment_value"]].itertuples(
        index=False, name=None
    ))
    masks = {key: membership_mask(mart, *key) for key in candidate_keys}
    rows = []
    for left, right in combinations_with_replacement(candidate_keys, 2):
        intersection = int((masks[left] & masks[right]).sum())
        union = int((masks[left] | masks[right]).sum())
        rows.append({
            "left_dimension": left[0], "left_value": left[1],
            "right_dimension": right[0], "right_value": right[1],
            "intersection_customer_count": intersection,
            "union_customer_count": union,
            "jaccard_overlap_pct": intersection / union * 100 if union else 0,
        })
    result = pd.DataFrame(rows)
    result["jaccard_overlap_pct"] = result["jaccard_overlap_pct"].round(6)
    return result


def validate(
    mart: pd.DataFrame, priority: pd.DataFrame, scenarios: pd.DataFrame,
    rubric: pd.DataFrame, overlap: pd.DataFrame,
) -> pd.DataFrame:
    """분모·점수·가중치·순위·중첩 구조를 검증한다."""
    checks = {
        "eligible_customer_rows": (len(mart), EXPECTED_ELIGIBLE),
        "duplicate_customers": (int(mart["customer_unique_id"].duplicated().sum()), 0),
        "candidate_key_duplicates": (
            int(priority.duplicated(["segment_dimension", "segment_value"]).sum()), 0
        ),
        "all_candidates_min_500": (
            bool(priority["eligible_customer_count"].ge(MIN_CUSTOMERS).all()), True
        ),
        "all_candidates_below_overall": (
            bool(priority["retention_rate_pct"].lt(OVERALL_RATE).all()), True
        ),
        "rubric_dimensions": (rubric["segment_dimension"].nunique(), len(DIMENSIONS)),
        "missing_actionability": (int(priority["actionability_raw_1_to_5"].isna().sum()), 0),
        "component_scores_in_0_1": (
            bool(priority[[
                "size_score", "gap_score", "first_value_score", "actionability_score"
            ]].apply(lambda x: x.between(0, 1).all()).all()), True
        ),
        "scenario_count": (scenarios["scenario"].nunique(), 3),
        "scenario_rows_reconcile": (len(scenarios), len(priority) * 3),
        "all_scenario_weights_sum_1": (
            all(abs(sum(weights.values()) - 1) < 1e-12 for weights in SCENARIOS.values()),
            True,
        ),
        "balanced_top10": (int(priority["balanced_priority_rank"].le(10).sum()), 10),
        "top10_overlap_rows": (len(overlap), 55),
        "overlap_in_0_100": (bool(overlap["jaccard_overlap_pct"].between(0, 100).all()), True),
        "valid_first_payment_counts": (
            bool(priority["first_payment_valid_count"].le(priority["eligible_customer_count"]).all()),
            True,
        ),
    }
    failures = {key: value for key, value in checks.items() if value[0] != value[1]}
    if failures:
        raise AssertionError(f"개선 기회 우선순위 검증 실패: {failures}")
    return pd.DataFrame([
        {"check_name": key, "actual": actual, "expected": expected, "status": "PASS"}
        for key, (actual, expected) in checks.items()
    ])


def save(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    print(f"생성: {path.relative_to(ROOT)} ({len(frame):,}행)")


def main() -> None:
    mart, segments = load_data()
    rubric = build_rubric()
    candidates = build_candidates(mart, segments, rubric)
    priority, scenarios = score_scenarios(candidates)
    overlap = build_overlap(mart, priority)
    checks = validate(mart, priority, scenarios, rubric, overlap)
    save(priority, PRIORITY_PATH)
    save(scenarios, SCENARIO_PATH)
    save(rubric, RUBRIC_PATH)
    save(overlap, OVERLAP_PATH)
    save(checks, CHECK_PATH)


if __name__ == "__main__":
    main()

"""Olist·H&M·REES46 기간별 재구매율을 비교 가능한 형태로 통합한다."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
OLIST_PATH = TABLES / "retention_window_summary.csv"
HM_PATH = TABLES / "hm_retention_window_summary.csv"
REES46_PATH = TABLES / "rees46_retention_window_summary.csv"
COMPARISON_PATH = TABLES / "external_retention_comparison.csv"
METHODOLOGY_PATH = TABLES / "external_retention_comparison_methodology.csv"
CHECK_PATH = TABLES / "external_retention_comparison_build_summary.csv"
WINDOWS = [30, 60, 90, 180, 365]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """세 결과표를 읽고 기간 키의 고유성을 확인한다."""
    for path in [OLIST_PATH, HM_PATH, REES46_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    olist = pd.read_csv(OLIST_PATH)
    hm = pd.read_csv(HM_PATH)
    rees46 = pd.read_csv(REES46_PATH)
    for name, frame in [("Olist", olist), ("H&M", hm), ("REES46", rees46)]:
        if frame["window_days"].duplicated().any():
            raise AssertionError(f"{name}: 기간 키가 중복됩니다.")
    return olist, hm, rees46


def build_comparison(
    olist: pd.DataFrame, hm: pd.DataFrame, rees46: pd.DataFrame
) -> pd.DataFrame:
    """기간별 비율·분자·분모와 Olist 대비 격차를 만든다."""
    base = pd.DataFrame({"window_days": WINDOWS})
    o = olist[[
        "window_days", "eligible_customer_count", "repeat_customer_count",
        "repeat_rate_pct",
    ]].rename(columns={
        "eligible_customer_count": "olist_eligible_customers",
        "repeat_customer_count": "olist_repeat_customers",
        "repeat_rate_pct": "olist_retention_rate_pct",
    })
    h = hm[[
        "window_days", "eligible_customer_count",
        "repeat_customer_count_excluding_zero_day",
        "repeat_rate_excluding_zero_day_pct",
    ]].rename(columns={
        "eligible_customer_count": "hm_eligible_customers",
        "repeat_customer_count_excluding_zero_day": "hm_repeat_customers",
        "repeat_rate_excluding_zero_day_pct": "hm_retention_rate_pct",
    })
    r = rees46[[
        "window_days", "eligible_customer_count", "repeat_customer_count",
        "retention_rate_pct",
    ]].rename(columns={
        "eligible_customer_count": "rees46_eligible_customers",
        "repeat_customer_count": "rees46_repeat_customers",
        "retention_rate_pct": "rees46_retention_rate_pct",
    })
    result = base.merge(o, on="window_days", how="left")
    result = result.merge(h, on="window_days", how="left")
    result = result.merge(r, on="window_days", how="left")
    result["hm_gap_vs_olist_pct_point"] = (
        result["hm_retention_rate_pct"] - result["olist_retention_rate_pct"]
    ).round(6)
    result["rees46_gap_vs_olist_pct_point"] = (
        result["rees46_retention_rate_pct"] - result["olist_retention_rate_pct"]
    ).round(6)
    result["hm_comparison_role"] = "동일 공식·보수적 외부 비교"
    result["rees46_comparison_role"] = result["window_days"].map({
        30: "동일 공식·구조 차이를 동반한 주 비교",
        60: "분모 대표성 부족으로 제한적 참고",
        90: "관찰기간 부족으로 비교 불가",
        180: "관찰기간 부족으로 비교 불가",
        365: "관찰기간 부족으로 비교 불가",
    })
    result["causal_interpretation_allowed"] = False
    return result


def build_methodology() -> pd.DataFrame:
    """데이터셋별 구매 단위와 비교 제한을 명시한다."""
    return pd.DataFrame([
        {
            "dataset": "Olist",
            "customer_key": "customer_unique_id",
            "purchase_event_definition": "배송 완료 order_id 1개",
            "primary_rate_rule": "N일 내 두 번째 배송 완료 주문",
            "observation_scope": "2016-09~2018-08 분석 관찰 범위",
            "comparison_limit": "마켓플레이스·다중 카테고리이며 데이터 시작 전 구매 이력은 알 수 없음",
        },
        {
            "dataset": "H&M",
            "customer_key": "customer_id",
            "purchase_event_definition": "고객·날짜·판매채널 조합 1개",
            "primary_rate_rule": "0일 재구매를 제외한 N일 내 두 번째 구매 이벤트",
            "observation_scope": "2018-09~2020-09",
            "comparison_limit": "패션 단일기업이며 실제 주문번호가 없어 구매 이벤트가 대체키에 의존",
        },
        {
            "dataset": "REES46",
            "customer_key": "user_id",
            "purchase_event_definition": "고객·세션 조합 1개",
            "primary_rate_rule": "N일 내 두 번째 관찰 구매 이벤트",
            "observation_scope": "2019-10~2019-11",
            "comparison_limit": "두 달 관찰·세션 대체키이며 첫 구매는 생애 최초 구매가 아님",
        },
    ])


def validate(
    comparison: pd.DataFrame, olist: pd.DataFrame, hm: pd.DataFrame,
    rees46: pd.DataFrame,
) -> pd.DataFrame:
    """원천값 재현과 결측 규칙을 검증한다."""
    checks = {
        "comparison_rows": (len(comparison), 5),
        "window_key_duplicates": (int(comparison["window_days"].duplicated().sum()), 0),
        "olist_30d_numerator": (
            int(comparison.loc[comparison.window_days.eq(30), "olist_repeat_customers"].iloc[0]),
            1373,
        ),
        "olist_30d_denominator": (
            int(comparison.loc[comparison.window_days.eq(30), "olist_eligible_customers"].iloc[0]),
            86772,
        ),
        "hm_30d_primary_numerator": (
            int(comparison.loc[comparison.window_days.eq(30), "hm_repeat_customers"].iloc[0]),
            372888,
        ),
        "rees46_30d_numerator": (
            int(comparison.loc[comparison.window_days.eq(30), "rees46_repeat_customers"].iloc[0]),
            139956,
        ),
        "rees46_available_windows": (
            int(comparison["rees46_retention_rate_pct"].notna().sum()), 2,
        ),
    }
    # 원천 표의 비율이 분자/분모로 재현되는지 별도로 확인한다.
    for name, frame, numerator, denominator, rate in [
        ("Olist", olist, "repeat_customer_count", "eligible_customer_count", "repeat_rate_pct"),
        ("H&M", hm, "repeat_customer_count_excluding_zero_day", "eligible_customer_count", "repeat_rate_excluding_zero_day_pct"),
        ("REES46", rees46, "repeat_customer_count", "eligible_customer_count", "retention_rate_pct"),
    ]:
        recalculated = frame[numerator] / frame[denominator] * 100
        max_difference = float((recalculated - frame[rate]).abs().max())
        checks[f"{name.lower()}_source_rate_max_rounding_diff_lt_0_0001"] = (
            max_difference < 0.0001, True
        )
    failures = {key: value for key, value in checks.items() if value[0] != value[1]}
    if failures:
        raise AssertionError(f"비교표 검증 실패: {failures}")
    return pd.DataFrame([
        {"check_name": key, "actual": actual, "expected": expected, "status": "PASS"}
        for key, (actual, expected) in checks.items()
    ])


def save(frame: pd.DataFrame, path: Path) -> None:
    """CSV를 임시 파일 경유로 저장한다."""
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    print(f"생성: {path.relative_to(ROOT)} ({len(frame):,}행)")


def main() -> None:
    olist, hm, rees46 = load_inputs()
    comparison = build_comparison(olist, hm, rees46)
    methodology = build_methodology()
    checks = validate(comparison, olist, hm, rees46)
    save(comparison, COMPARISON_PATH)
    save(methodology, METHODOLOGY_PATH)
    save(checks, CHECK_PATH)


if __name__ == "__main__":
    main()

"""REES46 30·60일 재구매율과 품질 플래그 민감도를 계산한다."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_MART_PATH = (
    PROJECT_ROOT / "data" / "external" / "rees46" / "processed"
    / "rees46_purchase_events.csv"
)
CUSTOMER_RETENTION_PATH = (
    PROJECT_ROOT / "data" / "external" / "rees46" / "processed"
    / "rees46_customer_retention_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
WINDOW_SUMMARY_PATH = OUTPUT_DIR / "rees46_retention_window_summary.csv"
SENSITIVITY_SUMMARY_PATH = OUTPUT_DIR / "rees46_retention_sensitivity_summary.csv"
BUILD_SUMMARY_PATH = OUTPUT_DIR / "rees46_retention_metrics_build_summary.csv"

OBSERVATION_END = pd.Timestamp("2019-11-30 23:59:44", tz="UTC")
WINDOW_DAYS = (30, 60)
EXPECTED_EVENTS = 1_402_758
EXPECTED_CUSTOMERS = 697_470
EXPECTED_REPEAT_CUSTOMERS = 256_959
EXPECTED_ELIGIBLE = {30: 347_118, 60: 14_064}

SCENARIOS = {
    "base_all_events": "모든 구매 이벤트 포함",
    "exclude_over_24h": "24시간 초과 세션 제외",
    "exclude_cross_month": "월 파일 경계 세션 제외",
    "exclude_either_flag": "24시간 초과 또는 월 경계 세션 제외",
}


def check_input() -> None:
    """구매 이벤트 마트와 필수 컬럼을 확인한다."""
    if not EVENT_MART_PATH.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {EVENT_MART_PATH}")
    columns = pd.read_csv(EVENT_MART_PATH, nrows=0).columns.tolist()
    required = [
        "purchase_event_key", "user_id", "purchase_event_start",
        "cross_month_file_flag", "session_over_24h_flag",
    ]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"구매 이벤트 마트 필수 컬럼 누락: {missing}")


def read_events() -> pd.DataFrame:
    """재구매율 계산에 필요한 구매 이벤트 컬럼만 읽는다."""
    events = pd.read_csv(
        EVENT_MART_PATH,
        usecols=[
            "purchase_event_key", "user_id", "purchase_event_start",
            "cross_month_file_flag", "session_over_24h_flag",
        ],
        dtype={"purchase_event_key": "string", "user_id": "string"},
    )
    events["purchase_event_start"] = pd.to_datetime(
        events["purchase_event_start"], errors="coerce", utc=True
    )
    if len(events) != EXPECTED_EVENTS:
        raise AssertionError(f"구매 이벤트 수 불일치: {len(events):,}")
    if int(events[[
        "purchase_event_key", "user_id", "purchase_event_start",
    ]].isna().any(axis=1).sum()):
        raise AssertionError("재구매율 필수값 결측 구매 이벤트가 있습니다.")
    if int(events["purchase_event_key"].duplicated().sum()):
        raise AssertionError("구매 이벤트 키 중복이 있습니다.")
    return events


def scenario_mask(events: pd.DataFrame, scenario: str) -> pd.Series:
    """민감도 정의별 포함 이벤트 마스크를 반환한다."""
    if scenario == "base_all_events":
        return pd.Series(True, index=events.index)
    if scenario == "exclude_over_24h":
        return ~events["session_over_24h_flag"]
    if scenario == "exclude_cross_month":
        return ~events["cross_month_file_flag"]
    if scenario == "exclude_either_flag":
        return ~(
            events["session_over_24h_flag"] | events["cross_month_file_flag"]
        )
    raise ValueError(f"알 수 없는 민감도 정의: {scenario}")


def build_customer_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """포함된 이벤트로 고객별 첫·두 번째 구매와 기간 플래그를 만든다."""
    ordered = events.sort_values(
        ["user_id", "purchase_event_start", "purchase_event_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["purchase_sequence"] = (
        ordered.groupby("user_id", sort=False, observed=True).cumcount() + 1
    )
    counts = (
        ordered.groupby("user_id", sort=False, observed=True).size()
        .rename("purchase_event_count")
    )
    first = ordered.loc[ordered["purchase_sequence"].eq(1), [
        "user_id", "purchase_event_key", "purchase_event_start",
    ]].rename(columns={
        "purchase_event_key": "first_purchase_event_key",
        "purchase_event_start": "first_purchase_time",
    }).set_index("user_id")
    second = ordered.loc[ordered["purchase_sequence"].eq(2), [
        "user_id", "purchase_event_key", "purchase_event_start",
    ]].rename(columns={
        "purchase_event_key": "second_purchase_event_key",
        "purchase_event_start": "second_purchase_time",
    }).set_index("user_id")

    customer = first.join(counts).join(second).reset_index()
    customer["seconds_to_second_purchase"] = (
        customer["second_purchase_time"] - customer["first_purchase_time"]
    ).dt.total_seconds()
    customer["days_to_second_purchase"] = (
        customer["seconds_to_second_purchase"] / 86_400
    )
    customer["observation_end_time"] = OBSERVATION_END
    customer["observation_days_available"] = (
        OBSERVATION_END - customer["first_purchase_time"]
    ).dt.total_seconds() / 86_400

    for days in WINDOW_DAYS:
        eligible = customer["observation_days_available"].ge(days)
        customer[f"eligible_{days}d_flag"] = eligible
        repeat_flag = pd.Series(pd.NA, index=customer.index, dtype="boolean")
        repeat_flag.loc[eligible] = (
            customer.loc[eligible, "second_purchase_time"].notna()
            & customer.loc[eligible, "seconds_to_second_purchase"].le(days * 86_400)
        )
        customer[f"repeat_within_{days}d_flag"] = repeat_flag
    return customer


def validate_scenario(
    scenario: str,
    included_events: pd.DataFrame,
    customer: pd.DataFrame,
) -> None:
    """민감도 정의별 고객 Grain과 기간 플래그를 검증한다."""
    if int(customer["user_id"].duplicated().sum()):
        raise AssertionError(f"{scenario}: 고객 중복이 있습니다.")
    if int(customer["purchase_event_count"].sum()) != len(included_events):
        raise AssertionError(f"{scenario}: 고객별 구매 수 합계가 이벤트 수와 다릅니다.")
    if int(customer["seconds_to_second_purchase"].lt(0).sum()):
        raise AssertionError(f"{scenario}: 음수 두 번째 구매 간격이 있습니다.")

    for days in WINDOW_DAYS:
        eligible = customer[f"eligible_{days}d_flag"]
        repeat = customer[f"repeat_within_{days}d_flag"]
        if int(repeat.isna().sum()) != int((~eligible).sum()):
            raise AssertionError(f"{scenario}: {days}일 결측과 부적격 고객 수가 다릅니다.")
        numerator = int(repeat.sum(skipna=True))
        denominator = int(eligible.sum())
        if numerator > denominator:
            raise AssertionError(f"{scenario}: {days}일 분자가 분모보다 큽니다.")

    if scenario == "base_all_events":
        checks = {
            "events": (len(included_events), EXPECTED_EVENTS),
            "customers": (len(customer), EXPECTED_CUSTOMERS),
            "repeat_customers": (
                int(customer["purchase_event_count"].ge(2).sum()),
                EXPECTED_REPEAT_CUSTOMERS,
            ),
        }
        for days, expected in EXPECTED_ELIGIBLE.items():
            checks[f"eligible_{days}d"] = (
                int(customer[f"eligible_{days}d_flag"].sum()), expected
            )
        failures = {key: values for key, values in checks.items() if values[0] != values[1]}
        if failures:
            raise AssertionError(f"기존 감사 결과와 불일치: {failures}")


def summarize_scenario(
    scenario: str,
    included_events: pd.DataFrame,
    customer: pd.DataFrame,
    base_event_count: int,
) -> list[dict]:
    """한 민감도 정의의 기간별 분자·분모·비율을 요약한다."""
    rows = []
    for days in WINDOW_DAYS:
        eligible = customer[f"eligible_{days}d_flag"]
        repeat = customer[f"repeat_within_{days}d_flag"]
        denominator = int(eligible.sum())
        numerator = int(repeat.sum(skipna=True))
        rate = numerator / denominator * 100 if denominator else pd.NA
        rows.append({
            "scenario": scenario,
            "scenario_description": SCENARIOS[scenario],
            "window_days": days,
            "included_purchase_events": len(included_events),
            "excluded_purchase_events": base_event_count - len(included_events),
            "included_customers": len(customer),
            "eligible_customer_count": denominator,
            "repeat_customer_count": numerator,
            "retention_rate_pct": round(rate, 6) if rate is not pd.NA else pd.NA,
            "ineligible_right_censored_count": len(customer) - denominator,
        })
    return rows


def create_build_summary(
    events: pd.DataFrame,
    base_customer: pd.DataFrame,
) -> pd.DataFrame:
    """기본 고객 지표 생성 품질을 요약한다."""
    metrics = {
        "source_purchase_events": len(events),
        "customer_metric_rows": len(base_customer),
        "unique_customers": int(base_customer["user_id"].nunique()),
        "duplicate_customer_rows": int(base_customer["user_id"].duplicated().sum()),
        "purchase_event_count_sum": int(base_customer["purchase_event_count"].sum()),
        "repeat_purchase_customers_all_period": int(
            base_customer["purchase_event_count"].ge(2).sum()
        ),
        "negative_second_purchase_gap": int(
            base_customer["seconds_to_second_purchase"].lt(0).sum()
        ),
        "eligible_30d_customers": int(base_customer["eligible_30d_flag"].sum()),
        "eligible_60d_customers": int(base_customer["eligible_60d_flag"].sum()),
        "eligible_90d_customers_prior_audit": 0,
        "observation_end_time": str(OBSERVATION_END),
    }
    return pd.DataFrame([
        {"check_name": key, "value": value, "status": "PASS"}
        for key, value in metrics.items()
    ])


def save_outputs(
    base_customer: pd.DataFrame,
    window_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    build_summary: pd.DataFrame,
) -> None:
    """검증된 고객 지표와 요약표를 원자적으로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (CUSTOMER_RETENTION_PATH, base_customer),
        (WINDOW_SUMMARY_PATH, window_summary),
        (SENSITIVITY_SUMMARY_PATH, sensitivity_summary),
        (BUILD_SUMMARY_PATH, build_summary),
    ]
    for target, frame in outputs:
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        print(f"생성: {target.relative_to(PROJECT_ROOT)} ({len(frame):,}행)")


def main() -> None:
    """기본 및 품질 플래그 민감도 재구매율을 계산한다."""
    check_input()
    events = read_events()
    scenario_rows = []
    customers = {}

    for scenario in SCENARIOS:
        mask = scenario_mask(events, scenario)
        included = events.loc[mask].copy()
        customer = build_customer_metrics(included)
        validate_scenario(scenario, included, customer)
        customers[scenario] = customer
        scenario_rows.extend(
            summarize_scenario(scenario, included, customer, len(events))
        )

    sensitivity = pd.DataFrame(scenario_rows)
    base_rates = (
        sensitivity.loc[sensitivity["scenario"].eq("base_all_events")]
        .set_index("window_days")["retention_rate_pct"]
    )
    sensitivity["difference_from_base_pct_point"] = sensitivity.apply(
        lambda row: round(
            row["retention_rate_pct"] - base_rates.loc[row["window_days"]], 6
        ),
        axis=1,
    )
    window_summary = sensitivity.loc[
        sensitivity["scenario"].eq("base_all_events"),
        [
            "window_days", "eligible_customer_count", "repeat_customer_count",
            "retention_rate_pct", "ineligible_right_censored_count",
        ],
    ].reset_index(drop=True)
    window_summary["interpretation_role"] = [
        "주 외부 비교 지표", "제한적 참고·민감도 지표",
    ]
    build_summary = create_build_summary(events, customers["base_all_events"])
    save_outputs(
        customers["base_all_events"], window_summary, sensitivity, build_summary
    )
    print("REES46 기간별 재구매율과 민감도 계산을 완료했습니다.")


if __name__ == "__main__":
    main()

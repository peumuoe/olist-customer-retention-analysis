"""REES46 고객별 구매 순서와 기간별 관찰 가능성을 감사한다."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MART_PATH = (
    PROJECT_ROOT
    / "data" / "external" / "rees46" / "processed"
    / "rees46_purchase_events.csv"
)
CUSTOMER_METRICS_PATH = (
    PROJECT_ROOT
    / "data" / "external" / "rees46" / "processed"
    / "rees46_customer_purchase_audit_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
AUDIT_SUMMARY_PATH = OUTPUT_DIR / "rees46_customer_purchase_order_audit_summary.csv"
COUNT_DISTRIBUTION_PATH = OUTPUT_DIR / "rees46_customer_purchase_count_distribution.csv"
ELIGIBILITY_SUMMARY_PATH = OUTPUT_DIR / "rees46_observation_window_eligibility_summary.csv"
GAP_DISTRIBUTION_PATH = OUTPUT_DIR / "rees46_second_purchase_gap_distribution.csv"

EXPECTED_PURCHASE_EVENTS = 1_402_758
EXPECTED_PURCHASE_USERS = 697_470
EXPECTED_OBSERVATION_END = pd.Timestamp("2019-11-30 23:59:44", tz="UTC")
WINDOW_DAYS = (30, 60, 90)


def check_input() -> None:
    """구매 이벤트 마트와 필수 컬럼을 확인한다."""
    if not MART_PATH.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {MART_PATH}")
    columns = pd.read_csv(MART_PATH, nrows=0).columns.tolist()
    required = [
        "purchase_event_key", "user_id", "purchase_event_start",
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag",
    ]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"구매 이벤트 마트 필수 컬럼 누락: {missing}")


def read_and_prepare_events() -> pd.DataFrame:
    """구매 이벤트를 읽고 고객별 시각 순서로 정렬한다."""
    usecols = [
        "purchase_event_key", "user_id", "purchase_event_start",
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag",
    ]
    events = pd.read_csv(
        MART_PATH,
        usecols=usecols,
        dtype={"purchase_event_key": "string", "user_id": "string"},
    )
    events["purchase_event_start"] = pd.to_datetime(
        events["purchase_event_start"], errors="coerce", utc=True
    )

    if len(events) != EXPECTED_PURCHASE_EVENTS:
        raise AssertionError(f"구매 이벤트 수 불일치: {len(events):,}")
    required_nulls = int(
        events[["purchase_event_key", "user_id", "purchase_event_start"]]
        .isna().any(axis=1).sum()
    )
    if required_nulls:
        raise AssertionError(f"필수값 결측 구매 이벤트: {required_nulls:,}")
    if int(events["purchase_event_key"].duplicated().sum()):
        raise AssertionError("구매 이벤트 키 중복이 발견됐습니다.")

    events = events.sort_values(
        ["user_id", "purchase_event_start", "purchase_event_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    events["purchase_sequence"] = (
        events.groupby("user_id", sort=False, observed=True).cumcount() + 1
    )
    return events


def build_customer_metrics(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """고객당 1행으로 첫·두 번째 구매와 관찰 가능 기간을 만든다."""
    observation_end = events["purchase_event_start"].max()
    grouped = events.groupby("user_id", sort=False, observed=True)

    counts = grouped.size().rename("purchase_event_count")
    last_time = grouped["purchase_event_start"].max().rename("last_purchase_time")
    any_flags = grouped[[
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag",
    ]].max().rename(columns={
        "cross_month_file_flag": "has_cross_month_event",
        "session_over_30m_flag": "has_over_30m_event",
        "session_over_24h_flag": "has_over_24h_event",
    })

    first = events.loc[events["purchase_sequence"].eq(1), [
        "user_id", "purchase_event_key", "purchase_event_start",
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag",
    ]].rename(columns={
        "purchase_event_key": "first_purchase_event_key",
        "purchase_event_start": "first_purchase_time",
        "cross_month_file_flag": "first_purchase_cross_month_flag",
        "session_over_30m_flag": "first_purchase_over_30m_flag",
        "session_over_24h_flag": "first_purchase_over_24h_flag",
    }).set_index("user_id")

    second = events.loc[events["purchase_sequence"].eq(2), [
        "user_id", "purchase_event_key", "purchase_event_start",
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag",
    ]].rename(columns={
        "purchase_event_key": "second_purchase_event_key",
        "purchase_event_start": "second_purchase_time",
        "cross_month_file_flag": "second_purchase_cross_month_flag",
        "session_over_30m_flag": "second_purchase_over_30m_flag",
        "session_over_24h_flag": "second_purchase_over_24h_flag",
    }).set_index("user_id")

    customer = (
        first.join(counts).join(second).join(last_time).join(any_flags).reset_index()
    )
    customer["seconds_to_second_purchase"] = (
        customer["second_purchase_time"] - customer["first_purchase_time"]
    ).dt.total_seconds()
    customer["days_to_second_purchase"] = (
        customer["seconds_to_second_purchase"] / 86_400
    )
    customer["same_timestamp_second_purchase_flag"] = (
        customer["seconds_to_second_purchase"].eq(0)
    )
    customer["observation_end_time"] = observation_end
    customer["observation_days_available"] = (
        observation_end - customer["first_purchase_time"]
    ).dt.total_seconds() / 86_400

    for days in WINDOW_DAYS:
        customer[f"eligible_{days}d_flag"] = (
            customer["observation_days_available"].ge(days)
        )
    return customer, observation_end


def validate_customer_metrics(
    customer: pd.DataFrame,
    observation_end: pd.Timestamp,
) -> dict[str, int | str]:
    """고객 Grain, 구매 순서, 관찰기간의 불변조건을 검증한다."""
    single_mask = customer["purchase_event_count"].eq(1)
    repeat_mask = customer["purchase_event_count"].ge(2)
    metrics: dict[str, int | str] = {
        "customer_rows": len(customer),
        "unique_users": int(customer["user_id"].nunique()),
        "duplicate_user_rows": int(customer["user_id"].duplicated(keep=False).sum()),
        "purchase_event_count_sum": int(customer["purchase_event_count"].sum()),
        "single_purchase_customers": int(single_mask.sum()),
        "repeat_purchase_customers": int(repeat_mask.sum()),
        "single_customer_with_second_purchase": int(
            customer.loc[single_mask, "second_purchase_time"].notna().sum()
        ),
        "repeat_customer_without_second_purchase": int(
            customer.loc[repeat_mask, "second_purchase_time"].isna().sum()
        ),
        "negative_second_purchase_gap": int(
            customer["seconds_to_second_purchase"].lt(0).sum()
        ),
        "same_timestamp_second_purchase_customers": int(
            customer["same_timestamp_second_purchase_flag"].sum()
        ),
        "first_after_last_purchase_rows": int(
            customer["first_purchase_time"].gt(customer["last_purchase_time"]).sum()
        ),
        "negative_observation_days_rows": int(
            customer["observation_days_available"].lt(0).sum()
        ),
        "customers_with_cross_month_event": int(customer["has_cross_month_event"].sum()),
        "customers_with_over_24h_event": int(customer["has_over_24h_event"].sum()),
        "observation_end_time": str(observation_end),
    }

    expected = {
        "customer_rows": EXPECTED_PURCHASE_USERS,
        "unique_users": EXPECTED_PURCHASE_USERS,
        "duplicate_user_rows": 0,
        "purchase_event_count_sum": EXPECTED_PURCHASE_EVENTS,
        "single_customer_with_second_purchase": 0,
        "repeat_customer_without_second_purchase": 0,
        "negative_second_purchase_gap": 0,
        "first_after_last_purchase_rows": 0,
        "negative_observation_days_rows": 0,
        "observation_end_time": str(EXPECTED_OBSERVATION_END),
    }
    failures = {
        key: (metrics[key], value)
        for key, value in expected.items()
        if metrics[key] != value
    }
    if failures:
        raise AssertionError(f"고객 구매 순서 감사 실패: {failures}")

    eligibility = [int(customer[f"eligible_{days}d_flag"].sum()) for days in WINDOW_DAYS]
    if eligibility != sorted(eligibility, reverse=True):
        raise AssertionError(f"관찰기간 증가에 따른 적격 고객 수가 감소하지 않습니다: {eligibility}")
    return metrics


def create_summaries(
    customer: pd.DataFrame,
    metrics: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """감사, 구매횟수, 적격 고객, 두 번째 구매 간격 요약표를 만든다."""
    audit_summary = pd.DataFrame([
        {"check_name": key, "value": value, "status": "PASS"}
        for key, value in metrics.items()
    ])

    count_distribution = (
        customer["purchase_event_count"].value_counts().sort_index()
        .rename_axis("purchase_event_count").reset_index(name="customer_count")
    )
    count_distribution["customer_rate_pct"] = (
        count_distribution["customer_count"] / len(customer) * 100
    ).round(6)

    eligibility_rows = []
    for days in WINDOW_DAYS:
        eligible_count = int(customer[f"eligible_{days}d_flag"].sum())
        eligibility_rows.append({
            "window_days": days,
            "eligible_customer_count": eligible_count,
            "eligible_customer_rate_pct": round(eligible_count / len(customer) * 100, 6),
            "ineligible_right_censored_count": len(customer) - eligible_count,
            "denominator_rule": f"첫 관찰 구매 후 데이터 종료까지 {days}일 이상 관찰 가능",
        })
    eligibility_summary = pd.DataFrame(eligibility_rows)

    gap = customer["seconds_to_second_purchase"]
    gap_rows = [
        ("두 번째 관찰 구매 없음", int(gap.isna().sum())),
        ("동일 시각", int(gap.eq(0).sum())),
        ("0초 초과~1일", int(gap.gt(0).mul(gap.le(86_400)).sum())),
        ("1일 초과~7일", int(gap.gt(86_400).mul(gap.le(7 * 86_400)).sum())),
        ("7일 초과~30일", int(gap.gt(7 * 86_400).mul(gap.le(30 * 86_400)).sum())),
        ("30일 초과~60일", int(gap.gt(30 * 86_400).mul(gap.le(60 * 86_400)).sum())),
        ("60일 초과", int(gap.gt(60 * 86_400).sum())),
    ]
    gap_distribution = pd.DataFrame(gap_rows, columns=["gap_band", "customer_count"])
    gap_distribution["customer_rate_pct"] = (
        gap_distribution["customer_count"] / len(customer) * 100
    ).round(6)
    return audit_summary, count_distribution, eligibility_summary, gap_distribution


def save_outputs(
    customer: pd.DataFrame,
    audit_summary: pd.DataFrame,
    count_distribution: pd.DataFrame,
    eligibility_summary: pd.DataFrame,
    gap_distribution: pd.DataFrame,
) -> None:
    """검증된 고객 마트와 요약표를 원자적으로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (CUSTOMER_METRICS_PATH, customer),
        (AUDIT_SUMMARY_PATH, audit_summary),
        (COUNT_DISTRIBUTION_PATH, count_distribution),
        (ELIGIBILITY_SUMMARY_PATH, eligibility_summary),
        (GAP_DISTRIBUTION_PATH, gap_distribution),
    ]
    for target, frame in outputs:
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        print(f"생성: {target.relative_to(PROJECT_ROOT)} ({len(frame):,}행)")


def main() -> None:
    """고객별 구매 순서와 관찰기간을 감사한다."""
    check_input()
    events = read_and_prepare_events()
    customer, observation_end = build_customer_metrics(events)
    metrics = validate_customer_metrics(customer, observation_end)
    summaries = create_summaries(customer, metrics)
    save_outputs(customer, *summaries)
    print("REES46 고객별 구매 순서와 관찰기간 감사를 완료했습니다.")


if __name__ == "__main__":
    main()

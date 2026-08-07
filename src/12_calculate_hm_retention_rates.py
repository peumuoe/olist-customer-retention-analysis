from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HM_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "HM"
    / "processed"
)

OUTPUT_TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

HM_PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PURCHASE_EVENT_MART_PATH = (
    HM_PROCESSED_DIR
    / "hm_purchase_events.csv"
)

CUSTOMER_METRICS_PATH = (
    HM_PROCESSED_DIR
    / "hm_customer_retention_metrics.csv"
)

RETENTION_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_retention_window_summary.csv"
)

BUILD_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_retention_metrics_build_summary.csv"
)

PRIOR_MART_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_purchase_event_mart_summary.csv"
)

CHUNK_SIZE = 500_000

RETENTION_WINDOWS = [
    30,
    60,
    90,
    180,
    365,
]

REQUIRED_COLUMNS = [
    "customer_id",
    "purchase_date",
    "sales_channel_id",
    "customer_purchase_event_number",
]


# ============================================================
# 2. 공통 함수
# ============================================================

def check_file_exists(
    file_path: Path,
) -> None:
    """필수 파일이 존재하는지 확인합니다."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )


def normalize_string(
    series: pd.Series,
) -> pd.Series:
    """문자열 ID의 좌우 공백을 제거합니다."""

    return (
        series
        .astype("string")
        .str.strip()
    )


def read_prior_summary_value(
    check_name: str,
) -> int | None:
    """
    이전 H&M 구매 이벤트 마트 요약에서
    특정 검증값을 읽습니다.
    """

    if not PRIOR_MART_SUMMARY_PATH.exists():
        return None

    summary = pd.read_csv(
        PRIOR_MART_SUMMARY_PATH,
        low_memory=False,
    )

    matched = summary.loc[
        summary["check_name"].eq(
            check_name
        ),
        "value",
    ]

    if matched.empty:
        return None

    return int(
        float(matched.iloc[0])
    )


# ============================================================
# 3. 고객별 첫 구매·두 번째 구매 수집
# ============================================================

def collect_customer_purchase_metrics() -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """
    H&M 구매 이벤트 마트를 청크 단위로 읽어
    고객별 첫 구매일, 두 번째 구매일,
    전체 구매 이벤트 수를 수집합니다.
    """

    check_file_exists(
        PURCHASE_EVENT_MART_PATH
    )

    first_purchase_dates: dict[
        str,
        pd.Timestamp,
    ] = {}

    second_purchase_dates: dict[
        str,
        pd.Timestamp,
    ] = {}

    customer_event_counts: dict[
        str,
        int,
    ] = {}

    total_event_rows = 0
    chunk_number = 0

    date_parse_failure_count = 0
    event_number_parse_failure_count = 0

    null_customer_id_rows = 0
    null_purchase_date_rows = 0
    null_sales_channel_rows = 0
    null_event_number_rows = 0

    duplicate_first_event_rows = 0
    duplicate_second_event_rows = 0

    minimum_purchase_date = None
    maximum_purchase_date = None

    previous_chunk_max_date = None
    dates_sorted_ascending = True

    reader = pd.read_csv(
        PURCHASE_EVENT_MART_PATH,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={
            "customer_id": "string",
            "purchase_date": "string",
            "sales_channel_id": "string",
        },
        low_memory=False,
    )

    for chunk in reader:
        chunk_number += 1
        total_event_rows += len(chunk)

        chunk["customer_id"] = normalize_string(
            chunk["customer_id"]
        )

        chunk["sales_channel_id"] = normalize_string(
            chunk["sales_channel_id"]
        )

        null_customer_id_rows += int(
            chunk["customer_id"]
            .isna()
            .sum()
        )

        null_purchase_date_rows += int(
            chunk["purchase_date"]
            .isna()
            .sum()
        )

        null_sales_channel_rows += int(
            chunk["sales_channel_id"]
            .isna()
            .sum()
        )

        # ----------------------------------------------------
        # 구매일 변환
        # ----------------------------------------------------

        parsed_dates = pd.to_datetime(
            chunk["purchase_date"],
            errors="coerce",
        )

        date_failure_mask = (
            chunk["purchase_date"].notna()
            & parsed_dates.isna()
        )

        date_parse_failure_count += int(
            date_failure_mask.sum()
        )

        if date_failure_mask.any():
            raise ValueError(
                "purchase_date 변환에 실패한 행이 있습니다."
            )

        chunk["purchase_date"] = parsed_dates

        # ----------------------------------------------------
        # 고객별 구매 이벤트 순서 변환
        # ----------------------------------------------------

        parsed_event_numbers = pd.to_numeric(
            chunk[
                "customer_purchase_event_number"
            ],
            errors="coerce",
        )

        event_number_failure_mask = (
            chunk[
                "customer_purchase_event_number"
            ].notna()
            & parsed_event_numbers.isna()
        )

        event_number_parse_failure_count += int(
            event_number_failure_mask.sum()
        )

        if event_number_failure_mask.any():
            raise ValueError(
                "customer_purchase_event_number "
                "변환에 실패한 행이 있습니다."
            )

        null_event_number_rows += int(
            parsed_event_numbers.isna().sum()
        )

        chunk[
            "customer_purchase_event_number"
        ] = (
            parsed_event_numbers
            .astype("Int64")
        )

        # ----------------------------------------------------
        # 날짜 정렬 검증
        # ----------------------------------------------------

        if not chunk[
            "purchase_date"
        ].is_monotonic_increasing:
            dates_sorted_ascending = False

        chunk_min_date = (
            chunk["purchase_date"].min()
        )

        chunk_max_date = (
            chunk["purchase_date"].max()
        )

        if (
            previous_chunk_max_date is not None
            and chunk_min_date
            < previous_chunk_max_date
        ):
            dates_sorted_ascending = False

        if not dates_sorted_ascending:
            raise ValueError(
                "구매 이벤트 마트가 날짜 오름차순으로 "
                "정렬되어 있지 않습니다."
            )

        previous_chunk_max_date = (
            chunk_max_date
        )

        if minimum_purchase_date is None:
            minimum_purchase_date = (
                chunk_min_date
            )

        maximum_purchase_date = (
            chunk_max_date
        )

        # ----------------------------------------------------
        # 고객별 첫 구매일 수집
        # ----------------------------------------------------

        first_rows = chunk.loc[
            chunk[
                "customer_purchase_event_number"
            ].eq(1),
            [
                "customer_id",
                "purchase_date",
            ],
        ]

        for row in first_rows.itertuples(
            index=False
        ):
            customer_id = row.customer_id
            purchase_date = row.purchase_date

            if customer_id in first_purchase_dates:
                duplicate_first_event_rows += 1
            else:
                first_purchase_dates[
                    customer_id
                ] = purchase_date

        # ----------------------------------------------------
        # 고객별 두 번째 구매일 수집
        # ----------------------------------------------------

        second_rows = chunk.loc[
            chunk[
                "customer_purchase_event_number"
            ].eq(2),
            [
                "customer_id",
                "purchase_date",
            ],
        ]

        for row in second_rows.itertuples(
            index=False
        ):
            customer_id = row.customer_id
            purchase_date = row.purchase_date

            if customer_id in second_purchase_dates:
                duplicate_second_event_rows += 1
            else:
                second_purchase_dates[
                    customer_id
                ] = purchase_date

        # ----------------------------------------------------
        # 고객별 총 구매 이벤트 수
        # ----------------------------------------------------

        chunk_customer_max = (
            chunk.groupby(
                "customer_id",
                sort=False,
            )[
                "customer_purchase_event_number"
            ]
            .max()
        )

        for customer_id, event_count in (
            chunk_customer_max.items()
        ):
            current_count = (
                customer_event_counts.get(
                    customer_id,
                    0,
                )
            )

            customer_event_counts[
                customer_id
            ] = max(
                current_count,
                int(event_count),
            )

        print(
            f"청크 {chunk_number} 처리 완료 "
            f"- 누적 구매 이벤트: "
            f"{total_event_rows:,} "
            f"- 누적 고객 수: "
            f"{len(customer_event_counts):,}"
        )

    if total_event_rows == 0:
        raise ValueError(
            "구매 이벤트 마트에 데이터가 없습니다."
        )

    # ========================================================
    # 고객당 1행 데이터 생성
    # ========================================================

    customer_metrics = pd.DataFrame(
        {
            "customer_id": list(
                customer_event_counts.keys()
            ),
            "total_purchase_event_count": list(
                customer_event_counts.values()
            ),
        }
    )

    customer_metrics[
        "first_purchase_date"
    ] = (
        customer_metrics[
            "customer_id"
        ]
        .map(first_purchase_dates)
    )

    customer_metrics[
        "second_purchase_date"
    ] = (
        customer_metrics[
            "customer_id"
        ]
        .map(second_purchase_dates)
    )

    customer_metrics[
        "is_repeat_customer_all_time"
    ] = (
        customer_metrics[
            "total_purchase_event_count"
        ].ge(2)
    )

    customer_metrics[
        "days_to_second_purchase"
    ] = (
        customer_metrics[
            "second_purchase_date"
        ]
        - customer_metrics[
            "first_purchase_date"
        ]
    ).dt.days.astype("Int64")

    customer_metrics[
        "is_zero_day_repeat"
    ] = (
        customer_metrics[
            "days_to_second_purchase"
        ].eq(0)
    ).astype("boolean")

    negative_interval_rows = int(
        customer_metrics[
            "days_to_second_purchase"
        ]
        .lt(0)
        .fillna(False)
        .sum()
    )

    if negative_interval_rows > 0:
        raise ValueError(
            "두 번째 구매일이 첫 구매일보다 빠른 "
            f"고객이 {negative_interval_rows:,}명 있습니다."
        )

    audit_values = {
        "total_event_rows":
            total_event_rows,
        "minimum_purchase_date":
            minimum_purchase_date,
        "maximum_purchase_date":
            maximum_purchase_date,
        "date_parse_failure_count":
            date_parse_failure_count,
        "event_number_parse_failure_count":
            event_number_parse_failure_count,
        "null_customer_id_rows":
            null_customer_id_rows,
        "null_purchase_date_rows":
            null_purchase_date_rows,
        "null_sales_channel_rows":
            null_sales_channel_rows,
        "null_event_number_rows":
            null_event_number_rows,
        "duplicate_first_event_rows":
            duplicate_first_event_rows,
        "duplicate_second_event_rows":
            duplicate_second_event_rows,
        "negative_interval_rows":
            negative_interval_rows,
        "dates_sorted_ascending":
            dates_sorted_ascending,
    }

    return (
        customer_metrics,
        audit_values,
    )


# ============================================================
# 4. 기간별 재구매 여부 계산
# ============================================================

def add_retention_window_columns(
    customer_metrics: pd.DataFrame,
    observation_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    고객별 30·60·90·180·365일
    관찰 가능 여부와 재구매 여부를 계산합니다.
    """

    metrics = customer_metrics.copy()

    for window_days in RETENTION_WINDOWS:
        eligible_column = (
            f"eligible_{window_days}d"
        )

        repeat_column = (
            f"repeat_within_{window_days}d"
        )

        first_purchase_cutoff = (
            observation_cutoff
            - pd.Timedelta(
                days=window_days
            )
        )

        eligible_mask = (
            metrics[
                "first_purchase_date"
            ]
            <= first_purchase_cutoff
        )

        repeated_within_window = (
            metrics[
                "second_purchase_date"
            ].notna()
            & (
                metrics[
                    "second_purchase_date"
                ]
                <= (
                    metrics[
                        "first_purchase_date"
                    ]
                    + pd.Timedelta(
                        days=window_days
                    )
                )
            )
        )

        metrics[
            eligible_column
        ] = eligible_mask

        repeat_result = pd.Series(
            pd.NA,
            index=metrics.index,
            dtype="boolean",
        )

        repeat_result.loc[
            eligible_mask
        ] = (
            repeated_within_window.loc[
                eligible_mask
            ]
        )

        metrics[
            repeat_column
        ] = repeat_result

    return metrics


# ============================================================
# 5. 기간별 재구매율 요약
# ============================================================

def create_retention_summary(
    customer_metrics: pd.DataFrame,
    observation_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    H&M의 기간별 재구매율과
    0일 재구매 제외 민감도 결과를 생성합니다.
    """

    records = []

    for window_days in RETENTION_WINDOWS:
        eligible_column = (
            f"eligible_{window_days}d"
        )

        repeat_column = (
            f"repeat_within_{window_days}d"
        )

        eligible_mask = (
            customer_metrics[
                eligible_column
            ]
        )

        repeat_mask = (
            customer_metrics[
                repeat_column
            ]
            .fillna(False)
        )

        zero_day_mask = (
            customer_metrics[
                "is_zero_day_repeat"
            ]
            .fillna(False)
        )

        eligible_customer_count = int(
            eligible_mask.sum()
        )

        repeat_customer_count = int(
            (
                eligible_mask
                & repeat_mask
            ).sum()
        )

        zero_day_repeat_customer_count = int(
            (
                eligible_mask
                & repeat_mask
                & zero_day_mask
            ).sum()
        )

        repeat_customer_count_excluding_zero_day = (
            repeat_customer_count
            - zero_day_repeat_customer_count
        )

        non_repeat_customer_count = (
            eligible_customer_count
            - repeat_customer_count
        )

        repeat_rate_pct = (
            repeat_customer_count
            / eligible_customer_count
            * 100
            if eligible_customer_count > 0
            else 0.0
        )

        repeat_rate_excluding_zero_day_pct = (
            repeat_customer_count_excluding_zero_day
            / eligible_customer_count
            * 100
            if eligible_customer_count > 0
            else 0.0
        )

        zero_day_share_of_repeat_pct = (
            zero_day_repeat_customer_count
            / repeat_customer_count
            * 100
            if repeat_customer_count > 0
            else 0.0
        )

        first_purchase_cutoff = (
            observation_cutoff
            - pd.Timedelta(
                days=window_days
            )
        )

        records.append(
            {
                "window_days":
                    window_days,
                "observation_cutoff":
                    observation_cutoff,
                "eligible_first_purchase_cutoff":
                    first_purchase_cutoff,
                "eligible_customer_count":
                    eligible_customer_count,
                "repeat_customer_count":
                    repeat_customer_count,
                "zero_day_repeat_customer_count":
                    zero_day_repeat_customer_count,
                "repeat_customer_count_excluding_zero_day":
                    repeat_customer_count_excluding_zero_day,
                "non_repeat_customer_count":
                    non_repeat_customer_count,
                "repeat_rate_pct": round(
                    repeat_rate_pct,
                    4,
                ),
                "repeat_rate_excluding_zero_day_pct":
                    round(
                        repeat_rate_excluding_zero_day_pct,
                        4,
                    ),
                "zero_day_share_of_repeat_pct":
                    round(
                        zero_day_share_of_repeat_pct,
                        4,
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 6. 생성 품질 검증 요약
# ============================================================

def create_build_summary(
    customer_metrics: pd.DataFrame,
    audit_values: dict[str, object],
) -> pd.DataFrame:
    """H&M 고객 재구매 지표 생성 결과를 검증합니다."""

    prior_event_rows = (
        read_prior_summary_value(
            "purchase_event_mart_rows"
        )
    )

    prior_unique_customers = (
        read_prior_summary_value(
            "unique_customers"
        )
    )

    current_event_rows = int(
        audit_values[
            "total_event_rows"
        ]
    )

    current_unique_customers = len(
        customer_metrics
    )

    event_rows_match_prior = (
        prior_event_rows is None
        or prior_event_rows
        == current_event_rows
    )

    customer_count_matches_prior = (
        prior_unique_customers is None
        or prior_unique_customers
        == current_unique_customers
    )

    eligible_counts = [
        int(
            customer_metrics[
                f"eligible_{window_days}d"
            ].sum()
        )
        for window_days in RETENTION_WINDOWS
    ]

    eligible_counts_non_increasing = all(
        current_count >= next_count
        for current_count, next_count
        in zip(
            eligible_counts,
            eligible_counts[1:],
        )
    )

    if not eligible_counts_non_increasing:
        raise ValueError(
            "관찰기간이 길어질수록 적격 고객 수가 "
            "감소해야 하지만 조건을 만족하지 않습니다."
        )

    summary_records = [
        {
            "check_name":
                "purchase_event_mart_rows",
            "value":
                current_event_rows,
        },
        {
            "check_name":
                "customer_metric_rows",
            "value":
                len(customer_metrics),
        },
        {
            "check_name":
                "unique_customer_ids",
            "value":
                customer_metrics[
                    "customer_id"
                ].nunique(),
        },
        {
            "check_name":
                "duplicate_customer_id_rows",
            "value": int(
                customer_metrics[
                    "customer_id"
                ]
                .duplicated(keep=False)
                .sum()
            ),
        },
        {
            "check_name":
                "null_first_purchase_rows",
            "value": int(
                customer_metrics[
                    "first_purchase_date"
                ]
                .isna()
                .sum()
            ),
        },
        {
            "check_name":
                "null_second_purchase_rows",
            "value": int(
                customer_metrics[
                    "second_purchase_date"
                ]
                .isna()
                .sum()
            ),
        },
        {
            "check_name":
                "all_time_repeat_customers",
            "value": int(
                customer_metrics[
                    "is_repeat_customer_all_time"
                ].sum()
            ),
        },
        {
            "check_name":
                "zero_day_repeat_customers_all_time",
            "value": int(
                customer_metrics[
                    "is_zero_day_repeat"
                ]
                .fillna(False)
                .sum()
            ),
        },
        {
            "check_name":
                "maximum_customer_event_count",
            "value": int(
                customer_metrics[
                    "total_purchase_event_count"
                ].max()
            ),
        },
        {
            "check_name":
                "minimum_purchase_date",
            "value":
                audit_values[
                    "minimum_purchase_date"
                ],
        },
        {
            "check_name":
                "observation_cutoff",
            "value":
                audit_values[
                    "maximum_purchase_date"
                ],
        },
        {
            "check_name":
                "date_parse_failures",
            "value":
                audit_values[
                    "date_parse_failure_count"
                ],
        },
        {
            "check_name":
                "event_number_parse_failures",
            "value":
                audit_values[
                    "event_number_parse_failure_count"
                ],
        },
        {
            "check_name":
                "duplicate_first_event_rows",
            "value":
                audit_values[
                    "duplicate_first_event_rows"
                ],
        },
        {
            "check_name":
                "duplicate_second_event_rows",
            "value":
                audit_values[
                    "duplicate_second_event_rows"
                ],
        },
        {
            "check_name":
                "negative_second_purchase_intervals",
            "value":
                audit_values[
                    "negative_interval_rows"
                ],
        },
        {
            "check_name":
                "dates_sorted_ascending",
            "value":
                audit_values[
                    "dates_sorted_ascending"
                ],
        },
        {
            "check_name":
                "event_rows_match_prior_mart",
            "value":
                event_rows_match_prior,
        },
        {
            "check_name":
                "customer_count_matches_prior_mart",
            "value":
                customer_count_matches_prior,
        },
        {
            "check_name":
                "eligible_counts_non_increasing",
            "value":
                eligible_counts_non_increasing,
        },
    ]

    return pd.DataFrame(
        summary_records
    )


# ============================================================
# 7. 실행 및 저장
# ============================================================

def main() -> None:
    print(
        "\n[H&M 기간별 재구매율 계산 시작]"
    )

    (
        customer_metrics,
        audit_values,
    ) = collect_customer_purchase_metrics()

    observation_cutoff = (
        audit_values[
            "maximum_purchase_date"
        ]
    )

    customer_metrics = (
        add_retention_window_columns(
            customer_metrics=(
                customer_metrics
            ),
            observation_cutoff=(
                observation_cutoff
            ),
        )
    )

    retention_summary = (
        create_retention_summary(
            customer_metrics=(
                customer_metrics
            ),
            observation_cutoff=(
                observation_cutoff
            ),
        )
    )

    build_summary = (
        create_build_summary(
            customer_metrics=(
                customer_metrics
            ),
            audit_values=(
                audit_values
            ),
        )
    )

    customer_metrics.to_csv(
        CUSTOMER_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    retention_summary.to_csv(
        RETENTION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    build_summary.to_csv(
        BUILD_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print(
        "\n[H&M 기간별 재구매율]"
    )

    print(
        retention_summary.to_string(
            index=False
        )
    )

    print(
        "\n[H&M 고객 재구매 지표 생성 검증]"
    )

    print(
        build_summary.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {CUSTOMER_METRICS_PATH}")
    print(f"- {RETENTION_SUMMARY_PATH}")
    print(f"- {BUILD_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 및 분석 기간
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

RETENTION_WINDOWS = [30, 60, 90, 180, 365]


# ============================================================
# 2. 반복구매 주문 마트 로딩
# ============================================================

def load_retention_orders() -> pd.DataFrame:
    """반복구매 분석용 주문 단위 마트를 불러옵니다."""

    file_path = (
        PROCESSED_DATA_DIR / "retention_orders.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    retention_orders = pd.read_csv(
        file_path,
        parse_dates=[
            "order_purchase_timestamp",
            "customer_first_purchase_timestamp",
            "previous_purchase_timestamp",
        ],
        low_memory=False,
    )

    required_columns = [
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in retention_orders.columns
    ]

    if missing_columns:
        raise ValueError(
            "필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    return retention_orders


# ============================================================
# 3. 고객 단위 반복구매 데이터 생성
# ============================================================

def build_customer_retention_metrics(
    retention_orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    주문 데이터를 고객당 1행으로 집계하고
    기간별 반복구매 여부를 계산합니다.
    """

    orders = retention_orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_id",
        ]
    ).copy()

    # 고객별 구매 순서를 다시 계산해 검증
    orders["_customer_order_number"] = (
        orders
        .groupby("customer_unique_id")
        .cumcount()
        .add(1)
    )

    observation_cutoff = orders[
        "order_purchase_timestamp"
    ].max()

    customer_metrics = (
        orders
        .groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            total_order_count=(
                "order_id",
                "nunique",
            ),
            first_purchase_timestamp=(
                "order_purchase_timestamp",
                "first",
            ),
            last_purchase_timestamp=(
                "order_purchase_timestamp",
                "last",
            ),
        )
    )

    second_purchase = (
        orders.loc[
            orders["_customer_order_number"].eq(2),
            [
                "customer_unique_id",
                "order_purchase_timestamp",
            ],
        ]
        .rename(
            columns={
                "order_purchase_timestamp":
                    "second_purchase_timestamp"
            }
        )
    )

    customer_metrics = customer_metrics.merge(
        second_purchase,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    customer_metrics["is_repeat_customer_all_time"] = (
        customer_metrics["total_order_count"].ge(2)
    )

    customer_metrics["days_to_second_purchase"] = (
        (
            customer_metrics[
                "second_purchase_timestamp"
            ]
            - customer_metrics[
                "first_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        .div(86_400)
        .round(6)
    )

    for window_days in RETENTION_WINDOWS:
        eligibility_column = (
            f"eligible_{window_days}d"
        )

        repeat_column = (
            f"repeat_within_{window_days}d"
        )

        eligibility_cutoff = (
            observation_cutoff
            - pd.Timedelta(days=window_days)
        )

        eligible_mask = (
            customer_metrics[
                "first_purchase_timestamp"
            ]
            <= eligibility_cutoff
        )

        repeated_within_window_mask = (
            customer_metrics[
                "second_purchase_timestamp"
            ].notna()
            & (
                customer_metrics[
                    "second_purchase_timestamp"
                ]
                <= (
                    customer_metrics[
                        "first_purchase_timestamp"
                    ]
                    + pd.Timedelta(days=window_days)
                )
            )
        )

        customer_metrics[eligibility_column] = (
            eligible_mask
        )

        # 관찰기간이 부족한 고객은 False가 아니라 NA로 표시
        repeat_result = pd.Series(
            pd.NA,
            index=customer_metrics.index,
            dtype="boolean",
        )

        repeat_result.loc[eligible_mask] = (
            repeated_within_window_mask.loc[
                eligible_mask
            ]
        )

        customer_metrics[repeat_column] = (
            repeat_result
        )

    return customer_metrics, observation_cutoff


# ============================================================
# 4. 기간별 반복구매율 요약
# ============================================================

def create_retention_window_summary(
    customer_metrics: pd.DataFrame,
    observation_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """30·60·90·180·365일 반복구매율을 요약합니다."""

    records = []

    for window_days in RETENTION_WINDOWS:
        eligibility_column = (
            f"eligible_{window_days}d"
        )

        repeat_column = (
            f"repeat_within_{window_days}d"
        )

        eligible_mask = customer_metrics[
            eligibility_column
        ]

        eligible_customer_count = int(
            eligible_mask.sum()
        )

        repeat_customer_count = int(
            customer_metrics.loc[
                eligible_mask,
                repeat_column,
            ].sum()
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

        first_purchase_cutoff = (
            observation_cutoff
            - pd.Timedelta(days=window_days)
        )

        records.append(
            {
                "window_days": window_days,
                "observation_cutoff":
                    observation_cutoff,
                "eligible_first_purchase_cutoff":
                    first_purchase_cutoff,
                "eligible_customer_count":
                    eligible_customer_count,
                "repeat_customer_count":
                    repeat_customer_count,
                "non_repeat_customer_count":
                    non_repeat_customer_count,
                "repeat_rate_pct": round(
                    repeat_rate_pct,
                    4,
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 5. 생성 결과 검증
# ============================================================

def create_build_summary(
    retention_orders: pd.DataFrame,
    customer_metrics: pd.DataFrame,
    observation_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """고객 반복구매 지표 생성 결과를 검증합니다."""

    eligible_counts = [
        int(
            customer_metrics[
                f"eligible_{window_days}d"
            ].sum()
        )
        for window_days in RETENTION_WINDOWS
    ]

    eligible_counts_are_non_increasing = all(
        current_count >= next_count
        for current_count, next_count
        in zip(
            eligible_counts,
            eligible_counts[1:],
        )
    )

    if not eligible_counts_are_non_increasing:
        raise ValueError(
            "기간이 길어질수록 적격 고객 수가 "
            "감소해야 하지만 조건을 만족하지 않습니다."
        )

    summary_records = [
        {
            "check_name":
                "retention_order_rows",
            "value": len(retention_orders),
        },
        {
            "check_name":
                "customer_metric_rows",
            "value": len(customer_metrics),
        },
        {
            "check_name":
                "unique_customer_ids",
            "value": customer_metrics[
                "customer_unique_id"
            ].nunique(),
        },
        {
            "check_name":
                "duplicate_customer_id_rows",
            "value": int(
                customer_metrics[
                    "customer_unique_id"
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
                    "first_purchase_timestamp"
                ].isna().sum()
            ),
        },
        {
            "check_name":
                "null_second_purchase_rows",
            "value": int(
                customer_metrics[
                    "second_purchase_timestamp"
                ].isna().sum()
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
                "observation_cutoff",
            "value": observation_cutoff,
        },
        {
            "check_name":
                "eligible_counts_non_increasing",
            "value":
                eligible_counts_are_non_increasing,
        },
    ]

    return pd.DataFrame(summary_records)


# ============================================================
# 6. 실행 및 저장
# ============================================================

def main() -> None:
    retention_orders = load_retention_orders()

    (
        customer_metrics,
        observation_cutoff,
    ) = build_customer_retention_metrics(
        retention_orders
    )

    retention_window_summary = (
        create_retention_window_summary(
            customer_metrics,
            observation_cutoff,
        )
    )

    build_summary = create_build_summary(
        retention_orders,
        customer_metrics,
        observation_cutoff,
    )

    customer_metrics_path = (
        PROCESSED_DATA_DIR
        / "customer_retention_metrics.csv"
    )

    retention_summary_path = (
        OUTPUT_TABLE_DIR
        / "retention_window_summary.csv"
    )

    build_summary_path = (
        OUTPUT_TABLE_DIR
        / "retention_metrics_build_summary.csv"
    )

    customer_metrics.to_csv(
        customer_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    retention_window_summary.to_csv(
        retention_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    build_summary.to_csv(
        build_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[기간별 반복구매율]")
    print(
        retention_window_summary.to_string(
            index=False
        )
    )

    print("\n[고객 반복구매 지표 생성 검증]")
    print(
        build_summary.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {customer_metrics_path}")
    print(f"- {retention_summary_path}")
    print(f"- {build_summary_path}")


if __name__ == "__main__":
    main()
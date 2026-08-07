from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 및 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HM_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "HM"
    / "raw"
)

OUTPUT_TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRANSACTIONS_PATH = (
    HM_RAW_DIR / "transactions_train.csv"
)

CHUNK_SIZE = 1_000_000

REQUIRED_COLUMNS = [
    "t_dat",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
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


def update_counter(
    target_counter: Counter,
    value_counts: pd.Series,
) -> None:
    """pandas 빈도표를 Counter에 누적합니다."""

    for key, value in value_counts.items():
        target_counter[int(key)] += int(value)


# ============================================================
# 3. 날짜 단위 거래 블록 처리
# ============================================================

def process_complete_date_block(
    block: pd.DataFrame,
) -> dict[str, object]:
    """
    날짜가 완전히 포함된 거래 블록을 처리합니다.

    같은 날짜의 일부 행이 다음 청크로 넘어가지 않도록
    완성된 날짜 데이터만 전달받아 집계합니다.
    """

    if block.empty:
        return {
            "processed_rows": 0,
            "channel_event_count": 0,
            "customer_day_count": 0,
            "single_item_event_count": 0,
            "multi_item_event_count": 0,
            "multiple_article_event_count": 0,
            "repeated_article_event_count": 0,
            "maximum_item_rows_per_event": 0,
            "multi_channel_customer_day_count": 0,
            "extra_channel_events": 0,
            "multi_channel_customers": set(),
            "item_count_distribution": Counter(),
            "channel_count_distribution": Counter(),
        }

    # --------------------------------------------------------
    # 후보 1:
    # 고객 + 거래일 + 판매 채널 = 구매 이벤트 1회
    # --------------------------------------------------------

    channel_events = (
        block.groupby(
            [
                "customer_id",
                "t_dat",
                "sales_channel_id",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            item_row_count=(
                "article_id",
                "size",
            ),
            unique_article_count=(
                "article_id",
                "nunique",
            ),
        )
    )

    single_item_event_count = int(
        channel_events[
            "item_row_count"
        ].eq(1).sum()
    )

    multi_item_event_count = int(
        channel_events[
            "item_row_count"
        ].gt(1).sum()
    )

    multiple_article_event_count = int(
        channel_events[
            "unique_article_count"
        ].gt(1).sum()
    )

    # 같은 상품이 같은 고객·날짜·채널에서
    # 두 행 이상 나타난 구매 이벤트
    repeated_article_event_count = int(
        (
            channel_events["item_row_count"]
            > channel_events["unique_article_count"]
        ).sum()
    )

    maximum_item_rows_per_event = int(
        channel_events[
            "item_row_count"
        ].max()
    )

    item_count_distribution = Counter()

    update_counter(
        item_count_distribution,
        channel_events[
            "item_row_count"
        ].value_counts(),
    )

    # --------------------------------------------------------
    # 후보 2:
    # 고객 + 거래일 = 구매 이벤트 1회
    #
    # 같은 날 여러 채널 구매 여부 확인
    # --------------------------------------------------------

    customer_days = (
        channel_events.groupby(
            [
                "customer_id",
                "t_dat",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            channel_count=(
                "sales_channel_id",
                "nunique",
            ),
            channel_event_count=(
                "sales_channel_id",
                "size",
            ),
            item_row_count=(
                "item_row_count",
                "sum",
            ),
        )
    )

    multi_channel_mask = (
        customer_days[
            "channel_count"
        ].gt(1)
    )

    multi_channel_customer_day_count = int(
        multi_channel_mask.sum()
    )

    multi_channel_customers = set(
        customer_days.loc[
            multi_channel_mask,
            "customer_id",
        ].tolist()
    )

    # 채널을 구매 이벤트 기준에 포함할 때
    # 고객+날짜 기준보다 추가로 생성되는 이벤트 수
    extra_channel_events = int(
        (
            customer_days[
                "channel_event_count"
            ]
            - 1
        ).sum()
    )

    channel_count_distribution = Counter()

    update_counter(
        channel_count_distribution,
        customer_days[
            "channel_count"
        ].value_counts(),
    )

    return {
        "processed_rows": len(block),
        "channel_event_count": len(
            channel_events
        ),
        "customer_day_count": len(
            customer_days
        ),
        "single_item_event_count":
            single_item_event_count,
        "multi_item_event_count":
            multi_item_event_count,
        "multiple_article_event_count":
            multiple_article_event_count,
        "repeated_article_event_count":
            repeated_article_event_count,
        "maximum_item_rows_per_event":
            maximum_item_rows_per_event,
        "multi_channel_customer_day_count":
            multi_channel_customer_day_count,
        "extra_channel_events":
            extra_channel_events,
        "multi_channel_customers":
            multi_channel_customers,
        "item_count_distribution":
            item_count_distribution,
        "channel_count_distribution":
            channel_count_distribution,
    }


# ============================================================
# 4. H&M 구매 이벤트 구조 감사
# ============================================================

def audit_purchase_event_structure() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    대용량 거래 데이터를 날짜 순서대로 청크 처리하여
    구매 이벤트 대체키 후보를 비교합니다.
    """

    check_file_exists(
        TRANSACTIONS_PATH
    )

    total_rows = 0
    processed_rows = 0
    chunk_number = 0

    parse_failure_count = 0
    source_columns: list[str] | None = None

    previous_chunk_max_date = None
    dates_sorted_ascending = True

    carryover = pd.DataFrame()

    channel_event_count = 0
    customer_day_count = 0

    single_item_event_count = 0
    multi_item_event_count = 0

    multiple_article_event_count = 0
    repeated_article_event_count = 0

    maximum_item_rows_per_event = 0

    multi_channel_customer_day_count = 0
    extra_channel_events = 0

    multi_channel_customers: set[str] = set()

    item_count_distribution: Counter = Counter()
    channel_count_distribution: Counter = Counter()

    reader = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={
            "t_dat": "string",
            "customer_id": "string",
            "article_id": "string",
            "sales_channel_id": "string",
        },
        low_memory=False,
    )

    for chunk in reader:
        chunk_number += 1

        if source_columns is None:
            source_columns = list(
                chunk.columns
            )

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in source_columns
            ]

            if missing_columns:
                raise ValueError(
                    "transactions_train.csv에 "
                    "필수 컬럼이 없습니다: "
                    f"{missing_columns}"
                )

        total_rows += len(chunk)

        parsed_dates = pd.to_datetime(
            chunk["t_dat"],
            errors="coerce",
        )

        parse_failure_mask = (
            chunk["t_dat"].notna()
            & parsed_dates.isna()
        )

        parse_failure_count += int(
            parse_failure_mask.sum()
        )

        if parse_failure_mask.any():
            raise ValueError(
                "거래일 변환에 실패한 행이 있습니다."
            )

        chunk["t_dat"] = parsed_dates

        # 청크 내부 날짜 정렬 확인
        if not chunk[
            "t_dat"
        ].is_monotonic_increasing:
            dates_sorted_ascending = False

        current_chunk_min_date = (
            chunk["t_dat"].min()
        )

        current_chunk_max_date = (
            chunk["t_dat"].max()
        )

        # 이전 청크와 현재 청크 사이 날짜 정렬 확인
        if (
            previous_chunk_max_date is not None
            and current_chunk_min_date
            < previous_chunk_max_date
        ):
            dates_sorted_ascending = False

        previous_chunk_max_date = (
            current_chunk_max_date
        )

        if not dates_sorted_ascending:
            raise ValueError(
                "거래 데이터가 날짜 오름차순으로 "
                "정렬되어 있지 않습니다. "
                "현재 청크 처리 방식으로는 "
                "정확한 집계가 불가능합니다."
            )

        if not carryover.empty:
            chunk = pd.concat(
                [
                    carryover,
                    chunk,
                ],
                ignore_index=True,
            )

        # 마지막 날짜는 다음 청크에도 이어질 수 있으므로 보류
        last_date = chunk[
            "t_dat"
        ].iloc[-1]

        complete_block = chunk.loc[
            chunk["t_dat"].lt(last_date)
        ].copy()

        carryover = chunk.loc[
            chunk["t_dat"].eq(last_date)
        ].copy()

        block_result = (
            process_complete_date_block(
                complete_block
            )
        )

        processed_rows += int(
            block_result[
                "processed_rows"
            ]
        )

        channel_event_count += int(
            block_result[
                "channel_event_count"
            ]
        )

        customer_day_count += int(
            block_result[
                "customer_day_count"
            ]
        )

        single_item_event_count += int(
            block_result[
                "single_item_event_count"
            ]
        )

        multi_item_event_count += int(
            block_result[
                "multi_item_event_count"
            ]
        )

        multiple_article_event_count += int(
            block_result[
                "multiple_article_event_count"
            ]
        )

        repeated_article_event_count += int(
            block_result[
                "repeated_article_event_count"
            ]
        )

        maximum_item_rows_per_event = max(
            maximum_item_rows_per_event,
            int(
                block_result[
                    "maximum_item_rows_per_event"
                ]
            ),
        )

        multi_channel_customer_day_count += int(
            block_result[
                "multi_channel_customer_day_count"
            ]
        )

        extra_channel_events += int(
            block_result[
                "extra_channel_events"
            ]
        )

        multi_channel_customers.update(
            block_result[
                "multi_channel_customers"
            ]
        )

        item_count_distribution.update(
            block_result[
                "item_count_distribution"
            ]
        )

        channel_count_distribution.update(
            block_result[
                "channel_count_distribution"
            ]
        )

        print(
            f"청크 {chunk_number} 처리 완료 "
            f"- 원본 누적 행: {total_rows:,} "
            f"- 집계 완료 행: {processed_rows:,}"
        )

    # 마지막 날짜 데이터 처리
    if not carryover.empty:
        final_result = (
            process_complete_date_block(
                carryover
            )
        )

        processed_rows += int(
            final_result[
                "processed_rows"
            ]
        )

        channel_event_count += int(
            final_result[
                "channel_event_count"
            ]
        )

        customer_day_count += int(
            final_result[
                "customer_day_count"
            ]
        )

        single_item_event_count += int(
            final_result[
                "single_item_event_count"
            ]
        )

        multi_item_event_count += int(
            final_result[
                "multi_item_event_count"
            ]
        )

        multiple_article_event_count += int(
            final_result[
                "multiple_article_event_count"
            ]
        )

        repeated_article_event_count += int(
            final_result[
                "repeated_article_event_count"
            ]
        )

        maximum_item_rows_per_event = max(
            maximum_item_rows_per_event,
            int(
                final_result[
                    "maximum_item_rows_per_event"
                ]
            ),
        )

        multi_channel_customer_day_count += int(
            final_result[
                "multi_channel_customer_day_count"
            ]
        )

        extra_channel_events += int(
            final_result[
                "extra_channel_events"
            ]
        )

        multi_channel_customers.update(
            final_result[
                "multi_channel_customers"
            ]
        )

        item_count_distribution.update(
            final_result[
                "item_count_distribution"
            ]
        )

        channel_count_distribution.update(
            final_result[
                "channel_count_distribution"
            ]
        )

    if processed_rows != total_rows:
        raise ValueError(
            "처리된 행 수와 원본 행 수가 다릅니다. "
            f"원본={total_rows:,}, "
            f"처리={processed_rows:,}"
        )

    multi_item_event_rate_pct = (
        multi_item_event_count
        / channel_event_count
        * 100
        if channel_event_count > 0
        else 0.0
    )

    multi_channel_customer_day_rate_pct = (
        multi_channel_customer_day_count
        / customer_day_count
        * 100
        if customer_day_count > 0
        else 0.0
    )

    channel_split_increase_rate_pct = (
        extra_channel_events
        / customer_day_count
        * 100
        if customer_day_count > 0
        else 0.0
    )

    audit_summary = pd.DataFrame(
        [
            {
                "check_name":
                    "transaction_rows",
                "value": total_rows,
            },
            {
                "check_name":
                    "processed_rows",
                "value": processed_rows,
            },
            {
                "check_name":
                    "transaction_date_parse_failures",
                "value": parse_failure_count,
            },
            {
                "check_name":
                    "dates_sorted_ascending",
                "value": dates_sorted_ascending,
            },
            {
                "check_name":
                    "customer_date_channel_event_count",
                "value": channel_event_count,
            },
            {
                "check_name":
                    "customer_date_event_count",
                "value": customer_day_count,
            },
            {
                "check_name":
                    "extra_events_from_channel_split",
                "value": extra_channel_events,
            },
            {
                "check_name":
                    "channel_split_increase_rate_pct",
                "value": round(
                    channel_split_increase_rate_pct,
                    4,
                ),
            },
            {
                "check_name":
                    "single_item_event_count",
                "value": single_item_event_count,
            },
            {
                "check_name":
                    "multi_item_event_count",
                "value": multi_item_event_count,
            },
            {
                "check_name":
                    "multi_item_event_rate_pct",
                "value": round(
                    multi_item_event_rate_pct,
                    4,
                ),
            },
            {
                "check_name":
                    "events_with_multiple_unique_articles",
                "value":
                    multiple_article_event_count,
            },
            {
                "check_name":
                    "events_with_repeated_same_article",
                "value":
                    repeated_article_event_count,
            },
            {
                "check_name":
                    "maximum_item_rows_per_event",
                "value":
                    maximum_item_rows_per_event,
            },
            {
                "check_name":
                    "multi_channel_customer_day_count",
                "value":
                    multi_channel_customer_day_count,
            },
            {
                "check_name":
                    "multi_channel_customer_day_rate_pct",
                "value": round(
                    multi_channel_customer_day_rate_pct,
                    4,
                ),
            },
            {
                "check_name":
                    "multi_channel_unique_customer_count",
                "value": len(
                    multi_channel_customers
                ),
            },
        ]
    )

    definition_comparison = pd.DataFrame(
        [
            {
                "purchase_event_definition":
                    "customer_id + t_dat",
                "event_count":
                    customer_day_count,
                "difference_from_customer_date":
                    0,
                "increase_rate_pct":
                    0.0,
            },
            {
                "purchase_event_definition":
                    (
                        "customer_id + t_dat "
                        "+ sales_channel_id"
                    ),
                "event_count":
                    channel_event_count,
                "difference_from_customer_date":
                    extra_channel_events,
                "increase_rate_pct": round(
                    channel_split_increase_rate_pct,
                    4,
                ),
            },
        ]
    )

    item_distribution_records = []

    for item_count in sorted(
        item_count_distribution
    ):
        event_count = (
            item_count_distribution[
                item_count
            ]
        )

        item_distribution_records.append(
            {
                "item_row_count":
                    item_count,
                "event_count":
                    event_count,
                "event_ratio_pct": round(
                    event_count
                    / channel_event_count
                    * 100,
                    4,
                ),
            }
        )

    item_distribution = pd.DataFrame(
        item_distribution_records
    )

    channel_distribution_records = []

    for channel_count in sorted(
        channel_count_distribution
    ):
        day_count = (
            channel_count_distribution[
                channel_count
            ]
        )

        channel_distribution_records.append(
            {
                "channel_count":
                    channel_count,
                "customer_day_count":
                    day_count,
                "customer_day_ratio_pct":
                    round(
                        day_count
                        / customer_day_count
                        * 100,
                        4,
                    ),
            }
        )

    channel_distribution = pd.DataFrame(
        channel_distribution_records
    )

    return (
        audit_summary,
        definition_comparison,
        item_distribution,
        channel_distribution,
    )


# ============================================================
# 5. 실행 및 저장
# ============================================================

def main() -> None:
    print(
        "\n[H&M 구매 이벤트 구조 감사 시작]"
    )

    (
        audit_summary,
        definition_comparison,
        item_distribution,
        channel_distribution,
    ) = audit_purchase_event_structure()

    audit_summary_path = (
        OUTPUT_TABLE_DIR
        / "hm_purchase_event_audit_summary.csv"
    )

    definition_comparison_path = (
        OUTPUT_TABLE_DIR
        / "hm_purchase_event_definition_comparison.csv"
    )

    item_distribution_path = (
        OUTPUT_TABLE_DIR
        / "hm_purchase_event_item_count_distribution.csv"
    )

    channel_distribution_path = (
        OUTPUT_TABLE_DIR
        / "hm_customer_day_channel_distribution.csv"
    )

    audit_summary.to_csv(
        audit_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    definition_comparison.to_csv(
        definition_comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    item_distribution.to_csv(
        item_distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    channel_distribution.to_csv(
        channel_distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[구매 이벤트 감사 결과]")
    print(
        audit_summary.to_string(
            index=False
        )
    )

    print("\n[구매 이벤트 정의 비교]")
    print(
        definition_comparison.to_string(
            index=False
        )
    )

    print("\n[고객·거래일별 채널 수 분포]")
    print(
        channel_distribution.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {audit_summary_path}")
    print(f"- {definition_comparison_path}")
    print(f"- {item_distribution_path}")
    print(f"- {channel_distribution_path}")


if __name__ == "__main__":
    main()
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

TRANSACTIONS_PATH = (
    HM_RAW_DIR
    / "transactions_train.csv"
)

PURCHASE_EVENT_MART_PATH = (
    HM_PROCESSED_DIR
    / "hm_purchase_events.csv"
)

MART_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_purchase_event_mart_summary.csv"
)

CUSTOMER_DISTRIBUTION_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_customer_purchase_event_count_distribution.csv"
)

PRIOR_AUDIT_PATH = (
    OUTPUT_TABLE_DIR
    / "hm_purchase_event_audit_summary.csv"
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


def normalize_string(
    series: pd.Series,
) -> pd.Series:
    """문자열 ID의 좌우 공백을 제거합니다."""

    return (
        series
        .astype("string")
        .str.strip()
    )


def read_prior_audit_value(
    check_name: str,
) -> int | None:
    """
    이전 구매 이벤트 감사 결과가 존재하면
    특정 검증값을 불러옵니다.
    """

    if not PRIOR_AUDIT_PATH.exists():
        return None

    prior_audit = pd.read_csv(
        PRIOR_AUDIT_PATH,
        low_memory=False,
    )

    matched = prior_audit.loc[
        prior_audit["check_name"].eq(
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
# 3. 완성된 날짜 블록을 구매 이벤트로 집계
# ============================================================

def build_event_block(
    block: pd.DataFrame,
    customer_event_counts: dict[str, int],
) -> pd.DataFrame:
    """
    거래 상품행을 고객·날짜·판매 채널 단위의
    구매 이벤트로 집계합니다.
    """

    if block.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "purchase_date",
                "sales_channel_id",
                "customer_purchase_event_number",
                "item_row_count",
                "unique_article_count",
                "event_price_sum",
                "event_price_mean",
            ]
        )

    purchase_events = (
        block.groupby(
            [
                "customer_id",
                "t_dat",
                "sales_channel_id",
            ],
            as_index=False,
            dropna=False,
            sort=True,
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
            event_price_sum=(
                "price",
                "sum",
            ),
            event_price_mean=(
                "price",
                "mean",
            ),
        )
        .rename(
            columns={
                "t_dat": "purchase_date",
            }
        )
    )

    purchase_events = (
        purchase_events.sort_values(
            [
                "purchase_date",
                "customer_id",
                "sales_channel_id",
            ]
        )
        .reset_index(drop=True)
    )

    # 현재 블록 안에서 고객별 구매 순서
    purchase_events[
        "_event_number_in_block"
    ] = (
        purchase_events
        .groupby(
            "customer_id",
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    # 이전 블록까지 발생한 고객별 구매 횟수
    previous_event_counts = (
        purchase_events["customer_id"]
        .map(customer_event_counts)
        .fillna(0)
        .astype("int64")
    )

    purchase_events[
        "customer_purchase_event_number"
    ] = (
        previous_event_counts
        + purchase_events[
            "_event_number_in_block"
        ]
    )

    # 다음 블록을 위해 고객별 누적 구매 횟수 갱신
    block_customer_counts = (
        purchase_events
        .groupby(
            "customer_id",
            sort=False,
        )
        .size()
    )

    for customer_id, event_count in (
        block_customer_counts.items()
    ):
        customer_event_counts[customer_id] = (
            customer_event_counts.get(
                customer_id,
                0,
            )
            + int(event_count)
        )

    purchase_events = (
        purchase_events.drop(
            columns="_event_number_in_block"
        )
    )

    purchase_events[
        "event_price_sum"
    ] = (
        purchase_events[
            "event_price_sum"
        ]
        .round(8)
    )

    purchase_events[
        "event_price_mean"
    ] = (
        purchase_events[
            "event_price_mean"
        ]
        .round(8)
    )

    column_order = [
        "customer_id",
        "purchase_date",
        "sales_channel_id",
        "customer_purchase_event_number",
        "item_row_count",
        "unique_article_count",
        "event_price_sum",
        "event_price_mean",
    ]

    return purchase_events[column_order]


# ============================================================
# 4. 구매 이벤트 마트 저장
# ============================================================

def append_event_block(
    purchase_events: pd.DataFrame,
    is_first_write: bool,
) -> None:
    """구매 이벤트 블록을 CSV에 이어서 저장합니다."""

    if purchase_events.empty:
        return

    if is_first_write:
        purchase_events.to_csv(
            PURCHASE_EVENT_MART_PATH,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )
    else:
        purchase_events.to_csv(
            PURCHASE_EVENT_MART_PATH,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d",
        )


# ============================================================
# 5. 마트 생성
# ============================================================

def build_purchase_event_mart() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    H&M 상품 거래 데이터를 청크 단위로 읽어
    구매 이벤트 마트를 생성합니다.
    """

    check_file_exists(
        TRANSACTIONS_PATH
    )

    # 재실행 시 과거 결과가 중복 누적되지 않도록 삭제
    if PURCHASE_EVENT_MART_PATH.exists():
        PURCHASE_EVENT_MART_PATH.unlink()

    total_transaction_rows = 0
    processed_transaction_rows = 0
    total_purchase_event_rows = 0

    chunk_number = 0
    parse_failure_count = 0
    price_parse_failure_count = 0

    dates_sorted_ascending = True
    previous_chunk_max_date = None
    previous_processed_max_date = None

    minimum_purchase_date = None
    maximum_purchase_date = None

    maximum_item_rows_per_event = 0
    maximum_unique_articles_per_event = 0

    customer_event_counts: dict[
        str,
        int,
    ] = {}

    carryover = pd.DataFrame()
    is_first_write = True

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
        total_transaction_rows += len(chunk)

        # -----------------------------------------------
        # 문자열 ID 표준화
        # -----------------------------------------------

        chunk["customer_id"] = (
            normalize_string(
                chunk["customer_id"]
            )
        )

        chunk["article_id"] = (
            normalize_string(
                chunk["article_id"]
            )
        )

        chunk["sales_channel_id"] = (
            normalize_string(
                chunk["sales_channel_id"]
            )
        )

        # -----------------------------------------------
        # 날짜 변환
        # -----------------------------------------------

        parsed_dates = pd.to_datetime(
            chunk["t_dat"],
            errors="coerce",
        )

        date_failure_mask = (
            chunk["t_dat"].notna()
            & parsed_dates.isna()
        )

        parse_failure_count += int(
            date_failure_mask.sum()
        )

        if date_failure_mask.any():
            raise ValueError(
                "거래일 변환에 실패한 행이 있습니다."
            )

        chunk["t_dat"] = parsed_dates

        # -----------------------------------------------
        # 가격 변환
        # -----------------------------------------------

        parsed_prices = pd.to_numeric(
            chunk["price"],
            errors="coerce",
        )

        price_failure_mask = (
            chunk["price"].notna()
            & parsed_prices.isna()
        )

        price_parse_failure_count += int(
            price_failure_mask.sum()
        )

        if price_failure_mask.any():
            raise ValueError(
                "가격 변환에 실패한 행이 있습니다."
            )

        chunk["price"] = parsed_prices

        # -----------------------------------------------
        # 날짜 정렬 검증
        # -----------------------------------------------

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

        if (
            previous_chunk_max_date is not None
            and current_chunk_min_date
            < previous_chunk_max_date
        ):
            dates_sorted_ascending = False

        if not dates_sorted_ascending:
            raise ValueError(
                "transactions_train.csv가 "
                "날짜 오름차순으로 정렬되어 있지 않습니다."
            )

        previous_chunk_max_date = (
            current_chunk_max_date
        )

        # 이전 청크의 마지막 날짜 행을 현재 청크와 결합
        if not carryover.empty:
            chunk = pd.concat(
                [
                    carryover,
                    chunk,
                ],
                ignore_index=True,
            )

        # 현재 청크의 마지막 날짜는 다음 청크에도
        # 이어질 수 있으므로 아직 처리하지 않고 보류
        last_date = chunk[
            "t_dat"
        ].iloc[-1]

        complete_block = chunk.loc[
            chunk["t_dat"].lt(last_date)
        ].copy()

        carryover = chunk.loc[
            chunk["t_dat"].eq(last_date)
        ].copy()

        if not complete_block.empty:
            block_min_date = (
                complete_block[
                    "t_dat"
                ].min()
            )

            block_max_date = (
                complete_block[
                    "t_dat"
                ].max()
            )

            # 이전에 처리한 날짜와 겹치면
            # 블록 간 중복 가능성이 있으므로 중단
            if (
                previous_processed_max_date
                is not None
                and block_min_date
                <= previous_processed_max_date
            ):
                raise ValueError(
                    "처리 날짜 블록이 이전 블록과 겹칩니다."
                )

            purchase_events = (
                build_event_block(
                    block=complete_block,
                    customer_event_counts=(
                        customer_event_counts
                    ),
                )
            )

            duplicate_key_rows = int(
                purchase_events.duplicated(
                    subset=[
                        "customer_id",
                        "purchase_date",
                        "sales_channel_id",
                    ],
                    keep=False,
                ).sum()
            )

            if duplicate_key_rows > 0:
                raise ValueError(
                    "구매 이벤트 대체키 중복이 "
                    f"{duplicate_key_rows:,}행 발생했습니다."
                )

            append_event_block(
                purchase_events=(
                    purchase_events
                ),
                is_first_write=(
                    is_first_write
                ),
            )

            is_first_write = False

            processed_transaction_rows += (
                len(complete_block)
            )

            total_purchase_event_rows += (
                len(purchase_events)
            )

            maximum_item_rows_per_event = max(
                maximum_item_rows_per_event,
                int(
                    purchase_events[
                        "item_row_count"
                    ].max()
                ),
            )

            maximum_unique_articles_per_event = max(
                maximum_unique_articles_per_event,
                int(
                    purchase_events[
                        "unique_article_count"
                    ].max()
                ),
            )

            if minimum_purchase_date is None:
                minimum_purchase_date = (
                    block_min_date
                )

            maximum_purchase_date = (
                block_max_date
            )

            previous_processed_max_date = (
                block_max_date
            )

        print(
            f"청크 {chunk_number} 처리 완료 "
            f"- 원본 누적 행: "
            f"{total_transaction_rows:,} "
            f"- 집계 완료 행: "
            f"{processed_transaction_rows:,} "
            f"- 구매 이벤트: "
            f"{total_purchase_event_rows:,}"
        )

    # ========================================================
    # 마지막 날짜 블록 처리
    # ========================================================

    if not carryover.empty:
        block_min_date = (
            carryover["t_dat"].min()
        )

        block_max_date = (
            carryover["t_dat"].max()
        )

        if (
            previous_processed_max_date
            is not None
            and block_min_date
            <= previous_processed_max_date
        ):
            raise ValueError(
                "마지막 날짜 블록이 이전 블록과 겹칩니다."
            )

        purchase_events = build_event_block(
            block=carryover,
            customer_event_counts=(
                customer_event_counts
            ),
        )

        duplicate_key_rows = int(
            purchase_events.duplicated(
                subset=[
                    "customer_id",
                    "purchase_date",
                    "sales_channel_id",
                ],
                keep=False,
            ).sum()
        )

        if duplicate_key_rows > 0:
            raise ValueError(
                "마지막 블록에서 구매 이벤트 "
                f"대체키 중복이 {duplicate_key_rows:,}행 "
                "발생했습니다."
            )

        append_event_block(
            purchase_events=purchase_events,
            is_first_write=is_first_write,
        )

        processed_transaction_rows += len(
            carryover
        )

        total_purchase_event_rows += len(
            purchase_events
        )

        maximum_item_rows_per_event = max(
            maximum_item_rows_per_event,
            int(
                purchase_events[
                    "item_row_count"
                ].max()
            ),
        )

        maximum_unique_articles_per_event = max(
            maximum_unique_articles_per_event,
            int(
                purchase_events[
                    "unique_article_count"
                ].max()
            ),
        )

        if minimum_purchase_date is None:
            minimum_purchase_date = (
                block_min_date
            )

        maximum_purchase_date = (
            block_max_date
        )

    # ========================================================
    # 최종 검증
    # ========================================================

    if (
        processed_transaction_rows
        != total_transaction_rows
    ):
        raise ValueError(
            "원본 행 수와 처리 행 수가 다릅니다. "
            f"원본={total_transaction_rows:,}, "
            f"처리={processed_transaction_rows:,}"
        )

    prior_transaction_rows = (
        read_prior_audit_value(
            "transaction_rows"
        )
    )

    prior_purchase_event_rows = (
        read_prior_audit_value(
            "customer_date_channel_event_count"
        )
    )

    transaction_rows_match_prior = (
        prior_transaction_rows is None
        or total_transaction_rows
        == prior_transaction_rows
    )

    purchase_event_rows_match_prior = (
        prior_purchase_event_rows is None
        or total_purchase_event_rows
        == prior_purchase_event_rows
    )

    if not transaction_rows_match_prior:
        raise ValueError(
            "마트의 원본 거래행 수가 "
            "이전 감사 결과와 다릅니다."
        )

    if not purchase_event_rows_match_prior:
        raise ValueError(
            "마트의 구매 이벤트 수가 "
            "이전 감사 결과와 다릅니다."
        )

    # ========================================================
    # 고객별 구매 이벤트 분포
    # ========================================================

    customer_count_distribution = Counter(
        customer_event_counts.values()
    )

    total_customers = len(
        customer_event_counts
    )

    single_event_customers = sum(
        customer_count
        for event_count, customer_count
        in customer_count_distribution.items()
        if event_count == 1
    )

    repeat_customers = sum(
        customer_count
        for event_count, customer_count
        in customer_count_distribution.items()
        if event_count >= 2
    )

    maximum_customer_event_count = max(
        customer_event_counts.values()
    )

    distribution_records = []

    for event_count in sorted(
        customer_count_distribution
    ):
        customer_count = (
            customer_count_distribution[
                event_count
            ]
        )

        distribution_records.append(
            {
                "purchase_event_count":
                    event_count,
                "customer_count":
                    customer_count,
                "customer_ratio_pct": round(
                    customer_count
                    / total_customers
                    * 100,
                    4,
                ),
            }
        )

    customer_distribution = (
        pd.DataFrame(
            distribution_records
        )
    )

    # ========================================================
    # 마트 생성 요약
    # ========================================================

    mart_summary = pd.DataFrame(
        [
            {
                "check_name":
                    "source_transaction_rows",
                "value":
                    total_transaction_rows,
            },
            {
                "check_name":
                    "processed_transaction_rows",
                "value":
                    processed_transaction_rows,
            },
            {
                "check_name":
                    "purchase_event_mart_rows",
                "value":
                    total_purchase_event_rows,
            },
            {
                "check_name":
                    "unique_customers",
                "value":
                    total_customers,
            },
            {
                "check_name":
                    "single_event_customers",
                "value":
                    single_event_customers,
            },
            {
                "check_name":
                    "repeat_customers_all_time",
                "value":
                    repeat_customers,
            },
            {
                "check_name":
                    "maximum_customer_event_count",
                "value":
                    maximum_customer_event_count,
            },
            {
                "check_name":
                    "minimum_purchase_date",
                "value":
                    minimum_purchase_date,
            },
            {
                "check_name":
                    "maximum_purchase_date",
                "value":
                    maximum_purchase_date,
            },
            {
                "check_name":
                    "maximum_item_rows_per_event",
                "value":
                    maximum_item_rows_per_event,
            },
            {
                "check_name":
                    "maximum_unique_articles_per_event",
                "value":
                    maximum_unique_articles_per_event,
            },
            {
                "check_name":
                    "transaction_date_parse_failures",
                "value":
                    parse_failure_count,
            },
            {
                "check_name":
                    "price_parse_failures",
                "value":
                    price_parse_failure_count,
            },
            {
                "check_name":
                    "dates_sorted_ascending",
                "value":
                    dates_sorted_ascending,
            },
            {
                "check_name":
                    "transaction_rows_match_prior_audit",
                "value":
                    transaction_rows_match_prior,
            },
            {
                "check_name":
                    "purchase_event_rows_match_prior_audit",
                "value":
                    purchase_event_rows_match_prior,
            },
        ]
    )

    return (
        mart_summary,
        customer_distribution,
    )


# ============================================================
# 6. 실행 및 저장
# ============================================================

def main() -> None:
    print(
        "\n[H&M 구매 이벤트 마트 생성 시작]"
    )

    (
        mart_summary,
        customer_distribution,
    ) = build_purchase_event_mart()

    mart_summary.to_csv(
        MART_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    customer_distribution.to_csv(
        CUSTOMER_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[H&M 구매 이벤트 마트 생성 결과]"
    )

    print(
        mart_summary.to_string(
            index=False
        )
    )

    print(
        "\n[고객별 구매 이벤트 횟수 분포]"
    )

    print(
        customer_distribution.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {PURCHASE_EVENT_MART_PATH}")
    print(f"- {MART_SUMMARY_PATH}")
    print(f"- {CUSTOMER_DISTRIBUTION_PATH}")


if __name__ == "__main__":
    main()
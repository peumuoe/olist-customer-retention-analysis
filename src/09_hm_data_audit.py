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

ARTICLES_PATH = (
    HM_RAW_DIR / "articles.csv"
)

# 거래 데이터가 약 3,178만 행이므로 나누어 읽음
CHUNK_SIZE = 1_000_000


# ============================================================
# 2. 공통 함수
# ============================================================

def normalize_article_id(
    series: pd.Series,
) -> pd.Series:
    """
    H&M article_id를 10자리 문자열로 표준화합니다.

    예:
    108775015 → 0108775015
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.zfill(10)
    )


def check_file_exists(
    file_path: Path,
) -> None:
    """필수 파일이 존재하는지 확인합니다."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )


# ============================================================
# 3. articles.csv 감사
# ============================================================

def audit_articles() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    set[str],
]:
    """
    상품 데이터의 행 수, 컬럼, 결측치,
    기본키 중복 여부를 점검합니다.
    """

    check_file_exists(ARTICLES_PATH)

    # article_id를 처음부터 문자열로 읽어 앞자리 0 보존
    articles = pd.read_csv(
        ARTICLES_PATH,
        dtype={
            "article_id": "string",
        },
        low_memory=False,
    )

    required_columns = [
        "article_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in articles.columns
    ]

    if missing_columns:
        raise ValueError(
            "articles.csv에 필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    # article_id 형식 통일
    articles["article_id"] = normalize_article_id(
        articles["article_id"]
    )

    article_summary = pd.DataFrame(
        [
            {
                "check_name": "article_rows",
                "value": len(articles),
            },
            {
                "check_name": "article_column_count",
                "value": articles.shape[1],
            },
            {
                "check_name": "unique_article_ids",
                "value": int(
                    articles["article_id"]
                    .nunique(dropna=True)
                ),
            },
            {
                "check_name": "null_article_id_rows",
                "value": int(
                    articles["article_id"]
                    .isna()
                    .sum()
                ),
            },
            {
                "check_name": "duplicate_article_id_rows",
                "value": int(
                    articles["article_id"]
                    .duplicated(keep=False)
                    .sum()
                ),
            },
            {
                "check_name": "exact_duplicate_rows",
                "value": int(
                    articles
                    .duplicated()
                    .sum()
                ),
            },
        ]
    )

    schema_records = []

    for column in articles.columns:
        schema_records.append(
            {
                "column_name": column,
                "dtype": str(
                    articles[column].dtype
                ),
                "null_count": int(
                    articles[column]
                    .isna()
                    .sum()
                ),
                "null_rate_pct": round(
                    articles[column]
                    .isna()
                    .mean()
                    * 100,
                    4,
                ),
                "unique_count": int(
                    articles[column]
                    .nunique(dropna=True)
                ),
            }
        )

    article_schema_summary = pd.DataFrame(
        schema_records
    )

    article_id_set = set(
        articles["article_id"]
        .dropna()
        .tolist()
    )

    return (
        article_summary,
        article_schema_summary,
        article_id_set,
    )


# ============================================================
# 4. transactions_train.csv 감사
# ============================================================

def audit_transactions(
    article_id_set: set[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    대용량 거래 데이터를 청크 단위로 읽어 다음을 점검합니다.

    - 전체 거래행 수
    - 고객 수
    - 상품 수
    - 판매 채널 수
    - 거래기간
    - 날짜 변환 실패
    - 필수 컬럼 결측
    - 상품 마스터 미연결 여부
    """

    check_file_exists(TRANSACTIONS_PATH)

    required_columns = [
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id",
    ]

    total_rows = 0
    chunk_number = 0

    parse_failure_count = 0

    null_counts = {
        column: 0
        for column in required_columns
    }

    minimum_transaction_date = None
    maximum_transaction_date = None

    unique_customers: set[str] = set()
    unique_articles: set[str] = set()
    unique_channels: set[str] = set()

    unmatched_article_rows = 0
    unmatched_article_ids: set[str] = set()

    source_columns: list[str] | None = None

    reader = pd.read_csv(
        TRANSACTIONS_PATH,
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
                for column in required_columns
                if column not in source_columns
            ]

            if missing_columns:
                raise ValueError(
                    "transactions_train.csv에 "
                    "필수 컬럼이 없습니다: "
                    f"{missing_columns}"
                )

        # article_id를 articles.csv와 동일하게 10자리로 통일
        chunk["article_id"] = normalize_article_id(
            chunk["article_id"]
        )

        total_rows += len(chunk)

        # 필수 컬럼별 결측치
        for column in required_columns:
            null_counts[column] += int(
                chunk[column]
                .isna()
                .sum()
            )

        # 날짜 변환
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

        chunk_min_date = parsed_dates.min()
        chunk_max_date = parsed_dates.max()

        if pd.notna(chunk_min_date):
            if (
                minimum_transaction_date is None
                or chunk_min_date
                < minimum_transaction_date
            ):
                minimum_transaction_date = (
                    chunk_min_date
                )

        if pd.notna(chunk_max_date):
            if (
                maximum_transaction_date is None
                or chunk_max_date
                > maximum_transaction_date
            ):
                maximum_transaction_date = (
                    chunk_max_date
                )

        # 고유 고객
        unique_customers.update(
            chunk["customer_id"]
            .dropna()
            .tolist()
        )

        # 고유 상품
        unique_articles.update(
            chunk["article_id"]
            .dropna()
            .tolist()
        )

        # 고유 판매 채널
        unique_channels.update(
            chunk["sales_channel_id"]
            .dropna()
            .tolist()
        )

        # 상품 마스터와 연결되지 않는 거래 상품 확인
        valid_article_mask = (
            chunk["article_id"].notna()
        )

        unmatched_mask = (
            valid_article_mask
            & ~chunk["article_id"].isin(
                article_id_set
            )
        )

        unmatched_values = (
            chunk.loc[
                unmatched_mask,
                "article_id",
            ]
        )

        unmatched_article_rows += int(
            unmatched_mask.sum()
        )

        unmatched_article_ids.update(
            unmatched_values.tolist()
        )

        print(
            f"청크 {chunk_number} 처리 완료 "
            f"- 누적 행 수: {total_rows:,}"
        )

    if source_columns is None:
        raise ValueError(
            "transactions_train.csv에 "
            "읽을 수 있는 데이터가 없습니다."
        )

    transaction_summary_records = [
        {
            "check_name": "transaction_rows",
            "value": total_rows,
        },
        {
            "check_name": "transaction_column_count",
            "value": len(source_columns),
        },
        {
            "check_name": "unique_customers",
            "value": len(unique_customers),
        },
        {
            "check_name":
                "unique_articles_in_transactions",
            "value": len(unique_articles),
        },
        {
            "check_name": "unique_sales_channels",
            "value": len(unique_channels),
        },
        {
            "check_name":
                "minimum_transaction_date",
            "value": minimum_transaction_date,
        },
        {
            "check_name":
                "maximum_transaction_date",
            "value": maximum_transaction_date,
        },
        {
            "check_name":
                "transaction_date_parse_failures",
            "value": parse_failure_count,
        },
        {
            "check_name":
                "unmatched_article_rows",
            "value": unmatched_article_rows,
        },
        {
            "check_name":
                "unmatched_unique_article_ids",
            "value": len(unmatched_article_ids),
        },
    ]

    for column in required_columns:
        transaction_summary_records.append(
            {
                "check_name":
                    f"null_{column}_rows",
                "value": null_counts[column],
            }
        )

    transaction_summary = pd.DataFrame(
        transaction_summary_records
    )

    transaction_column_summary = pd.DataFrame(
        [
            {
                "column_name": column,
                "is_required":
                    column in required_columns,
            }
            for column in source_columns
        ]
    )

    return (
        transaction_summary,
        transaction_column_summary,
    )


# ============================================================
# 5. 실행 및 저장
# ============================================================

def main() -> None:
    print("\n[1단계] H&M 상품 데이터 감사 시작")

    (
        article_summary,
        article_schema_summary,
        article_id_set,
    ) = audit_articles()

    print(
        f"상품 마스터 ID 수: "
        f"{len(article_id_set):,}"
    )

    print(
        "\n[2단계] H&M 거래 데이터 감사 시작"
    )

    (
        transaction_summary,
        transaction_column_summary,
    ) = audit_transactions(
        article_id_set=article_id_set
    )

    article_summary_path = (
        OUTPUT_TABLE_DIR
        / "hm_article_audit_summary.csv"
    )

    article_schema_path = (
        OUTPUT_TABLE_DIR
        / "hm_article_schema_summary.csv"
    )

    transaction_summary_path = (
        OUTPUT_TABLE_DIR
        / "hm_transaction_audit_summary.csv"
    )

    transaction_columns_path = (
        OUTPUT_TABLE_DIR
        / "hm_transaction_columns.csv"
    )

    article_summary.to_csv(
        article_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    article_schema_summary.to_csv(
        article_schema_path,
        index=False,
        encoding="utf-8-sig",
    )

    transaction_summary.to_csv(
        transaction_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    transaction_column_summary.to_csv(
        transaction_columns_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[H&M 상품 데이터 감사 결과]")
    print(
        article_summary.to_string(
            index=False
        )
    )

    print("\n[H&M 거래 데이터 감사 결과]")
    print(
        transaction_summary.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {article_summary_path}")
    print(f"- {article_schema_path}")
    print(f"- {transaction_summary_path}")
    print(f"- {transaction_columns_path}")


if __name__ == "__main__":
    main()
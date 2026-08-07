from pathlib import Path

import pandas as pd


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 검증할 파일과 키
KEY_RULES = [
    {
        "table_name": "customers",
        "file_name": "olist_customers_dataset.csv",
        "key_columns": ["customer_id"],
        "key_type": "primary_key",
    },
    {
        "table_name": "orders",
        "file_name": "olist_orders_dataset.csv",
        "key_columns": ["order_id"],
        "key_type": "primary_key",
    },
    {
        "table_name": "products",
        "file_name": "olist_products_dataset.csv",
        "key_columns": ["product_id"],
        "key_type": "primary_key",
    },
    {
        "table_name": "sellers",
        "file_name": "olist_sellers_dataset.csv",
        "key_columns": ["seller_id"],
        "key_type": "primary_key",
    },
    {
        "table_name": "category_translation",
        "file_name": "product_category_name_translation.csv",
        "key_columns": ["product_category_name"],
        "key_type": "primary_key",
    },
    {
        "table_name": "order_items",
        "file_name": "olist_order_items_dataset.csv",
        "key_columns": ["order_id", "order_item_id"],
        "key_type": "composite_key",
    },
    {
        "table_name": "payments",
        "file_name": "olist_order_payments_dataset.csv",
        "key_columns": ["order_id", "payment_sequential"],
        "key_type": "composite_key",
    },
    {
        "table_name": "reviews",
        "file_name": "olist_order_reviews_dataset.csv",
        "key_columns": ["review_id"],
        "key_type": "candidate_key",
    },
]


def load_csv(file_name: str) -> pd.DataFrame:
    """data/raw에서 CSV 파일을 불러옵니다."""

    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    return pd.read_csv(
        file_path,
        low_memory=False,
    )


def audit_key(
    table_name: str,
    df: pd.DataFrame,
    key_columns: list[str],
    key_type: str,
) -> dict[str, object]:
    """키 컬럼의 결측과 중복 여부를 검사합니다."""

    missing_columns = [
        column
        for column in key_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{table_name}에 다음 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    null_key_rows = int(
        df[key_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    # 같은 키를 가진 모든 행을 중복 행으로 계산
    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    duplicate_key_rows = int(
        duplicate_mask.sum()
    )

    unique_key_count = int(
        df[key_columns]
        .drop_duplicates()
        .shape[0]
    )

    return {
        "table_name": table_name,
        "key_type": key_type,
        "key_columns": " + ".join(key_columns),
        "row_count": len(df),
        "null_key_rows": null_key_rows,
        "duplicate_key_rows": duplicate_key_rows,
        "unique_key_count": unique_key_count,
        "is_unique": duplicate_key_rows == 0,
    }


def main() -> None:
    results = []

    for rule in KEY_RULES:
        df = load_csv(rule["file_name"])

        result = audit_key(
            table_name=rule["table_name"],
            df=df,
            key_columns=rule["key_columns"],
            key_type=rule["key_type"],
        )

        results.append(result)

    key_audit = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "key_audit.csv"

    key_audit.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[기본키 및 복합키 검증 결과]")
    print(key_audit.to_string(index=False))

    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
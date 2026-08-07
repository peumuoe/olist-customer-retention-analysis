from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CSV 로딩
# ============================================================

def load_csv(
    file_name: str,
    usecols: list[str],
) -> pd.DataFrame:
    """필요한 컬럼만 선택하여 CSV를 불러옵니다."""

    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    return pd.read_csv(
        file_path,
        usecols=usecols,
        low_memory=False,
    )


# ============================================================
# 3. 외래키 검증 함수
# ============================================================

def audit_foreign_key(
    relationship_name: str,
    child_df: pd.DataFrame,
    child_key: str,
    parent_df: pd.DataFrame,
    parent_key: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """
    자식 테이블의 외래키가 부모 테이블에 존재하는지 확인합니다.
    """

    child_values = child_df[child_key]
    parent_values = parent_df[parent_key]

    # 결측치가 아닌 외래키만 검사
    valid_child_mask = child_values.notna()

    unmatched_mask = (
        valid_child_mask
        & ~child_values.isin(parent_values)
    )

    checked_rows = int(valid_child_mask.sum())
    unmatched_rows = int(unmatched_mask.sum())
    matched_rows = checked_rows - unmatched_rows

    matched_rate_pct = (
        round(matched_rows / checked_rows * 100, 4)
        if checked_rows > 0
        else 0.0
    )

    summary = {
        "relationship": relationship_name,
        "child_row_count": len(child_df),
        "null_child_key_rows": int(
            child_values.isna().sum()
        ),
        "checked_rows": checked_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "unmatched_unique_keys": int(
            child_values.loc[unmatched_mask].nunique()
        ),
        "matched_rate_pct": matched_rate_pct,
    }

    unmatched_detail = (
        child_df.loc[
            unmatched_mask,
            [child_key],
        ]
        .value_counts()
        .reset_index(name="row_count")
    )

    unmatched_detail.insert(
        0,
        "relationship",
        relationship_name,
    )

    unmatched_detail = unmatched_detail.rename(
        columns={child_key: "unmatched_key"}
    )

    return summary, unmatched_detail


# ============================================================
# 4. 실행
# ============================================================

def main() -> None:
    customers = load_csv(
        "olist_customers_dataset.csv",
        ["customer_id"],
    )

    orders = load_csv(
        "olist_orders_dataset.csv",
        ["order_id", "customer_id"],
    )

    order_items = load_csv(
        "olist_order_items_dataset.csv",
        ["order_id", "product_id", "seller_id"],
    )

    products = load_csv(
        "olist_products_dataset.csv",
        ["product_id", "product_category_name"],
    )

    sellers = load_csv(
        "olist_sellers_dataset.csv",
        ["seller_id"],
    )

    payments = load_csv(
        "olist_order_payments_dataset.csv",
        ["order_id"],
    )

    reviews = load_csv(
        "olist_order_reviews_dataset.csv",
        ["order_id"],
    )

    category_translation = load_csv(
        "product_category_name_translation.csv",
        ["product_category_name"],
    )

    relationship_rules = [
        {
            "relationship_name": (
                "orders.customer_id "
                "-> customers.customer_id"
            ),
            "child_df": orders,
            "child_key": "customer_id",
            "parent_df": customers,
            "parent_key": "customer_id",
        },
        {
            "relationship_name": (
                "order_items.order_id "
                "-> orders.order_id"
            ),
            "child_df": order_items,
            "child_key": "order_id",
            "parent_df": orders,
            "parent_key": "order_id",
        },
        {
            "relationship_name": (
                "order_items.product_id "
                "-> products.product_id"
            ),
            "child_df": order_items,
            "child_key": "product_id",
            "parent_df": products,
            "parent_key": "product_id",
        },
        {
            "relationship_name": (
                "order_items.seller_id "
                "-> sellers.seller_id"
            ),
            "child_df": order_items,
            "child_key": "seller_id",
            "parent_df": sellers,
            "parent_key": "seller_id",
        },
        {
            "relationship_name": (
                "payments.order_id "
                "-> orders.order_id"
            ),
            "child_df": payments,
            "child_key": "order_id",
            "parent_df": orders,
            "parent_key": "order_id",
        },
        {
            "relationship_name": (
                "reviews.order_id "
                "-> orders.order_id"
            ),
            "child_df": reviews,
            "child_key": "order_id",
            "parent_df": orders,
            "parent_key": "order_id",
        },
        {
            "relationship_name": (
                "products.product_category_name "
                "-> category_translation.product_category_name"
            ),
            "child_df": products.loc[
                products["product_category_name"].notna()
            ],
            "child_key": "product_category_name",
            "parent_df": category_translation,
            "parent_key": "product_category_name",
        },
    ]

    summary_results = []
    unmatched_results = []

    for rule in relationship_rules:
        summary, unmatched_detail = audit_foreign_key(
            relationship_name=rule["relationship_name"],
            child_df=rule["child_df"],
            child_key=rule["child_key"],
            parent_df=rule["parent_df"],
            parent_key=rule["parent_key"],
        )

        summary_results.append(summary)

        if not unmatched_detail.empty:
            unmatched_results.append(unmatched_detail)

    foreign_key_audit = pd.DataFrame(summary_results)

    if unmatched_results:
        unmatched_foreign_keys = pd.concat(
            unmatched_results,
            ignore_index=True,
        )
    else:
        unmatched_foreign_keys = pd.DataFrame(
            columns=[
                "relationship",
                "unmatched_key",
                "row_count",
            ]
        )

    audit_output_path = (
        OUTPUT_DIR / "foreign_key_audit.csv"
    )

    unmatched_output_path = (
        OUTPUT_DIR / "unmatched_foreign_keys.csv"
    )

    foreign_key_audit.to_csv(
        audit_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    unmatched_foreign_keys.to_csv(
        unmatched_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[외래키 연결 검증 결과]")
    print(foreign_key_audit.to_string(index=False))

    print("\n[연결되지 않은 외래키]")
    if unmatched_foreign_keys.empty:
        print("연결되지 않은 외래키가 없습니다.")
    else:
        print(
            unmatched_foreign_keys.to_string(
                index=False
            )
        )

    print("\n저장 완료:")
    print(f"- {audit_output_path}")
    print(f"- {unmatched_output_path}")


if __name__ == "__main__":
    main()
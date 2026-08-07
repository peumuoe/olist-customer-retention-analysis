from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 데이터 로딩
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """주문 및 고객 데이터를 불러옵니다."""

    orders_path = (
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    customers_path = (
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    )

    if not orders_path.exists():
        raise FileNotFoundError(
            f"주문 파일을 찾을 수 없습니다: {orders_path}"
        )

    if not customers_path.exists():
        raise FileNotFoundError(
            f"고객 파일을 찾을 수 없습니다: {customers_path}"
        )

    orders = pd.read_csv(
        orders_path,
        usecols=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ],
        low_memory=False,
    )

    customers = pd.read_csv(
        customers_path,
        usecols=[
            "customer_id",
            "customer_unique_id",
        ],
        low_memory=False,
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    return orders, customers


# ============================================================
# 3. 반복구매 주문 마트 생성
# ============================================================

def build_retention_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """배송 완료 주문과 고유 고객 ID를 결합합니다."""

    delivered_orders = (
        orders.loc[
            orders["order_status"].eq("delivered"),
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
            ],
        ]
        .copy()
    )

    retention_orders = delivered_orders.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_customer_rows = int(
        retention_orders["_merge"].ne("both").sum()
    )

    if unmatched_customer_rows > 0:
        raise ValueError(
            "고객 테이블과 연결되지 않은 주문이 있습니다: "
            f"{unmatched_customer_rows}행"
        )

    retention_orders = retention_orders.drop(
        columns="_merge"
    )

    required_columns = [
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_purchase_timestamp",
    ]

    missing_required_rows = (
        retention_orders[required_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing_required_rows > 0:
        raise ValueError(
            "필수값이 누락된 주문이 있습니다: "
            f"{missing_required_rows}행"
        )

    retention_orders = retention_orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_id",
        ]
    ).reset_index(drop=True)

    # 고객별 구매 순서
    retention_orders["customer_order_number"] = (
        retention_orders
        .groupby("customer_unique_id")
        .cumcount()
        .add(1)
    )

    # 고객별 최초 구매 시각
    retention_orders[
        "customer_first_purchase_timestamp"
    ] = (
        retention_orders
        .groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .transform("min")
    )

    # 직전 구매 시각
    retention_orders[
        "previous_purchase_timestamp"
    ] = (
        retention_orders
        .groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .shift(1)
    )

    # 직전 구매 후 경과일
    retention_orders[
        "days_since_previous_purchase"
    ] = (
        (
            retention_orders[
                "order_purchase_timestamp"
            ]
            - retention_orders[
                "previous_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        .div(86_400)
        .round(6)
    )

    column_order = [
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_status",
        "order_purchase_timestamp",
        "customer_order_number",
        "customer_first_purchase_timestamp",
        "previous_purchase_timestamp",
        "days_since_previous_purchase",
    ]

    return retention_orders[column_order]


# ============================================================
# 4. 마트 품질 검증 요약
# ============================================================

def create_build_summary(
    source_orders: pd.DataFrame,
    retention_orders: pd.DataFrame,
) -> pd.DataFrame:
    """생성된 주문 마트의 품질을 검증합니다."""

    customer_order_counts = (
        retention_orders
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    summary_records = [
        {
            "check_name": "source_order_rows",
            "value": len(source_orders),
        },
        {
            "check_name": "source_delivered_order_rows",
            "value": int(
                source_orders[
                    "order_status"
                ].eq("delivered").sum()
            ),
        },
        {
            "check_name": "retention_mart_rows",
            "value": len(retention_orders),
        },
        {
            "check_name": "unique_order_ids",
            "value": retention_orders[
                "order_id"
            ].nunique(),
        },
        {
            "check_name": "duplicate_order_id_rows",
            "value": int(
                retention_orders[
                    "order_id"
                ].duplicated(keep=False).sum()
            ),
        },
        {
            "check_name": "null_order_id_rows",
            "value": int(
                retention_orders[
                    "order_id"
                ].isna().sum()
            ),
        },
        {
            "check_name": "null_customer_id_rows",
            "value": int(
                retention_orders[
                    "customer_id"
                ].isna().sum()
            ),
        },
        {
            "check_name": "null_customer_unique_id_rows",
            "value": int(
                retention_orders[
                    "customer_unique_id"
                ].isna().sum()
            ),
        },
        {
            "check_name": "unique_customers",
            "value": customer_order_counts.size,
        },
        {
            "check_name": "single_order_customers",
            "value": int(
                customer_order_counts.eq(1).sum()
            ),
        },
        {
            "check_name": "repeat_customers",
            "value": int(
                customer_order_counts.ge(2).sum()
            ),
        },
        {
            "check_name": "maximum_customer_order_count",
            "value": int(
                customer_order_counts.max()
            ),
        },
        {
            "check_name": "minimum_purchase_timestamp",
            "value": retention_orders[
                "order_purchase_timestamp"
            ].min(),
        },
        {
            "check_name": "maximum_purchase_timestamp",
            "value": retention_orders[
                "order_purchase_timestamp"
            ].max(),
        },
    ]

    return pd.DataFrame(summary_records)


def create_order_count_distribution(
    retention_orders: pd.DataFrame,
) -> pd.DataFrame:
    """고객별 주문 횟수 분포를 생성합니다."""

    customer_order_counts = (
        retention_orders
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    distribution = (
        customer_order_counts
        .value_counts()
        .sort_index()
        .rename_axis("order_count")
        .reset_index(name="customer_count")
    )

    distribution["customer_ratio_pct"] = (
        distribution["customer_count"]
        / distribution["customer_count"].sum()
        * 100
    ).round(4)

    return distribution


# ============================================================
# 5. 실행 및 저장
# ============================================================

def main() -> None:
    orders, customers = load_data()

    retention_orders = build_retention_orders(
        orders=orders,
        customers=customers,
    )

    build_summary = create_build_summary(
        source_orders=orders,
        retention_orders=retention_orders,
    )

    order_count_distribution = (
        create_order_count_distribution(
            retention_orders
        )
    )

    retention_orders_path = (
        PROCESSED_DATA_DIR / "retention_orders.csv"
    )

    summary_path = (
        OUTPUT_TABLE_DIR
        / "retention_orders_build_summary.csv"
    )

    distribution_path = (
        OUTPUT_TABLE_DIR
        / "retention_customer_order_count_distribution.csv"
    )

    retention_orders.to_csv(
        retention_orders_path,
        index=False,
        encoding="utf-8-sig",
    )

    build_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    order_count_distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[반복구매 주문 마트 생성 요약]")
    print(build_summary.to_string(index=False))

    print("\n[고객별 주문 횟수 분포]")
    print(order_count_distribution.to_string(index=False))

    print("\n저장 완료:")
    print(f"- {retention_orders_path}")
    print(f"- {summary_path}")
    print(f"- {distribution_path}")


if __name__ == "__main__":
    main()
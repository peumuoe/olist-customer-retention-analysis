from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 주문 데이터 불러오기
# ============================================================

def load_orders() -> pd.DataFrame:
    """주문 상태와 주요 날짜를 불러옵니다."""

    file_path = (
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    orders = pd.read_csv(
        file_path,
        usecols=[
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
        ],
        low_memory=False,
    )

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce",
        )

    if orders["order_purchase_timestamp"].isna().any():
        raise ValueError(
            "order_purchase_timestamp에 결측값이 있습니다."
        )

    orders["purchase_month"] = (
        orders["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    return orders


# ============================================================
# 3. 월별 주문 상태 요약
# ============================================================

def create_monthly_status_summary(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """주문 발생 월별 상태 분포를 생성합니다."""

    monthly_base = (
        orders.groupby(
            "purchase_month",
            as_index=False,
        )
        .agg(
            total_order_count=(
                "order_id",
                "nunique",
            ),
            first_purchase_timestamp=(
                "order_purchase_timestamp",
                "min",
            ),
            last_purchase_timestamp=(
                "order_purchase_timestamp",
                "max",
            ),
        )
    )

    status_counts = pd.crosstab(
        orders["purchase_month"],
        orders["order_status"],
    ).reset_index()

    status_columns = [
        column
        for column in status_counts.columns
        if column != "purchase_month"
    ]

    status_counts = status_counts.rename(
        columns={
            column: f"{column}_count"
            for column in status_columns
        }
    )

    monthly_summary = monthly_base.merge(
        status_counts,
        on="purchase_month",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        column
        for column in monthly_summary.columns
        if column.endswith("_count")
        and column != "total_order_count"
    ]

    for column in count_columns:
        monthly_summary[column] = (
            monthly_summary[column]
            .fillna(0)
            .astype(int)
        )

    if "delivered_count" not in monthly_summary.columns:
        monthly_summary["delivered_count"] = 0

    monthly_summary["non_delivered_count"] = (
        monthly_summary["total_order_count"]
        - monthly_summary["delivered_count"]
    )

    monthly_summary["delivered_rate_pct"] = (
        monthly_summary["delivered_count"]
        / monthly_summary["total_order_count"]
        * 100
    ).round(4)

    for column in count_columns:
        status_name = column.removesuffix("_count")

        monthly_summary[
            f"{status_name}_rate_pct"
        ] = (
            monthly_summary[column]
            / monthly_summary["total_order_count"]
            * 100
        ).round(4)

    monthly_summary = monthly_summary.sort_values(
        "purchase_month"
    ).reset_index(drop=True)

    return monthly_summary


# ============================================================
# 4. 주문 상태별 발생 기간
# ============================================================

def create_status_purchase_range(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """각 주문 상태가 어느 기간에 발생했는지 확인합니다."""

    summary = (
        orders.groupby(
            "order_status",
            as_index=False,
        )
        .agg(
            order_count=(
                "order_id",
                "nunique",
            ),
            minimum_purchase_timestamp=(
                "order_purchase_timestamp",
                "min",
            ),
            maximum_purchase_timestamp=(
                "order_purchase_timestamp",
                "max",
            ),
            missing_customer_delivery_date=(
                "order_delivered_customer_date",
                lambda series: int(
                    series.isna().sum()
                ),
            ),
        )
        .sort_values(
            "order_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# 5. 품질 검증 요약
# ============================================================

def create_audit_summary(
    orders: pd.DataFrame,
    monthly_summary: pd.DataFrame,
) -> pd.DataFrame:
    """월별 집계와 원본 주문 수가 일치하는지 검증합니다."""

    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered")
    ]

    maximum_order_purchase = (
        orders["order_purchase_timestamp"].max()
    )

    maximum_delivered_purchase = (
        delivered_orders[
            "order_purchase_timestamp"
        ].max()
    )

    months_after_last_delivered = (
        orders.loc[
            orders["order_purchase_timestamp"]
            > maximum_delivered_purchase,
            "purchase_month",
        ]
        .nunique()
    )

    records = [
        {
            "check_name": "source_order_rows",
            "value": len(orders),
        },
        {
            "check_name":
                "monthly_total_order_count",
            "value": int(
                monthly_summary[
                    "total_order_count"
                ].sum()
            ),
        },
        {
            "check_name":
                "monthly_delivered_order_count",
            "value": int(
                monthly_summary[
                    "delivered_count"
                ].sum()
            ),
        },
        {
            "check_name":
                "source_delivered_order_count",
            "value": int(
                orders[
                    "order_status"
                ].eq("delivered").sum()
            ),
        },
        {
            "check_name":
                "minimum_order_purchase_timestamp",
            "value": orders[
                "order_purchase_timestamp"
            ].min(),
        },
        {
            "check_name":
                "maximum_order_purchase_timestamp",
            "value": maximum_order_purchase,
        },
        {
            "check_name":
                "maximum_delivered_purchase_timestamp",
            "value": maximum_delivered_purchase,
        },
        {
            "check_name":
                "months_with_orders_after_last_delivered",
            "value": int(months_after_last_delivered),
        },
    ]

    return pd.DataFrame(records)


# ============================================================
# 6. 실행 및 저장
# ============================================================

def main() -> None:
    orders = load_orders()

    monthly_summary = (
        create_monthly_status_summary(orders)
    )

    status_purchase_range = (
        create_status_purchase_range(orders)
    )

    audit_summary = create_audit_summary(
        orders,
        monthly_summary,
    )

    # 마지막 6개월만 별도 저장
    tail_month_summary = (
        monthly_summary
        .tail(6)
        .reset_index(drop=True)
    )

    monthly_output_path = (
        OUTPUT_TABLE_DIR
        / "order_monthly_status_summary.csv"
    )

    tail_output_path = (
        OUTPUT_TABLE_DIR
        / "order_tail_month_status_summary.csv"
    )

    status_range_output_path = (
        OUTPUT_TABLE_DIR
        / "order_status_purchase_range.csv"
    )

    audit_output_path = (
        OUTPUT_TABLE_DIR
        / "order_monthly_status_audit_summary.csv"
    )

    monthly_summary.to_csv(
        monthly_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    tail_month_summary.to_csv(
        tail_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_purchase_range.to_csv(
        status_range_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit_summary.to_csv(
        audit_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[최근 6개월 주문 상태]")
    print(
        tail_month_summary.to_string(
            index=False
        )
    )

    print("\n[주문 상태별 주문 발생 기간]")
    print(
        status_purchase_range.to_string(
            index=False
        )
    )

    print("\n[월별 주문 상태 검증]")
    print(
        audit_summary.to_string(
            index=False
        )
    )

    print("\n저장 완료:")
    print(f"- {monthly_output_path}")
    print(f"- {tail_output_path}")
    print(f"- {status_range_output_path}")
    print(f"- {audit_output_path}")


if __name__ == "__main__":
    main()
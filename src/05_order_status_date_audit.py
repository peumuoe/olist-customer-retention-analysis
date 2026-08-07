from pathlib import Path

import pandas as pd


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 날짜 컬럼
# ============================================================

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ============================================================
# 3. 주문 데이터 로딩 및 날짜 변환
# ============================================================

def load_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본 주문 데이터를 불러오고 날짜형 데이터로 변환합니다."""

    file_path = RAW_DATA_DIR / "olist_orders_dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {file_path}"
        )

    raw_orders = pd.read_csv(
        file_path,
        low_memory=False,
    )

    orders = raw_orders.copy()

    for column in DATE_COLUMNS:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce",
        )

    return raw_orders, orders


# ============================================================
# 4. 주문 상태별 요약
# ============================================================

def create_status_summary(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """주문 상태별 주문 수와 날짜 결측을 확인합니다."""

    summary = (
        orders.groupby(
            "order_status",
            dropna=False,
            as_index=False,
        )
        .agg(
            order_count=("order_id", "nunique"),
            missing_purchase_timestamp=(
                "order_purchase_timestamp",
                lambda series: int(series.isna().sum()),
            ),
            missing_approved_at=(
                "order_approved_at",
                lambda series: int(series.isna().sum()),
            ),
            missing_carrier_date=(
                "order_delivered_carrier_date",
                lambda series: int(series.isna().sum()),
            ),
            missing_customer_delivery_date=(
                "order_delivered_customer_date",
                lambda series: int(series.isna().sum()),
            ),
        )
    )

    total_orders = summary["order_count"].sum()

    summary["order_ratio_pct"] = (
        summary["order_count"]
        / total_orders
        * 100
    ).round(2)

    summary = summary.sort_values(
        "order_count",
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# 5. 날짜 컬럼 품질 및 범위 확인
# ============================================================

def create_date_column_summary(
    raw_orders: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """날짜 컬럼별 결측, 변환 실패, 날짜 범위를 확인합니다."""

    records = []

    for column in DATE_COLUMNS:
        original_not_null = raw_orders[column].notna()
        parsed_not_null = orders[column].notna()

        parse_failure_mask = (
            original_not_null
            & ~parsed_not_null
        )

        records.append(
            {
                "column_name": column,
                "row_count": len(orders),
                "original_missing_count": int(
                    raw_orders[column].isna().sum()
                ),
                "parse_failure_count": int(
                    parse_failure_mask.sum()
                ),
                "parsed_missing_count": int(
                    orders[column].isna().sum()
                ),
                "min_datetime": orders[column].min(),
                "max_datetime": orders[column].max(),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 6. 날짜 논리 검증
# ============================================================

def build_check_record(
    check_name: str,
    eligible_mask: pd.Series,
    issue_mask: pd.Series,
    interpretation: str,
) -> dict[str, object]:
    """날짜 품질 검증 결과를 한 행으로 생성합니다."""

    eligible_rows = int(eligible_mask.sum())
    issue_rows = int(issue_mask.sum())

    issue_rate_pct = (
        round(issue_rows / eligible_rows * 100, 4)
        if eligible_rows > 0
        else 0.0
    )

    return {
        "check_name": check_name,
        "eligible_rows": eligible_rows,
        "issue_rows": issue_rows,
        "issue_rate_pct": issue_rate_pct,
        "interpretation": interpretation,
    }


def create_date_quality_checks(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """주문 상태 및 날짜 순서의 논리적 이상 여부를 확인합니다."""

    purchase = orders["order_purchase_timestamp"]
    approved = orders["order_approved_at"]
    carrier = orders["order_delivered_carrier_date"]
    delivered = orders["order_delivered_customer_date"]
    estimated = orders["order_estimated_delivery_date"]

    delivered_status = (
        orders["order_status"] == "delivered"
    )

    checks = []

    eligible = delivered_status
    issue = (
        delivered_status
        & delivered.isna()
    )

    checks.append(
        build_check_record(
            check_name="delivered_status_missing_delivery_date",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation=(
                "배송 완료 상태이지만 고객 배송 완료일이 없는 주문"
            ),
        )
    )

    eligible = ~delivered_status
    issue = (
        ~delivered_status
        & delivered.notna()
    )

    checks.append(
        build_check_record(
            check_name="non_delivered_status_with_delivery_date",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation=(
                "배송 완료 상태가 아니지만 고객 배송 완료일이 존재하는 주문"
            ),
        )
    )

    eligible = purchase.notna() & approved.notna()
    issue = eligible & (approved < purchase)

    checks.append(
        build_check_record(
            check_name="approved_before_purchase",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation="결제 승인일이 주문일보다 빠른 주문",
        )
    )

    eligible = purchase.notna() & carrier.notna()
    issue = eligible & (carrier < purchase)

    checks.append(
        build_check_record(
            check_name="carrier_before_purchase",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation="물류사 인계일이 주문일보다 빠른 주문",
        )
    )

    eligible = purchase.notna() & delivered.notna()
    issue = eligible & (delivered < purchase)

    checks.append(
        build_check_record(
            check_name="delivery_before_purchase",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation="고객 배송 완료일이 주문일보다 빠른 주문",
        )
    )

    eligible = carrier.notna() & delivered.notna()
    issue = eligible & (delivered < carrier)

    checks.append(
        build_check_record(
            check_name="delivery_before_carrier",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation=(
                "고객 배송 완료일이 물류사 인계일보다 빠른 주문"
            ),
        )
    )

    eligible = purchase.notna() & estimated.notna()
    issue = eligible & (estimated < purchase)

    checks.append(
        build_check_record(
            check_name="estimated_delivery_before_purchase",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation=(
                "예상 배송 완료일이 주문일보다 빠른 주문"
            ),
        )
    )

    eligible = delivered.notna() & estimated.notna()
    issue = eligible & (delivered > estimated)

    checks.append(
        build_check_record(
            check_name="delivered_after_estimated_date",
            eligible_mask=eligible,
            issue_mask=issue,
            interpretation=(
                "예상 배송일보다 실제 배송이 늦은 주문"
            ),
        )
    )

    return pd.DataFrame(checks)


# ============================================================
# 7. 실행 및 결과 저장
# ============================================================

def main() -> None:
    raw_orders, orders = load_orders()

    status_summary = create_status_summary(orders)

    date_column_summary = create_date_column_summary(
        raw_orders,
        orders,
    )

    date_quality_checks = create_date_quality_checks(
        orders
    )

    status_output_path = (
        OUTPUT_DIR / "order_status_summary.csv"
    )

    date_column_output_path = (
        OUTPUT_DIR / "order_date_column_summary.csv"
    )

    quality_output_path = (
        OUTPUT_DIR / "order_date_quality_checks.csv"
    )

    status_summary.to_csv(
        status_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    date_column_summary.to_csv(
        date_column_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    date_quality_checks.to_csv(
        quality_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[주문 상태별 요약]")
    print(status_summary.to_string(index=False))

    print("\n[날짜 컬럼 품질 및 범위]")
    print(date_column_summary.to_string(index=False))

    print("\n[날짜 논리 검증]")
    print(date_quality_checks.to_string(index=False))

    print("\n저장 완료:")
    print(f"- {status_output_path}")
    print(f"- {date_column_output_path}")
    print(f"- {quality_output_path}")


if __name__ == "__main__":
    main()
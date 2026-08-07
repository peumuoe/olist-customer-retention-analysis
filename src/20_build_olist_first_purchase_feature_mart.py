"""Olist 첫 관찰 구매 특성 마트와 30일 세그먼트 재구매율을 생성한다."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
RETENTION_ORDERS = PROCESSED / "retention_orders.csv"
CUSTOMER_RETENTION = PROCESSED / "customer_retention_metrics.csv"
MART_PATH = PROCESSED / "olist_first_purchase_customer_mart.csv"
SEGMENT_PATH = TABLES / "olist_first_purchase_segment_retention.csv"
BUILD_PATH = TABLES / "olist_first_purchase_feature_mart_build_summary.csv"
MIN_SAMPLE = 200
EXPECTED_CUSTOMERS = 93_358
EXPECTED_ELIGIBLE_30D = 86_772
EXPECTED_REPEAT_30D = 1_373


def read_sources() -> dict[str, pd.DataFrame]:
    """필요한 원천과 기존 재구매 마트를 읽는다."""
    files = {
        "retention_orders": RETENTION_ORDERS,
        "retention": CUSTOMER_RETENTION,
        "orders": RAW / "olist_orders_dataset.csv",
        "customers": RAW / "olist_customers_dataset.csv",
        "items": RAW / "olist_order_items_dataset.csv",
        "products": RAW / "olist_products_dataset.csv",
        "translation": RAW / "product_category_name_translation.csv",
        "payments": RAW / "olist_order_payments_dataset.csv",
        "reviews": RAW / "olist_order_reviews_dataset.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"필수 파일 누락: {missing}")
    return {name: pd.read_csv(path) for name, path in files.items()}


def aggregate_items(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """상품행을 주문 1행으로 집계하고 대표 카테고리를 정한다."""
    items = data["items"].merge(
        data["products"][["product_id", "product_category_name"]],
        on="product_id", how="left", validate="many_to_one",
    ).merge(
        data["translation"], on="product_category_name", how="left",
        validate="many_to_one",
    )
    items["category"] = items["product_category_name_english"].fillna(
        items["product_category_name"]
    ).fillna("unknown")
    totals = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "size"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        category_count=("category", "nunique"),
        item_price_sum=("price", "sum"),
        freight_value_sum=("freight_value", "sum"),
    )
    counts = items.groupby(["order_id", "category"], as_index=False).size()
    primary = counts.sort_values(
        ["order_id", "size", "category"], ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("order_id").rename(columns={"category": "primary_category"})
    return totals.merge(
        primary[["order_id", "primary_category"]], on="order_id",
        how="left", validate="one_to_one",
    )


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """결제행을 주문 1행으로 집계한다."""
    totals = payments.groupby("order_id", as_index=False).agg(
        payment_value_sum=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_type_count=("payment_type", "nunique"),
    )
    values = payments.groupby(["order_id", "payment_type"], as_index=False).agg(
        payment_type_value=("payment_value", "sum")
    )
    primary = values.sort_values(
        ["order_id", "payment_type_value", "payment_type"],
        ascending=[True, False, True], kind="mergesort",
    ).drop_duplicates("order_id").rename(columns={"payment_type": "primary_payment_type"})
    return totals.merge(
        primary[["order_id", "primary_payment_type"]], on="order_id",
        how="left", validate="one_to_one",
    )


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """복수 리뷰를 주문별 기술통계로 집계한다."""
    return reviews.groupby("order_id", as_index=False).agg(
        review_count=("review_id", "size"),
        review_score_mean=("review_score", "mean"),
        review_score_min=("review_score", "min"),
        review_comment_present=("review_comment_message", lambda x: x.notna().any()),
    )


def add_groups(mart: pd.DataFrame) -> pd.DataFrame:
    """기초 프로파일링에 사용할 해석 가능한 구간을 만든다."""
    mart = mart.copy()
    mart["order_value_band"] = pd.qcut(
        mart["payment_value_sum"], 4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop",
    ).astype("string").fillna("unknown")
    mart["item_count_group"] = pd.cut(
        mart["item_count"], bins=[-np.inf, 1, 2, 4, np.inf],
        labels=["1", "2", "3-4", "5+"],
    ).astype("string").fillna("unknown")
    mart["seller_count_group"] = np.where(
        mart["seller_count"].eq(1), "single_seller", "multiple_sellers"
    )
    mart["installment_group"] = pd.cut(
        mart["payment_installments_max"], bins=[-np.inf, 1, 3, 6, np.inf],
        labels=["1", "2-3", "4-6", "7+"],
    ).astype("string").fillna("unknown")
    mart["delivery_experience_group"] = pd.cut(
        mart["delivery_delay_days"],
        bins=[-np.inf, 0, 3, 7, np.inf],
        labels=["on_time_or_early", "late_1_3d", "late_4_7d", "late_8d_plus"],
    ).astype("string").fillna("unknown")
    mart["review_score_group"] = pd.cut(
        mart["review_score_mean"], bins=[0, 1, 2, 3, 4, 5],
        labels=["1", "2", "3", "4", "5"], include_lowest=True,
    ).astype("string").fillna("no_review")
    return mart


def build_mart(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """고객별 첫 관찰 구매 1행 마트를 만든다."""
    first = data["retention_orders"].loc[
        data["retention_orders"]["customer_order_number"].eq(1),
        ["order_id", "customer_id", "customer_unique_id", "order_purchase_timestamp"],
    ].copy()
    orders = data["orders"][[
        "order_id", "order_delivered_customer_date", "order_estimated_delivery_date",
    ]].copy()
    for column in ["order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")
    first["order_purchase_timestamp"] = pd.to_datetime(
        first["order_purchase_timestamp"], errors="coerce"
    )
    customers = data["customers"][[
        "customer_id", "customer_state", "customer_city",
    ]]
    mart = first.merge(orders, on="order_id", how="left", validate="one_to_one")
    mart = mart.merge(customers, on="customer_id", how="left", validate="many_to_one")
    mart = mart.merge(aggregate_items(data), on="order_id", how="left", validate="one_to_one")
    mart = mart.merge(aggregate_payments(data["payments"]), on="order_id", how="left", validate="one_to_one")
    mart = mart.merge(aggregate_reviews(data["reviews"]), on="order_id", how="left", validate="one_to_one")
    retention_columns = [
        "customer_unique_id", "first_purchase_timestamp", "eligible_30d", "repeat_within_30d",
    ]
    retention = data["retention"][retention_columns].copy()
    retention["first_purchase_timestamp"] = pd.to_datetime(
        retention["first_purchase_timestamp"], errors="coerce"
    )
    mart = mart.merge(retention, on="customer_unique_id", how="left", validate="one_to_one")
    mart["first_timestamp_match_flag"] = (
        mart["order_purchase_timestamp"].eq(mart["first_purchase_timestamp"])
    )
    mart["delivery_days"] = (
        mart["order_delivered_customer_date"] - mart["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    mart["delivery_delay_days"] = (
        mart["order_delivered_customer_date"] - mart["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    mart["freight_share"] = (
        mart["freight_value_sum"]
        / (mart["item_price_sum"] + mart["freight_value_sum"])
    ).where((mart["item_price_sum"] + mart["freight_value_sum"]).gt(0))
    mart["review_count"] = mart["review_count"].fillna(0).astype("int64")
    mart["review_present_flag"] = mart["review_count"].gt(0)
    mart["review_comment_present"] = mart["review_comment_present"].fillna(False)
    return add_groups(mart)


def build_segments(mart: pd.DataFrame) -> pd.DataFrame:
    """30일 적격 고객의 단변량 세그먼트 재구매율을 계산한다."""
    eligible = mart.loc[mart["eligible_30d"].eq(True)].copy()
    eligible["repeat_within_30d"] = eligible["repeat_within_30d"].astype(bool)
    overall_rate = eligible["repeat_within_30d"].mean() * 100
    dimensions = {
        "primary_category": "order_time",
        "customer_state": "order_time",
        "order_value_band": "order_time",
        "item_count_group": "order_time",
        "seller_count_group": "order_time",
        "primary_payment_type": "order_time",
        "installment_group": "order_time",
        "delivery_experience_group": "experience_complete",
        "review_score_group": "experience_complete",
    }
    rows = []
    total = len(eligible)
    for dimension, availability in dimensions.items():
        grouped = eligible.assign(
            segment_value=eligible[dimension].astype("string").fillna("unknown")
        ).groupby("segment_value", dropna=False)["repeat_within_30d"].agg(
            eligible_customer_count="size", repeat_customer_count="sum",
        ).reset_index()
        grouped["segment_dimension"] = dimension
        grouped["feature_availability"] = availability
        grouped["retention_rate_pct"] = (
            grouped["repeat_customer_count"] / grouped["eligible_customer_count"] * 100
        )
        # 낮은 사건률에서도 경계가 0~100%를 벗어나지 않는 Wilson 95% 구간.
        z = 1.959963984540054
        n = grouped["eligible_customer_count"].astype(float)
        p = grouped["repeat_customer_count"] / n
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        grouped["retention_rate_ci95_low_pct"] = (center - margin) * 100
        grouped["retention_rate_ci95_high_pct"] = (center + margin) * 100
        grouped["gap_vs_overall_pct_point"] = grouped["retention_rate_pct"] - overall_rate
        grouped["customer_share_pct"] = grouped["eligible_customer_count"] / total * 100
        grouped["meets_min_sample_200"] = grouped["eligible_customer_count"].ge(MIN_SAMPLE)
        grouped["descriptive_repeat_shortfall_vs_overall"] = (
            grouped["eligible_customer_count"] * overall_rate / 100
            - grouped["repeat_customer_count"]
        ).clip(lower=0)
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    numeric = [
        "retention_rate_pct", "retention_rate_ci95_low_pct",
        "retention_rate_ci95_high_pct", "gap_vs_overall_pct_point", "customer_share_pct",
        "descriptive_repeat_shortfall_vs_overall",
    ]
    result[numeric] = result[numeric].round(6)
    return result[[
        "segment_dimension", "segment_value", "feature_availability",
        "eligible_customer_count", "repeat_customer_count", "retention_rate_pct",
        "retention_rate_ci95_low_pct", "retention_rate_ci95_high_pct",
        "gap_vs_overall_pct_point", "customer_share_pct", "meets_min_sample_200",
        "descriptive_repeat_shortfall_vs_overall",
    ]].sort_values(
        ["segment_dimension", "eligible_customer_count"], ascending=[True, False]
    )


def validate(mart: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """고객 Grain, 기존 KPI 재현, 시점 규칙을 검증한다."""
    eligible = mart["eligible_30d"].eq(True)
    checks = {
        "customer_rows": (len(mart), EXPECTED_CUSTOMERS),
        "unique_customers": (mart["customer_unique_id"].nunique(), EXPECTED_CUSTOMERS),
        "duplicate_customers": (int(mart["customer_unique_id"].duplicated().sum()), 0),
        "duplicate_first_order_ids": (int(mart["order_id"].duplicated().sum()), 0),
        "first_timestamp_mismatches": (int((~mart["first_timestamp_match_flag"]).sum()), 0),
        "eligible_30d": (int(eligible.sum()), EXPECTED_ELIGIBLE_30D),
        "repeat_30d": (int(mart.loc[eligible, "repeat_within_30d"].sum()), EXPECTED_REPEAT_30D),
        "missing_item_aggregation": (int(mart["item_count"].isna().sum()), 0),
        # 결제 원천에 존재하는 행은 모두 주문과 연결되지만, 역방향으로는
        # 배송 완료 첫 주문 1건에 결제행이 없다. 임의의 0원으로 대체하지 않는다.
        "missing_payment_aggregation": (int(mart["payment_value_sum"].isna().sum()), 1),
        "segment_dimensions": (segments["segment_dimension"].nunique(), 9),
        "invalid_feature_availability": (
            int((~segments["feature_availability"].isin(
                ["order_time", "experience_complete"]
            )).sum()),
            0,
        ),
    }
    # 각 차원은 동일한 30일 적격 고객 수를 정확히 한 번씩 포함해야 한다.
    dimension_totals = segments.groupby("segment_dimension")["eligible_customer_count"].sum()
    checks["all_dimension_denominators_reconcile"] = (
        bool(dimension_totals.eq(EXPECTED_ELIGIBLE_30D).all()), True
    )
    failures = {key: value for key, value in checks.items() if value[0] != value[1]}
    if failures:
        raise AssertionError(f"첫 구매 특성 마트 검증 실패: {failures}")
    return pd.DataFrame([
        {"check_name": key, "actual": actual, "expected": expected, "status": "PASS"}
        for key, (actual, expected) in checks.items()
    ])


def save(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    print(f"생성: {path.relative_to(ROOT)} ({len(frame):,}행)")


def main() -> None:
    data = read_sources()
    mart = build_mart(data)
    segments = build_segments(mart)
    checks = validate(mart, segments)
    save(mart, MART_PATH)
    save(segments, SEGMENT_PATH)
    save(checks, BUILD_PATH)


if __name__ == "__main__":
    main()

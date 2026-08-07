"""REES46 purchase 상품 로그를 고객·세션 단위 구매 이벤트 마트로 집계한다."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "external" / "rees46" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "external" / "rees46" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
INPUT_FILES = [RAW_DIR / "2019-Oct.csv", RAW_DIR / "2019-Nov.csv"]
MART_PATH = PROCESSED_DIR / "rees46_purchase_events.csv"
BUILD_SUMMARY_PATH = OUTPUT_DIR / "rees46_purchase_event_mart_summary.csv"
FLAG_SUMMARY_PATH = OUTPUT_DIR / "rees46_purchase_event_quality_flag_summary.csv"
DISTRIBUTION_SUMMARY_PATH = OUTPUT_DIR / "rees46_purchase_event_distribution_summary.csv"
CHUNK_SIZE = 1_000_000

USE_COLUMNS = [
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
]
EXPECTED_PURCHASE_ROWS = 1_659_788
EXPECTED_PURCHASE_EVENTS = 1_402_758
EXPECTED_PURCHASE_USERS = 697_470
EXPECTED_CROSS_MONTH_EVENTS = 16
EXPECTED_OVER_24H_EVENTS = 68


def check_inputs() -> None:
    """필수 원본 파일과 컬럼을 확인한다."""
    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in USE_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path.name} 필수 컬럼 누락: {missing}")


def collect_purchase_rows() -> pd.DataFrame:
    """두 원본에서 purchase 행만 청크 단위로 선별한다."""
    frames = []
    dtypes = {
        "event_type": "string", "product_id": "string",
        "category_id": "string", "category_code": "string",
        "brand": "string", "user_id": "string", "user_session": "string",
    }

    for path in INPUT_FILES:
        processed_rows = 0
        for chunk in pd.read_csv(
            path,
            usecols=USE_COLUMNS,
            chunksize=CHUNK_SIZE,
            dtype=dtypes,
            low_memory=False,
        ):
            processed_rows += len(chunk)
            purchase = chunk.loc[chunk["event_type"].eq("purchase")].copy()
            if purchase.empty:
                continue
            purchase["source_file"] = path.name
            frames.append(purchase)
            print(f"{path.name}: 원본 {processed_rows:,}행까지 처리")

    if not frames:
        raise ValueError("purchase 행을 찾지 못했습니다.")
    purchases = pd.concat(frames, ignore_index=True)
    if len(purchases) != EXPECTED_PURCHASE_ROWS:
        raise AssertionError(
            f"purchase 행 수가 기존 감사와 다릅니다: {len(purchases):,}"
        )
    return purchases


def prepare_purchase_rows(purchases: pd.DataFrame) -> pd.DataFrame:
    """날짜·가격을 변환하고 필수 품질 조건을 검증한다."""
    purchases["purchase_time"] = pd.to_datetime(
        purchases["event_time"], errors="coerce", utc=True
    )
    purchases["recorded_item_price"] = pd.to_numeric(
        purchases["price"], errors="coerce"
    )

    required = ["user_id", "user_session", "purchase_time", "recorded_item_price"]
    nulls = purchases[required].isna().sum()
    if int(nulls.sum()) != 0:
        raise AssertionError(f"purchase 필수값 결측이 발견됐습니다: {nulls.to_dict()}")
    if int(purchases["recorded_item_price"].lt(0).sum()) != 0:
        raise AssertionError("purchase에서 음수 가격이 발견됐습니다.")

    purchases["category_code_missing"] = purchases["category_code"].isna()
    purchases["brand_missing"] = purchases["brand"].isna()
    return purchases


def build_mart(purchases: pd.DataFrame) -> pd.DataFrame:
    """고객·세션당 1행의 구매 이벤트 마트를 생성한다."""
    grouped = purchases.groupby(
        ["user_id", "user_session"], sort=False, observed=True
    )
    mart = grouped.agg(
        purchase_event_start=("purchase_time", "min"),
        purchase_event_end=("purchase_time", "max"),
        purchase_row_count=("product_id", "size"),
        distinct_timestamp_count=("purchase_time", "nunique"),
        distinct_product_count=("product_id", "nunique"),
        distinct_category_id_count=("category_id", "nunique"),
        distinct_category_code_count=("category_code", "nunique"),
        distinct_brand_count=("brand", "nunique"),
        recorded_item_price_sum=("recorded_item_price", "sum"),
        recorded_item_price_mean=("recorded_item_price", "mean"),
        source_file_count=("source_file", "nunique"),
        contains_missing_category_code=("category_code_missing", "max"),
        contains_missing_brand=("brand_missing", "max"),
    ).reset_index()

    mart["purchase_event_duration_seconds"] = (
        mart["purchase_event_end"] - mart["purchase_event_start"]
    ).dt.total_seconds().astype("int64")
    mart["cross_month_file_flag"] = mart["source_file_count"].gt(1)
    mart["session_over_30m_flag"] = mart["purchase_event_duration_seconds"].gt(1_800)
    mart["session_over_24h_flag"] = mart["purchase_event_duration_seconds"].gt(86_400)
    mart["purchase_event_key"] = (
        mart["user_id"].astype("string")
        + "|"
        + mart["user_session"].astype("string")
    )

    columns = [
        "purchase_event_key", "user_id", "user_session",
        "purchase_event_start", "purchase_event_end",
        "purchase_event_duration_seconds", "purchase_row_count",
        "distinct_timestamp_count", "distinct_product_count",
        "distinct_category_id_count", "distinct_category_code_count",
        "distinct_brand_count", "recorded_item_price_sum",
        "recorded_item_price_mean", "source_file_count",
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag", "contains_missing_category_code",
        "contains_missing_brand",
    ]
    return mart[columns]


def validate_mart(mart: pd.DataFrame) -> dict[str, int | float]:
    """마트의 행 고유성, 원본 보존, 날짜·플래그를 검증한다."""
    metrics = {
        "mart_rows": len(mart),
        "unique_purchase_event_keys": int(mart["purchase_event_key"].nunique()),
        "duplicate_purchase_event_key_rows": int(
            mart["purchase_event_key"].duplicated(keep=False).sum()
        ),
        "purchase_users": int(mart["user_id"].nunique()),
        "purchase_row_count_sum": int(mart["purchase_row_count"].sum()),
        "required_null_rows": int(
            mart[[
                "purchase_event_key", "user_id", "user_session",
                "purchase_event_start", "purchase_event_end",
            ]].isna().any(axis=1).sum()
        ),
        "negative_duration_rows": int(
            mart["purchase_event_duration_seconds"].lt(0).sum()
        ),
        "cross_month_events": int(mart["cross_month_file_flag"].sum()),
        "over_30m_events": int(mart["session_over_30m_flag"].sum()),
        "over_24h_events": int(mart["session_over_24h_flag"].sum()),
        "zero_recorded_price_sum_events": int(
            mart["recorded_item_price_sum"].eq(0).sum()
        ),
        "negative_recorded_price_sum_events": int(
            mart["recorded_item_price_sum"].lt(0).sum()
        ),
        "recorded_item_price_sum_total": round(
            float(mart["recorded_item_price_sum"].sum()), 2
        ),
    }

    expected = {
        "mart_rows": EXPECTED_PURCHASE_EVENTS,
        "unique_purchase_event_keys": EXPECTED_PURCHASE_EVENTS,
        "duplicate_purchase_event_key_rows": 0,
        "purchase_users": EXPECTED_PURCHASE_USERS,
        "purchase_row_count_sum": EXPECTED_PURCHASE_ROWS,
        "required_null_rows": 0,
        "negative_duration_rows": 0,
        "cross_month_events": EXPECTED_CROSS_MONTH_EVENTS,
        "over_24h_events": EXPECTED_OVER_24H_EVENTS,
        "zero_recorded_price_sum_events": 0,
        "negative_recorded_price_sum_events": 0,
    }
    failures = {
        key: (metrics[key], value)
        for key, value in expected.items()
        if metrics[key] != value
    }
    if failures:
        raise AssertionError(f"구매 이벤트 마트 검증 실패: {failures}")
    return metrics


def create_summaries(
    mart: pd.DataFrame,
    metrics: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """마트 생성 요약과 품질 플래그 요약을 만든다."""
    summary = pd.DataFrame([
        {"check_name": key, "value": value, "status": "PASS"}
        for key, value in metrics.items()
    ])
    flag_rows = []
    for column in [
        "cross_month_file_flag", "session_over_30m_flag",
        "session_over_24h_flag", "contains_missing_category_code",
        "contains_missing_brand",
    ]:
        count = int(mart[column].sum())
        flag_rows.append({
            "flag_name": column,
            "event_count": count,
            "event_rate_pct": round(count / len(mart) * 100, 6),
            "treatment": "기본 마트에 유지; 후속 민감도·세그먼트 분석에서 구분",
        })
    multi_row_count = int(mart["purchase_row_count"].gt(1).sum())
    multi_product_count = int(mart["distinct_product_count"].gt(1).sum())
    price = mart["recorded_item_price_sum"]
    distribution = pd.DataFrame([
        {"metric": "single_purchase_row_events", "value": int(mart["purchase_row_count"].eq(1).sum())},
        {"metric": "multi_purchase_row_events", "value": multi_row_count},
        {"metric": "multi_purchase_row_event_rate_pct", "value": round(multi_row_count / len(mart) * 100, 6)},
        {"metric": "single_product_events", "value": int(mart["distinct_product_count"].eq(1).sum())},
        {"metric": "multi_product_events", "value": multi_product_count},
        {"metric": "multi_product_event_rate_pct", "value": round(multi_product_count / len(mart) * 100, 6)},
        {"metric": "recorded_item_price_sum_min", "value": round(float(price.min()), 2)},
        {"metric": "recorded_item_price_sum_p25", "value": round(float(price.quantile(0.25)), 2)},
        {"metric": "recorded_item_price_sum_median", "value": round(float(price.median()), 2)},
        {"metric": "recorded_item_price_sum_p75", "value": round(float(price.quantile(0.75)), 2)},
        {"metric": "recorded_item_price_sum_p95", "value": round(float(price.quantile(0.95)), 2)},
        {"metric": "recorded_item_price_sum_max", "value": round(float(price.max()), 2)},
        {"metric": "recorded_item_price_sum_mean", "value": round(float(price.mean()), 2)},
    ])
    return summary, pd.DataFrame(flag_rows), distribution


def save_outputs(
    mart: pd.DataFrame,
    summary: pd.DataFrame,
    flag_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
) -> None:
    """검증된 결과를 임시 파일에 쓴 뒤 최종 경로로 교체한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (MART_PATH, mart),
        (BUILD_SUMMARY_PATH, summary),
        (FLAG_SUMMARY_PATH, flag_summary),
        (DISTRIBUTION_SUMMARY_PATH, distribution_summary),
    ]
    for target, frame in outputs:
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        print(f"생성: {target.relative_to(PROJECT_ROOT)} ({len(frame):,}행)")


def main() -> None:
    """REES46 구매 이벤트 마트를 생성하고 품질을 검증한다."""
    check_inputs()
    purchases = prepare_purchase_rows(collect_purchase_rows())
    mart = build_mart(purchases)
    metrics = validate_mart(mart)
    summary, flag_summary, distribution_summary = create_summaries(mart, metrics)
    save_outputs(mart, summary, flag_summary, distribution_summary)
    print("REES46 구매 이벤트 마트 생성과 검증을 완료했습니다.")


if __name__ == "__main__":
    main()

"""REES46 2019년 10월·11월 원본 이벤트 데이터의 품질을 감사한다."""

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "external" / "rees46" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
INPUT_FILES = [RAW_DIR / "2019-Oct.csv", RAW_DIR / "2019-Nov.csv"]
CHUNK_SIZE = 1_000_000
REQUIRED_COLUMNS = [
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
]
NULL_COLUMNS = [
    "user_session", "user_id", "product_id", "category_id",
    "category_code", "brand", "price",
]


def check_inputs() -> None:
    """입력 파일과 헤더가 기대한 구조인지 확인한다."""
    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path.name} 필수 컬럼 누락: {missing}")


def stable_row_hash(frame: pd.DataFrame) -> pd.Series:
    """완전 중복 하한 검사용 64비트 행 해시를 만든다."""
    return pd.util.hash_pandas_object(
        frame[REQUIRED_COLUMNS], index=False, categorize=True
    )


def audit_file(path: Path) -> dict:
    """한 파일을 청크 단위로 전수 읽고 감사 통계를 반환한다."""
    counters = {
        "rows": 0,
        "date_parse_failures": 0,
        "price_parse_failures": 0,
        "zero_price_rows": 0,
        "negative_price_rows": 0,
        "chunk_internal_duplicate_rows": 0,
        "chunk_boundary_duplicate_rows": 0,
    }
    null_counts = Counter()
    event_counts = Counter()
    purchase_null_counts = Counter()
    unique_values = {
        "user_id": set(), "product_id": set(), "category_id": set(),
        "category_code": set(), "brand": set(),
    }
    purchase_unique = {"user_id": set(), "user_session": set()}
    purchase_products = set()
    purchase_categories = set()
    purchase_brands = set()
    min_time = None
    max_time = None
    previous_last_hash = None
    chunk_count = 0

    dtypes = {
        "event_type": "string", "product_id": "string",
        "category_id": "string", "category_code": "string",
        "brand": "string", "user_id": "string", "user_session": "string",
    }
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, dtype=dtypes, low_memory=False):
        chunk_count += 1
        counters["rows"] += len(chunk)
        parsed_time = pd.to_datetime(chunk["event_time"], errors="coerce", utc=True)
        parsed_price = pd.to_numeric(chunk["price"], errors="coerce")
        counters["date_parse_failures"] += int(parsed_time.isna().sum())
        counters["price_parse_failures"] += int(
            parsed_price.isna().sum() - chunk["price"].isna().sum()
        )
        counters["zero_price_rows"] += int(parsed_price.eq(0).sum())
        counters["negative_price_rows"] += int(parsed_price.lt(0).sum())
        valid_time = parsed_time.dropna()
        if not valid_time.empty:
            current_min, current_max = valid_time.min(), valid_time.max()
            min_time = current_min if min_time is None else min(min_time, current_min)
            max_time = current_max if max_time is None else max(max_time, current_max)

        event_counts.update(chunk["event_type"].dropna().tolist())
        for column in NULL_COLUMNS:
            null_counts[column] += int(chunk[column].isna().sum())
        for column, values in unique_values.items():
            values.update(chunk[column].dropna().tolist())

        purchase_mask = chunk["event_type"].eq("purchase")
        purchase = chunk.loc[purchase_mask]
        for column in NULL_COLUMNS:
            purchase_null_counts[column] += int(purchase[column].isna().sum())
        purchase_unique["user_id"].update(purchase["user_id"].dropna().tolist())
        purchase_unique["user_session"].update(purchase["user_session"].dropna().tolist())
        purchase_products.update(purchase["product_id"].dropna().tolist())
        purchase_categories.update(purchase["category_code"].dropna().tolist())
        purchase_brands.update(purchase["brand"].dropna().tolist())

        hashes = stable_row_hash(chunk)
        counters["chunk_internal_duplicate_rows"] += int(hashes.duplicated().sum())
        if previous_last_hash is not None and len(hashes):
            counters["chunk_boundary_duplicate_rows"] += int(hashes.iloc[0] == previous_last_hash)
        if len(hashes):
            previous_last_hash = int(hashes.iloc[-1])

        print(f"{path.name}: {counters['rows']:,}행 처리 완료")

    return {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "column_count": len(REQUIRED_COLUMNS),
        "chunk_count": chunk_count,
        **counters,
        "min_event_time": min_time,
        "max_event_time": max_time,
        "event_counts": event_counts,
        "null_counts": null_counts,
        "purchase_null_counts": purchase_null_counts,
        "unique_values": unique_values,
        "purchase_rows": event_counts["purchase"],
        "purchase_unique_values": purchase_unique,
        "purchase_product_values": purchase_products,
        "purchase_category_values": purchase_categories,
        "purchase_brand_values": purchase_brands,
        "purchase_users": len(purchase_unique["user_id"]),
        "purchase_sessions": len(purchase_unique["user_session"]),
        "purchase_products": len(purchase_products),
        "purchase_categories": len(purchase_categories),
        "purchase_brands": len(purchase_brands),
    }


def make_outputs(results: list[dict]) -> dict[str, pd.DataFrame]:
    """감사 결과를 재현 가능한 표 형태로 변환한다."""
    total_rows = sum(result["rows"] for result in results)
    all_events = sum((result["event_counts"] for result in results), Counter())
    global_uniques = {}
    for column in results[0]["unique_values"]:
        global_uniques[column] = set().union(
            *(result["unique_values"][column] for result in results)
        )

    file_rows = []
    for result in results:
        file_rows.append({
            "file_name": result["file_name"],
            "file_size_bytes": result["file_size_bytes"],
            "column_count": result["column_count"],
            "chunk_count": result["chunk_count"],
            "row_count": result["rows"],
            "date_parse_failures": result["date_parse_failures"],
            "price_parse_failures": result["price_parse_failures"],
            "confirmed_duplicate_rows_lower_bound": (
                result["chunk_internal_duplicate_rows"]
                + result["chunk_boundary_duplicate_rows"]
            ),
            "duplicate_check_scope": "청크 내부 전수 + 인접 청크 경계; 비인접 청크 간은 미검증",
        })

    event_rows = []
    for result in results:
        for event_type, count in sorted(result["event_counts"].items()):
            event_rows.append({
                "file_name": result["file_name"],
                "event_type": event_type,
                "row_count": count,
                "row_rate_pct": round(count / result["rows"] * 100, 6),
            })
    for event_type, count in sorted(all_events.items()):
        event_rows.append({
            "file_name": "ALL",
            "event_type": event_type,
            "row_count": count,
            "row_rate_pct": round(count / total_rows * 100, 6),
        })

    purchase_rows = []
    for result in results:
        row = {
            "file_name": result["file_name"],
            "purchase_rows": result["purchase_rows"],
            "purchase_users": result["purchase_users"],
            "purchase_sessions": result["purchase_sessions"],
            "purchase_products": result["purchase_products"],
            "purchase_categories_non_null": result["purchase_categories"],
            "purchase_brands_non_null": result["purchase_brands"],
            "purchase_user_session_nulls": result["purchase_null_counts"]["user_session"],
            "purchase_user_id_nulls": result["purchase_null_counts"]["user_id"],
        }
        for column in NULL_COLUMNS:
            row[f"purchase_{column}_nulls"] = result["purchase_null_counts"][column]
        purchase_rows.append(row)
    overall_purchase_row = {
        "file_name": "ALL",
        "purchase_rows": sum(r["purchase_rows"] for r in results),
        "purchase_users": len(set().union(*(r["purchase_unique_values"]["user_id"] for r in results))),
        "purchase_sessions": len(set().union(*(r["purchase_unique_values"]["user_session"] for r in results))),
        "purchase_products": len(set().union(*(r["purchase_product_values"] for r in results))),
        "purchase_categories_non_null": len(set().union(*(r["purchase_category_values"] for r in results))),
        "purchase_brands_non_null": len(set().union(*(r["purchase_brand_values"] for r in results))),
        "purchase_user_session_nulls": sum(r["purchase_null_counts"]["user_session"] for r in results),
        "purchase_user_id_nulls": sum(r["purchase_null_counts"]["user_id"] for r in results),
    }
    for column in NULL_COLUMNS:
        overall_purchase_row[f"purchase_{column}_nulls"] = sum(
            r["purchase_null_counts"][column] for r in results
        )
    purchase_rows.append(overall_purchase_row)

    column_rows = []
    for result in results:
        for column in NULL_COLUMNS:
            column_rows.append({
                "file_name": result["file_name"],
                "column_name": column,
                "null_count": result["null_counts"][column],
                "null_rate_pct": round(result["null_counts"][column] / result["rows"] * 100, 6),
                "unique_count": (
                    len(result["unique_values"][column])
                    if column in result["unique_values"] else pd.NA
                ),
            })
    for column in NULL_COLUMNS:
        column_rows.append({
            "file_name": "ALL",
            "column_name": column,
            "null_count": sum(r["null_counts"][column] for r in results),
            "null_rate_pct": round(
                sum(r["null_counts"][column] for r in results) / total_rows * 100, 6
            ),
            "unique_count": (
                len(global_uniques[column]) if column in global_uniques else pd.NA
            ),
        })

    date_rows = [{
        "file_name": result["file_name"],
        "min_event_time": result["min_event_time"],
        "max_event_time": result["max_event_time"],
        "date_parse_failures": result["date_parse_failures"],
        "zero_price_rows": result["zero_price_rows"],
        "negative_price_rows": result["negative_price_rows"],
        "supports_30_day_window": (
            result["max_event_time"] - result["min_event_time"]
        ).total_seconds() >= 30 * 86400,
        "supports_60_day_window": False,
        "supports_90_day_window": False,
    } for result in results]
    overall_span = max(r["max_event_time"] for r in results) - min(r["min_event_time"] for r in results)
    date_rows.append({
        "file_name": "ALL",
        "min_event_time": min(r["min_event_time"] for r in results),
        "max_event_time": max(r["max_event_time"] for r in results),
        "date_parse_failures": sum(r["date_parse_failures"] for r in results),
        "zero_price_rows": sum(r["zero_price_rows"] for r in results),
        "negative_price_rows": sum(r["negative_price_rows"] for r in results),
        "supports_30_day_window": overall_span.total_seconds() >= 30 * 86400,
        "supports_60_day_window": overall_span.total_seconds() >= 60 * 86400,
        "supports_90_day_window": overall_span.total_seconds() >= 90 * 86400,
    })

    return {
        "rees46_file_audit_summary.csv": pd.DataFrame(file_rows),
        "rees46_event_type_summary.csv": pd.DataFrame(event_rows),
        "rees46_purchase_data_audit_summary.csv": pd.DataFrame(purchase_rows),
        "rees46_column_summary.csv": pd.DataFrame(column_rows),
        "rees46_date_range_summary.csv": pd.DataFrame(date_rows),
    }


def validate(results: list[dict], outputs: dict[str, pd.DataFrame]) -> None:
    """산출표 간 핵심 합계와 논리 조건을 검증한다."""
    total_rows = sum(result["rows"] for result in results)
    event_total = int(
        outputs["rees46_event_type_summary.csv"]
        .loc[lambda x: x["file_name"].eq("ALL"), "row_count"].sum()
    )
    purchase_total = sum(result["purchase_rows"] for result in results)
    purchase_event_total = int(
        outputs["rees46_event_type_summary.csv"]
        .loc[lambda x: x["file_name"].eq("ALL") & x["event_type"].eq("purchase"), "row_count"].sum()
    )
    if event_total != total_rows:
        raise AssertionError("event_type 합계와 전체 행 수가 다릅니다.")
    if purchase_total != purchase_event_total:
        raise AssertionError("purchase 집계가 event_type 집계와 다릅니다.")
    if any(r["purchase_users"] > len(r["unique_values"]["user_id"]) for r in results):
        raise AssertionError("구매 고객 수가 전체 고객 수보다 큽니다.")
    if any(r["min_event_time"] > r["max_event_time"] for r in results):
        raise AssertionError("최소 날짜가 최대 날짜보다 큽니다.")


def main() -> None:
    """입력을 감사하고 5개 결과표를 원자적으로 저장한다."""
    check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [audit_file(path) for path in INPUT_FILES]
    outputs = make_outputs(results)
    validate(results, outputs)
    for name, frame in outputs.items():
        target = OUTPUT_DIR / name
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        print(f"생성: {target.relative_to(PROJECT_ROOT)} ({len(frame):,}행)")
    print("REES46 원본 데이터 감사와 산출물 검증을 완료했습니다.")


if __name__ == "__main__":
    main()

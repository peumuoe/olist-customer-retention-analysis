"""REES46 purchase 행의 구매 이벤트 대체키 후보를 전수 감사한다."""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "external" / "rees46" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
INPUT_FILES = [RAW_DIR / "2019-Oct.csv", RAW_DIR / "2019-Nov.csv"]
CHUNK_SIZE = 1_000_000
USE_COLUMNS = [
    "event_time", "event_type", "product_id", "category_id",
    "category_code", "brand", "price", "user_id", "user_session",
]


def check_inputs() -> None:
    """필수 파일과 컬럼을 확인한다."""
    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in USE_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path.name} 필수 컬럼 누락: {missing}")


def collect_purchase_rows() -> dict:
    """purchase 행만 전수 선별하여 대체키 후보와 세션 품질을 집계한다."""
    file_counts = Counter()
    users = set()
    sessions = set()
    user_sessions = set()
    user_times = set()
    user_session_times = set()
    exact_hashes = set()
    session_users = defaultdict(set)
    session_times = defaultdict(set)
    session_products = defaultdict(set)
    session_row_counts = Counter()
    session_files = defaultdict(set)
    zero_price_sessions = set()
    negative_price_rows = 0
    price_parse_failures = 0
    date_parse_failures = 0
    purchase_nulls = Counter()
    total_purchase_rows = 0
    confirmed_exact_duplicate_excess_rows = 0

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

            total_purchase_rows += len(purchase)
            file_counts[path.name] += len(purchase)
            parsed_time = pd.to_datetime(purchase["event_time"], errors="coerce", utc=True)
            parsed_price = pd.to_numeric(purchase["price"], errors="coerce")
            date_parse_failures += int(parsed_time.isna().sum())
            price_parse_failures += int(
                parsed_price.isna().sum() - purchase["price"].isna().sum()
            )
            negative_price_rows += int(parsed_price.lt(0).sum())
            zero_mask = parsed_price.eq(0)

            for column in USE_COLUMNS:
                purchase_nulls[column] += int(purchase[column].isna().sum())

            valid = purchase.loc[
                purchase["user_id"].notna()
                & purchase["user_session"].notna()
                & parsed_time.notna()
            ].copy()
            valid["parsed_time"] = parsed_time.loc[valid.index]

            users.update(valid["user_id"].tolist())
            sessions.update(valid["user_session"].tolist())
            user_sessions.update(zip(valid["user_id"], valid["user_session"]))
            user_times.update(zip(valid["user_id"], valid["parsed_time"]))
            user_session_times.update(
                zip(valid["user_id"], valid["user_session"], valid["parsed_time"])
            )

            hashes = pd.util.hash_pandas_object(
                purchase[USE_COLUMNS], index=False, categorize=True
            ).astype("uint64")
            for value in hashes.tolist():
                if value in exact_hashes:
                    confirmed_exact_duplicate_excess_rows += 1
                else:
                    exact_hashes.add(value)

            for row in valid.itertuples(index=False):
                session = row.user_session
                session_users[session].add(row.user_id)
                session_times[session].add(row.parsed_time)
                session_products[session].add(row.product_id)
                session_row_counts[session] += 1
                session_files[session].add(path.name)

            zero_price_sessions.update(
                purchase.loc[
                    zero_mask & purchase["user_session"].notna(), "user_session"
                ].tolist()
            )
            print(f"{path.name}: 원본 {processed_rows:,}행까지 처리")

    return {
        "file_counts": file_counts,
        "total_purchase_rows": total_purchase_rows,
        "users": users,
        "sessions": sessions,
        "user_sessions": user_sessions,
        "user_times": user_times,
        "user_session_times": user_session_times,
        "session_users": session_users,
        "session_times": session_times,
        "session_products": session_products,
        "session_row_counts": session_row_counts,
        "session_files": session_files,
        "zero_price_sessions": zero_price_sessions,
        "negative_price_rows": negative_price_rows,
        "price_parse_failures": price_parse_failures,
        "date_parse_failures": date_parse_failures,
        "purchase_nulls": purchase_nulls,
        "confirmed_exact_duplicate_excess_rows": confirmed_exact_duplicate_excess_rows,
    }


def distribution(counter: Counter, name: str) -> pd.DataFrame:
    """횟수별 세션 분포표를 만든다."""
    counts = Counter(counter.values())
    total = sum(counts.values())
    return pd.DataFrame([
        {
            "metric": name,
            "value": value,
            "session_count": count,
            "session_rate_pct": round(count / total * 100, 6),
        }
        for value, count in sorted(counts.items())
    ])


def make_outputs(audit: dict) -> dict[str, pd.DataFrame]:
    """감사 집계를 결과표로 변환한다."""
    session_user_counts = Counter({
        session: len(values) for session, values in audit["session_users"].items()
    })
    session_time_counts = Counter({
        session: len(values) for session, values in audit["session_times"].items()
    })
    session_product_counts = Counter({
        session: len(values) for session, values in audit["session_products"].items()
    })
    session_duration_seconds = {
        session: int((max(values) - min(values)).total_seconds())
        for session, values in audit["session_times"].items()
    }
    gap_split_counts = {}
    for threshold_minutes in (30, 60):
        threshold_seconds = threshold_minutes * 60
        event_count = 0
        split_sessions = 0
        for values in audit["session_times"].values():
            ordered = sorted(values)
            session_events = 1
            for previous, current in zip(ordered, ordered[1:]):
                if (current - previous).total_seconds() > threshold_seconds:
                    session_events += 1
            event_count += session_events
            split_sessions += int(session_events > 1)
        gap_split_counts[threshold_minutes] = {
            "event_count": event_count,
            "split_sessions": split_sessions,
        }

    candidate_rows = [
        {
            "candidate_definition": "purchase 원본 행",
            "event_count": audit["total_purchase_rows"],
            "grain": "상품 purchase 로그 1행",
            "known_risk": "다상품 구매와 확인된 완전 중복으로 구매 횟수 과대 가능",
        },
        {
            "candidate_definition": "user_session",
            "event_count": len(audit["sessions"]),
            "grain": "세션 1개",
            "known_risk": "한 세션 안의 여러 결제를 합칠 가능성",
        },
        {
            "candidate_definition": "user_id + user_session",
            "event_count": len(audit["user_sessions"]),
            "grain": "고객·세션 1개",
            "known_risk": "한 세션 안의 여러 결제를 합칠 가능성",
        },
        {
            "candidate_definition": "user_id + event_time",
            "event_count": len(audit["user_times"]),
            "grain": "고객·정확한 시각 1개",
            "known_risk": "같은 결제의 상품 로그 시각이 다르면 여러 구매로 분리",
        },
        {
            "candidate_definition": "user_id + user_session + event_time",
            "event_count": len(audit["user_session_times"]),
            "grain": "고객·세션·정확한 시각 1개",
            "known_risk": "같은 결제의 상품 로그 시각이 다르면 여러 구매로 분리",
        },
        {
            "candidate_definition": "user_id + user_session, 연속 purchase 간격 30분 초과 시 분리",
            "event_count": gap_split_counts[30]["event_count"],
            "grain": "고객·세션 내 30분 간격 구매 묶음",
            "known_risk": "실제 결제 경계가 아니라 시간 기준을 임의 적용",
        },
        {
            "candidate_definition": "user_id + user_session, 연속 purchase 간격 60분 초과 시 분리",
            "event_count": gap_split_counts[60]["event_count"],
            "grain": "고객·세션 내 60분 간격 구매 묶음",
            "known_risk": "실제 결제 경계가 아니라 시간 기준을 임의 적용",
        },
    ]

    summary_rows = [
        {"check_name": "purchase_rows", "value": audit["total_purchase_rows"]},
        {"check_name": "purchase_users", "value": len(audit["users"])},
        {"check_name": "purchase_sessions", "value": len(audit["sessions"])},
        {"check_name": "user_session_pairs", "value": len(audit["user_sessions"])},
        {"check_name": "sessions_with_multiple_users", "value": sum(v > 1 for v in session_user_counts.values())},
        {"check_name": "sessions_with_multiple_timestamps", "value": sum(v > 1 for v in session_time_counts.values())},
        {"check_name": "sessions_with_multiple_products", "value": sum(v > 1 for v in session_product_counts.values())},
        {"check_name": "sessions_crossing_month_files", "value": sum(len(v) > 1 for v in audit["session_files"].values())},
        {"check_name": "sessions_over_5_minutes", "value": sum(v > 300 for v in session_duration_seconds.values())},
        {"check_name": "sessions_over_30_minutes", "value": sum(v > 1_800 for v in session_duration_seconds.values())},
        {"check_name": "sessions_over_1_hour", "value": sum(v > 3_600 for v in session_duration_seconds.values())},
        {"check_name": "sessions_over_24_hours", "value": sum(v > 86_400 for v in session_duration_seconds.values())},
        {"check_name": "maximum_session_duration_seconds", "value": max(session_duration_seconds.values())},
        {"check_name": "sessions_split_by_30_minute_gap", "value": gap_split_counts[30]["split_sessions"]},
        {"check_name": "sessions_split_by_60_minute_gap", "value": gap_split_counts[60]["split_sessions"]},
        {"check_name": "zero_price_purchase_sessions", "value": len(audit["zero_price_sessions"])},
        {"check_name": "negative_price_purchase_rows", "value": audit["negative_price_rows"]},
        {"check_name": "date_parse_failures", "value": audit["date_parse_failures"]},
        {"check_name": "price_parse_failures", "value": audit["price_parse_failures"]},
        {"check_name": "confirmed_exact_duplicate_excess_rows", "value": audit["confirmed_exact_duplicate_excess_rows"]},
    ]
    for name, count in sorted(audit["file_counts"].items()):
        summary_rows.append({"check_name": f"{name}_purchase_rows", "value": count})
    for column, count in audit["purchase_nulls"].items():
        summary_rows.append({"check_name": f"null_{column}_rows", "value": count})

    session_distribution = pd.concat([
        distribution(audit["session_row_counts"], "purchase_rows_per_session"),
        distribution(session_time_counts, "distinct_timestamps_per_session"),
        distribution(session_product_counts, "distinct_products_per_session"),
        distribution(session_user_counts, "distinct_users_per_session"),
    ], ignore_index=True)

    duration_bins = [
        ("0초", 0, 0),
        ("1~10초", 1, 10),
        ("11~60초", 11, 60),
        ("61~300초", 61, 300),
        ("301~1800초", 301, 1_800),
        ("1801~3600초", 1_801, 3_600),
        ("1~24시간", 3_601, 86_400),
        ("24시간 초과", 86_401, float("inf")),
    ]
    duration_rows = []
    total_sessions = len(session_duration_seconds)
    for label, lower, upper in duration_bins:
        count = sum(lower <= value <= upper for value in session_duration_seconds.values())
        duration_rows.append({
            "duration_band": label,
            "session_count": count,
            "session_rate_pct": round(count / total_sessions * 100, 6),
        })

    return {
        "rees46_purchase_key_candidate_comparison.csv": pd.DataFrame(candidate_rows),
        "rees46_purchase_session_audit_summary.csv": pd.DataFrame(summary_rows),
        "rees46_purchase_session_distribution.csv": session_distribution,
        "rees46_purchase_session_duration_summary.csv": pd.DataFrame(duration_rows),
    }


def validate(audit: dict, outputs: dict[str, pd.DataFrame]) -> None:
    """핵심 합계와 후보키 논리를 검증한다."""
    expected_file_counts = {"2019-Oct.csv": 742_849, "2019-Nov.csv": 916_939}
    if dict(audit["file_counts"]) != expected_file_counts:
        raise AssertionError("이전 purchase 감사 결과와 파일별 행 수가 다릅니다.")
    if sum(audit["file_counts"].values()) != audit["total_purchase_rows"]:
        raise AssertionError("파일별 purchase 합계와 전체 purchase 행 수가 다릅니다.")
    if len(audit["sessions"]) > len(audit["user_sessions"]):
        raise AssertionError("세션 수가 고객·세션 조합 수보다 큽니다.")
    if len(audit["users"]) > audit["total_purchase_rows"]:
        raise AssertionError("구매 고객 수가 purchase 행 수보다 큽니다.")
    if audit["purchase_nulls"]["user_session"] != 0:
        raise AssertionError("이전 감사와 달리 purchase user_session 결측이 발견됐습니다.")
    if outputs["rees46_purchase_key_candidate_comparison.csv"].empty:
        raise AssertionError("대체키 후보 비교표가 비어 있습니다.")


def main() -> None:
    """대체키 후보를 감사하고 검증된 결과표를 원자적으로 저장한다."""
    check_inputs()
    audit = collect_purchase_rows()
    outputs = make_outputs(audit)
    validate(audit, outputs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        target = OUTPUT_DIR / name
        temporary = target.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(target)
        print(f"생성: {target.relative_to(PROJECT_ROOT)} ({len(frame):,}행)")
    print("REES46 구매 이벤트 대체키 후보 감사를 완료했습니다.")


if __name__ == "__main__":
    main()

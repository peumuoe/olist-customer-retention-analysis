"""Run the SQL version of the Olist retention analysis with DuckDB.

Usage (from the project root):
    pip install duckdb
    python sql/run_sql.py

Each numbered .sql file runs in order against data/raw/*.csv.
Result tables are written to outputs/sql/ and checked against the
numbers produced by the pandas pipeline (src/06, src/07).
"""

import statistics
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sql"

RESULT_TABLES = [
    "retention_window_summary",
    "second_order_timing",
    "kpi_vs_model_target",
    "first_category_repeat",
    "monthly_cohort_30d",
    "first_order_value_quartile_30d",
    "purchase_interval_by_step",
]

# Numbers reported by the pandas pipeline (README, outputs/tables).
EXPECTED_RETENTION_PCT = {30: 1.5823, 60: 1.9507, 90: 2.2756, 180: 3.1052, 365: 4.5331}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET file_search_path = '{PROJECT_ROOT.as_posix()}'")

    for path in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        con.execute(path.read_text(encoding="utf-8"))
        print(f"ran {path.name}")

    for table in RESULT_TABLES:
        df = con.table(table).df()
        df.to_csv(OUTPUT_DIR / f"{table}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{table}]")
        print(df.head(15).to_string(index=False))

    windows = con.table("retention_window_summary").df()
    mismatches = [
        (int(row.window_days), row.repeat_rate_pct, EXPECTED_RETENTION_PCT[int(row.window_days)])
        for row in windows.itertuples()
        if abs(row.repeat_rate_pct - EXPECTED_RETENTION_PCT[int(row.window_days)]) > 0.0001
    ]
    if mismatches:
        raise SystemExit(f"retention rates differ from the pandas pipeline: {mismatches}")
    print("\nretention rates match the pandas pipeline (src/07).")

    benchmark_optimization(con)
    print(f"saved to {OUTPUT_DIR}")


def benchmark_optimization(con: duckdb.DuckDBPyConnection, repeats: int = 5) -> None:
    """Compare join-then-aggregate with aggregate-then-join on the 1M-row geolocation table."""
    opt_dir = SQL_DIR / "optimization"
    before = (opt_dir / "before_join_then_aggregate.sql").read_text(encoding="utf-8")
    after = (opt_dir / "after_aggregate_then_join.sql").read_text(encoding="utf-8")

    before_df = con.execute(before).df()
    after_df = con.execute(after).df()
    if not before_df.equals(after_df):
        raise SystemExit("optimized query returns a different result")

    join_rows = {
        "before": con.execute(
            "SELECT COUNT(*) FROM raw_customers_geo c JOIN raw_geolocation g "
            "ON c.customer_zip_code_prefix = g.geolocation_zip_code_prefix"
        ).fetchone()[0],
        "after": con.execute(
            "SELECT COUNT(*) FROM raw_customers_geo c JOIN "
            "(SELECT DISTINCT geolocation_zip_code_prefix FROM raw_geolocation) z "
            "ON c.customer_zip_code_prefix = z.geolocation_zip_code_prefix"
        ).fetchone()[0],
    }

    rows = []
    for name, query in (("before", before), ("after", after)):
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            con.execute(query).fetchall()
            timings.append((time.perf_counter() - start) * 1000)
        plan = con.execute("EXPLAIN ANALYZE " + query).fetchall()
        (OUTPUT_DIR / f"explain_{name}.txt").write_text(plan[0][1], encoding="utf-8")
        rows.append({
            "version": name,
            "rows_after_join": join_rows[name],
            "median_ms": round(statistics.median(timings), 1),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "query_optimization.csv", index=False, encoding="utf-8-sig")
    print("\n[query_optimization] same result before/after")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

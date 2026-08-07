"""공개 재구매 벤치마크의 정의·출처·비교 가능성을 구조화한다."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "tables" / "public_retention_benchmark_audit.csv"
SUMMARY = ROOT / "outputs" / "tables" / "public_retention_benchmark_audit_summary.csv"
ACCESS_DATE = "2026-07-20"

ECOMMERCE_EUROPE_URL = (
    "https://www.ecommerce-europe.eu/wp-content/uploads/2016/06/"
    "Ecommerce-Benchmark-Retail-Report-2016.pdf"
)
SHOPIFY_FIELDS_URL = (
    "https://help.shopify.com/en/manual/reports-and-analytics/"
    "shopify-reports/report-types/analytics-fields"
)
SHOPIFY_VISITOR_URL = (
    "https://www.shopify.com/enterprise/blog/returning-ecommerce-visitors"
)
SENDER_URL = (
    "https://www.sender.net/marketing-glossary/repeat-purchase-rate/statistics/"
)


def row(**kwargs) -> dict:
    """공통 감사 필드를 포함한 한 행을 만든다."""
    base = {
        "access_date": ACCESS_DATE,
        "benchmark_value_low_pct": pd.NA,
        "benchmark_value_high_pct": pd.NA,
        "sample_scope": "",
        "metric_formula": "",
        "time_basis": "",
        "unit_of_analysis": "customer",
        "comparison_grade": "",
        "adoption_decision": "",
        "decision_reason": "",
    }
    base.update(kwargs)
    return base


def build_audit() -> pd.DataFrame:
    """원문에서 확인한 공개 벤치마크 후보를 구조화한다."""
    rows = []
    # 보고서의 업종별 값: 1년간 고유 고객 중 서로 다른 주문 2개 이상 고객 비율.
    for segment, value in [
        ("Department stores", 37.00),
        ("Electronics", 27.47),
        ("Fashion", 32.50),
        ("Food", 46.57),
        ("Home & Garden", 19.99),
        ("Specialty store", 28.30),
        ("Pure player", 28.00),
        ("Omnichannel", 31.00),
    ]:
        rows.append(row(
            source_id=f"ecommerce_europe_2016_{segment.lower().replace(' ', '_').replace('&', 'and')}",
            publisher="Ecommerce Europe / Ecommerce Foundation",
            publication_year=2016,
            source_title="Ecommerce Benchmark & Retail Report 2016",
            source_url=ECOMMERCE_EUROPE_URL,
            benchmark_name="Annual repeat buyers",
            segment=segment,
            benchmark_value_low_pct=value,
            benchmark_value_high_pct=value,
            sample_scope="1,068개 소매업체의 익명 벤치마크 응답",
            metric_formula="1년 동안 서로 다른 주문을 2개 이상 한 고객 / 같은 1년의 고유 고객",
            time_basis="1년 측정기간",
            unit_of_analysis="customer",
            comparison_grade="보조 비교",
            adoption_decision="채택",
            decision_reason="고객·주문 기반 공식과 1년 기준이 공개됐지만 첫 구매 후 고정 365일 추적은 아님",
        ))

    rows.append(row(
        source_id="shopify_returning_customer_rate_range",
        publisher="Shopify",
        publication_year=pd.NA,
        source_title="Analytics data points (fields) reference",
        source_url=SHOPIFY_FIELDS_URL,
        benchmark_name="Returning customer rate",
        segment="Most Shopify stores",
        benchmark_value_low_pct=20.0,
        benchmark_value_high_pct=40.0,
        sample_scope="Shopify 공식 문서의 일반적 표시 범위; 표본 수와 분포 미공개",
        metric_formula="선택 기간의 returning customers / 주문 고객",
        time_basis="사용자가 선택한 보고기간",
        unit_of_analysis="customer",
        comparison_grade="참고선",
        adoption_decision="채택",
        decision_reason="공식 출처지만 첫 구매 코호트·고정 N일 지표가 아니고 표본 방법론이 공개되지 않음",
    ))

    rows.append(row(
        source_id="shopify_returning_visitor_rate_30",
        publisher="Shopify",
        publication_year=2025,
        source_title="Returning Visitor Rate: Benchmarks and Conversion Strategies",
        source_url=SHOPIFY_VISITOR_URL,
        benchmark_name="Returning visitor rate",
        segment="Ecommerce websites",
        benchmark_value_low_pct=30.0,
        benchmark_value_high_pct=30.0,
        sample_scope="기사의 일반적 권고값; 표본 방법론 미공개",
        metric_formula="재방문자 / 고유 방문자",
        time_basis="선택한 웹 분석 기간",
        unit_of_analysis="visitor",
        comparison_grade="비교 불가",
        adoption_decision="제외",
        decision_reason="구매 고객이 아니라 웹 방문자를 측정해 Olist 재구매율과 분석 단위가 다름",
    ))

    rows.append(row(
        source_id="sender_average_repeat_purchase_28_2",
        publisher="Sender",
        publication_year=2025,
        source_title="Repeat Purchase Rate Statistics (2025–2026)",
        source_url=SENDER_URL,
        benchmark_name="Average ecommerce repeat purchase rate",
        segment="Average ecommerce store",
        benchmark_value_low_pct=28.2,
        benchmark_value_high_pct=28.2,
        sample_scope="여러 2차 출처 종합 주장; 통합 표본·가중 방식 미공개",
        metric_formula="반복 구매 고객 / 전체 고객으로 설명되나 관찰창 불명확",
        time_basis="불명확",
        unit_of_analysis="customer",
        comparison_grade="비교 불가",
        adoption_decision="제외",
        decision_reason="원자료·표본·관찰기간·통합 방법론을 재현할 수 없음",
    ))
    return pd.DataFrame(rows)


def validate(audit: pd.DataFrame) -> pd.DataFrame:
    """출처 추적성과 판정 규칙을 검증한다."""
    checks = {
        "audit_rows": (len(audit), 11),
        "unique_source_ids": (audit["source_id"].nunique(), 11),
        "missing_source_url": (int(audit["source_url"].isna().sum()), 0),
        "missing_metric_formula": (int(audit["metric_formula"].eq("").sum()), 0),
        "adopted_rows": (int(audit["adoption_decision"].eq("채택").sum()), 9),
        "excluded_rows": (int(audit["adoption_decision"].eq("제외").sum()), 2),
        "direct_comparison_rows": (
            int(audit["comparison_grade"].eq("직접 비교").sum()), 0
        ),
        "visitor_metric_excluded": (
            bool(audit.loc[audit.unit_of_analysis.eq("visitor"), "adoption_decision"].eq("제외").all()),
            True,
        ),
        "values_between_0_and_100": (
            bool(audit["benchmark_value_low_pct"].between(0, 100).all()
                 and audit["benchmark_value_high_pct"].between(0, 100).all()),
            True,
        ),
    }
    failures = {key: value for key, value in checks.items() if value[0] != value[1]}
    if failures:
        raise AssertionError(f"공개 벤치마크 감사 검증 실패: {failures}")
    return pd.DataFrame([
        {"check_name": key, "actual": actual, "expected": expected, "status": "PASS"}
        for key, (actual, expected) in checks.items()
    ])


def save(frame: pd.DataFrame, path: Path) -> None:
    """검증된 표를 UTF-8 CSV로 저장한다."""
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    print(f"생성: {path.relative_to(ROOT)} ({len(frame):,}행)")


def main() -> None:
    audit = build_audit()
    summary = validate(audit)
    save(audit, OUTPUT)
    save(summary, SUMMARY)


if __name__ == "__main__":
    main()

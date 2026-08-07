"""Build a public, aggregate-only BI data mart and data dictionary."""

from pathlib import Path
import hashlib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "tables"
OUT = ROOT / "data" / "portfolio_data"
DOCS = ROOT / "docs"
AS_OF_DATE = "2026-08-07"


def save(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_retention() -> pd.DataFrame:
    official = pd.read_csv(SOURCE / "retention_window_summary.csv")
    official = official.rename(columns={
        "eligible_customer_count": "eligible_customers",
        "repeat_customer_count": "repeat_customers",
        "repeat_rate_pct": "retention_rate_pct",
    })
    official["metric_type"] = "official_any_second_delivered_order"
    official["metric_role"] = "business_diagnosis"
    official["definition_note"] = "second delivered order within the full observation window"

    survival = pd.read_csv(SOURCE / "olist_repurchase_survival_summary.csv")
    operational = survival.loc[
        survival["definition"].eq("operational_over_24h"),
        ["horizon_days", "fixed_eligible_customer_count", "fixed_repeat_customer_count", "fixed_window_repurchase_pct"],
    ].rename(columns={
        "horizon_days": "window_days",
        "fixed_eligible_customer_count": "eligible_customers",
        "fixed_repeat_customer_count": "repeat_customers",
        "fixed_window_repurchase_pct": "retention_rate_pct",
    })
    operational["observation_cutoff"] = official["observation_cutoff"].iloc[0]
    operational["eligible_first_purchase_cutoff"] = pd.NA
    operational["non_repeat_customer_count"] = operational["eligible_customers"] - operational["repeat_customers"]
    operational["metric_type"] = "operational_second_order_over_24h"
    operational["metric_role"] = "ml_and_crm_target"
    operational["definition_note"] = "second delivered order must occur over 24 hours after first order"
    columns = [
        "metric_type", "metric_role", "window_days", "observation_cutoff",
        "eligible_first_purchase_cutoff", "eligible_customers", "repeat_customers",
        "non_repeat_customer_count", "retention_rate_pct", "definition_note",
    ]
    return pd.concat([official[columns], operational[columns]], ignore_index=True)


def build_external_long() -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "external_retention_comparison.csv")
    rows = []
    for row in source.itertuples(index=False):
        for platform, role in [
            ("Olist", "internal_official_metric"),
            ("H&M", row.hm_comparison_role),
            ("REES46", row.rees46_comparison_role),
        ]:
            prefix = platform.lower().replace("&", "m").replace(" ", "")
            if platform == "H&M":
                prefix = "hm"
            eligible = getattr(row, f"{prefix}_eligible_customers")
            repeat = getattr(row, f"{prefix}_repeat_customers")
            rate = getattr(row, f"{prefix}_retention_rate_pct")
            rows.append({
                "platform": platform,
                "window_days": row.window_days,
                "eligible_customers": eligible,
                "repeat_customers": repeat,
                "retention_rate_pct": rate,
                "comparison_role": role,
                "causal_interpretation_allowed": False,
                "comparison_note": "definitions, merchandise, channel, period and business model differ",
            })
    return pd.DataFrame(rows)


def build_segment_priority() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "olist_improvement_opportunity_priority.csv")
    keep = [
        "segment_dimension", "segment_value", "feature_availability",
        "eligible_customer_count", "repeat_customer_count", "retention_rate_pct",
        "retention_rate_ci95_low_pct", "retention_rate_ci95_high_pct",
        "gap_vs_overall_pct_point", "customer_share_pct", "median_first_payment_value",
        "actionability_raw_1_to_5", "balanced_priority_score", "balanced_priority_rank",
        "rank_robustness",
    ]
    result = frame[keep].copy()
    result["interpretation_limit"] = "observational priority; not causal uplift or future CLV"
    return result


def build_model_performance() -> pd.DataFrame:
    summary = pd.read_csv(SOURCE / "olist_ml_test_model_summary.csv")
    lift = pd.read_csv(SOURCE / "olist_ml_test_lift_recall.csv")
    result = lift.merge(summary, on="model_name", how="left", validate="many_to_one")
    result["evaluation_scope"] = "frozen_independent_time_test"
    result["allowed_use"] = "candidate ranking diagnostics"
    result["prohibited_use"] = "campaign uplift, individual certainty, or causal effect"
    return result


def build_capacity() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "olist_crm_capacity_frontier.csv")
    frame["observed_not_incremental"] = True
    frame["interpretation_note"] = "historical natural repeat purchase concentration; not campaign incrementality"
    return frame


def build_data_dictionary(files: dict[str, pd.DataFrame], lineage: dict[str, str]) -> pd.DataFrame:
    descriptions = {
        "metric_type": "재구매 지표 정의 구분",
        "metric_role": "지표의 의사결정 사용 목적",
        "window_days": "첫 관찰 구매 이후 결과 관찰일수",
        "retention_rate_pct": "적격 고객 중 재구매 고객 비율(%)",
        "eligible_customers": "해당 결과창을 완전히 관찰할 수 있는 고객 수",
        "repeat_customers": "정의된 결과창 안에서 재구매한 고객 수",
        "platform": "비교 데이터 플랫폼",
        "comparison_role": "외부 비교의 허용 수준",
        "segment_dimension": "고객군을 구성한 변수",
        "segment_value": "고객군 값",
        "balanced_priority_score": "규모·격차·가치·개입가능성 기반 전문가 판단 점수",
        "balanced_priority_rank": "균형형 우선순위 순위",
        "model_name": "예측모델 이름",
        "top_fraction": "점수 상위 선택 비율",
        "lift_at_k": "선택군 양성률을 전체 양성률로 나눈 값",
        "recall_at_k_pct": "전체 양성 중 선택군이 포착한 비율(%)",
        "selected_fraction": "CRM 발송 후보로 선택하는 비율",
        "contacts_per_observed_repurchase": "선택 고객 수/과거 관찰 재구매자 수; 증분구매 비용이 아님",
        "action_id": "의사결정 Action Item 식별자",
        "decision_status": "현재 실행·보류·금지 상태",
        "priority_score": "Action Item 전문가 판단 우선순위 점수",
        "risk_id": "CRM 부작용·리스크 식별자",
        "guardrail_or_threshold": "확대 제한에 사용할 보호지표 또는 조직 사전 기준",
        "gate": "CRM 런치 조건 식별자",
        "status": "런치 조건 준비 상태",
        "claim_id": "최종 보고 주장 식별자",
        "evidence_level": "주장을 지지하는 근거 수준",
        "source_table": "주장의 직접 원천 산출물",
    }
    grains = {
        "fact_retention.csv": "지표 정의 × 관찰기간 1행",
        "fact_external_benchmark.csv": "플랫폼 × 관찰기간 1행",
        "fact_segment_priority.csv": "고객군 차원 × 고객군 값 1행",
        "fact_model_performance.csv": "모델 × 상위 선택비율 1행",
        "fact_crm_capacity.csv": "평가범위 × 선택비율 1행",
        "fact_action_priority.csv": "Action Item 1행",
        "fact_crm_risk.csv": "리스크 1행",
        "fact_launch_gate.csv": "런치 게이트 1행",
        "fact_claim_evidence.csv": "최종 주장 1행",
    }
    rows = []
    for name, frame in files.items():
        for column in frame.columns:
            rows.append({
                "dataset": name,
                "grain": grains[name],
                "column_name": column,
                "pandas_dtype": str(frame[column].dtype),
                "nullable": bool(frame[column].isna().any()),
                "description_ko": descriptions.get(column, column.replace("_", " ")),
                "source_lineage": lineage[name],
                "public_release_class": "aggregate_public_portfolio",
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "fact_retention.csv": build_retention(),
        "fact_external_benchmark.csv": build_external_long(),
        "fact_segment_priority.csv": build_segment_priority(),
        "fact_model_performance.csv": build_model_performance(),
        "fact_crm_capacity.csv": build_capacity(),
        "fact_action_priority.csv": pd.read_csv(SOURCE / "olist_decision_action_priority.csv"),
        "fact_crm_risk.csv": pd.read_csv(SOURCE / "olist_crm_risk_register.csv"),
        "fact_launch_gate.csv": pd.read_csv(SOURCE / "olist_crm_launch_gate_checklist.csv"),
        "fact_claim_evidence.csv": pd.read_csv(SOURCE / "olist_final_claim_evidence_matrix.csv"),
    }
    lineage = {
        "fact_retention.csv": "retention_window_summary.csv + olist_repurchase_survival_summary.csv",
        "fact_external_benchmark.csv": "external_retention_comparison.csv",
        "fact_segment_priority.csv": "olist_improvement_opportunity_priority.csv",
        "fact_model_performance.csv": "olist_ml_test_model_summary.csv + olist_ml_test_lift_recall.csv",
        "fact_crm_capacity.csv": "olist_crm_capacity_frontier.csv",
        "fact_action_priority.csv": "olist_decision_action_priority.csv",
        "fact_crm_risk.csv": "olist_crm_risk_register.csv",
        "fact_launch_gate.csv": "olist_crm_launch_gate_checklist.csv",
        "fact_claim_evidence.csv": "olist_final_claim_evidence_matrix.csv",
    }
    paths = {name: save(frame, name) for name, frame in files.items()}
    dictionary = build_data_dictionary(files, lineage)
    dictionary_path = save(dictionary, "data_dictionary.csv")

    manifest_rows = []
    for name, path in paths.items():
        content = path.read_bytes()
        manifest_rows.append({
            "dataset": name,
            "row_count": len(files[name]),
            "column_count": len(files[name].columns),
            "file_size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source_lineage": lineage[name],
            "contains_customer_id": False,
            "contains_order_id": False,
            "as_of_date": AS_OF_DATE,
        })
    manifest = pd.DataFrame(manifest_rows)
    save(manifest, "data_manifest.csv")

    forbidden = {"customer_unique_id", "customer_id", "order_id", "seller_id", "review_id"}
    exposed = sorted({c for frame in files.values() for c in frame.columns} & forbidden)
    checks = pd.DataFrame([
        ("public_fact_table_count", len(files), 9, len(files) == 9),
        ("dictionary_dataset_count", dictionary["dataset"].nunique(), 9, dictionary["dataset"].nunique() == 9),
        ("dictionary_column_rows", len(dictionary), sum(len(x.columns) for x in files.values()), len(dictionary) == sum(len(x.columns) for x in files.values())),
        ("customer_or_order_identifier_columns", len(exposed), 0, len(exposed) == 0),
        ("manifest_row_count", len(manifest), 9, len(manifest) == 9),
        ("empty_fact_tables", sum(frame.empty for frame in files.values()), 0, all(not frame.empty for frame in files.values())),
        ("external_causal_allowed_true", int(files["fact_external_benchmark.csv"]["causal_interpretation_allowed"].astype(bool).sum()), 0, not files["fact_external_benchmark.csv"]["causal_interpretation_allowed"].astype(bool).any()),
        ("capacity_incremental_flag_false", int((~files["fact_crm_capacity.csv"]["observed_not_incremental"].astype(bool)).sum()), 0, files["fact_crm_capacity.csv"]["observed_not_incremental"].astype(bool).all()),
        ("dictionary_file_exists", int(dictionary_path.exists()), 1, dictionary_path.exists()),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    save(checks, "build_summary.csv")

    readme = f"""# Olist Portfolio BI Data Mart

기준일: {AS_OF_DATE}

Tableau, Power BI, Looker Studio에서 공통으로 사용할 수 있도록 만든 공개용 집계 데이터입니다.

## 공개 원칙

- 고객·주문·판매자·리뷰 식별자를 포함하지 않습니다.
- Olist/H&M/REES46 대용량 원본과 고객 단위 모델 점수는 포함하지 않습니다.
- 외부 비교는 인과 비교가 아니며 `causal_interpretation_allowed=False`입니다.
- CRM 수용량의 재구매는 과거 자연 재구매이며 캠페인 증분효과가 아닙니다.
- 역사적 후보를 현재 연락 가능한 고객으로 해석하지 않습니다.

## 파일

| 파일 | Grain | 사용 목적 |
|---|---|---|
| `fact_retention.csv` | 지표 정의 × 관찰기간 | 공식 KPI와 운영 타깃 |
| `fact_external_benchmark.csv` | 플랫폼 × 관찰기간 | 외부 관찰값 비교 |
| `fact_segment_priority.csv` | 고객군 차원 × 값 | 개선기회 우선순위 |
| `fact_model_performance.csv` | 모델 × 선택비율 | PR-AUC·Lift·Recall |
| `fact_crm_capacity.csv` | 범위 × 선택비율 | 발송 수용량 |
| `fact_action_priority.csv` | Action Item | 자원 우선순위 |
| `fact_crm_risk.csv` | 리스크 | Side Effect·Guardrail |
| `fact_launch_gate.csv` | 런치 조건 | 실행 준비상태 |
| `fact_claim_evidence.csv` | 최종 주장 | 주장–근거 추적 |
| `data_dictionary.csv` | 데이터셋 × 컬럼 | 데이터 사전 |
| `data_manifest.csv` | 데이터셋 | 행·열·해시·출처 |

각 CSV는 UTF-8 with BOM으로 저장되어 Windows BI 도구에서 한글을 읽을 수 있습니다.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    dictionary_md = [
        "# BI 데이터 사전", "", f"기준일: {AS_OF_DATE}", "",
        "상세 컬럼 단위 사전은 `data_dictionary.csv`를 사용합니다.", "",
        "## 데이터 모델 원칙", "",
        "- 모든 테이블은 집계 Grain이며 고객·주문 식별자를 포함하지 않습니다.",
        "- 테이블 간 강제 관계보다 각 분석 페이지의 목적별 Fact로 사용합니다.",
        "- 관찰기간 필터는 `window_days`, 선택비율 필터는 `top_fraction` 또는 `selected_fraction`을 사용합니다.",
        "- 공식 KPI와 운영 타깃은 `metric_type`으로 반드시 구분합니다.",
        "- 외부 비교와 CRM 관찰값을 인과효과로 표현하지 않습니다.", "",
    ]
    for name, frame in files.items():
        dictionary_md.extend([
            f"## {name}", "", f"- Grain: {dictionary.loc[dictionary['dataset'].eq(name), 'grain'].iloc[0]}",
            f"- 행 수: {len(frame):,}", f"- 원천: `{lineage[name]}`", "",
        ])
    (DOCS / "portfolio_bi_data_dictionary.md").write_text("\n".join(dictionary_md), encoding="utf-8")

    print("Portfolio BI mart")
    print(manifest[["dataset", "row_count", "column_count", "file_size_bytes"]].to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

"""Create the final report figures from verified Olist analysis tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "reports" / "figures"
SUMMARY_PATH = TABLES / "olist_final_report_figure_build_summary.csv"

NAVY = "#16324F"
BLUE = "#2E75B6"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D95D39"
GRAY = "#6B7280"
LIGHT = "#E8EEF4"


def setup_style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def retention_curve():
    data = pd.read_csv(TABLES / "retention_window_summary.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(
        data["window_days"], data["repeat_rate_pct"],
        color=BLUE, marker="o", linewidth=2.8, markersize=7,
    )
    ax.fill_between(
        data["window_days"], data["repeat_rate_pct"],
        color=BLUE, alpha=0.10,
    )
    for row in data.itertuples():
        ax.annotate(
            f"{row.repeat_rate_pct:.2f}%",
            (row.window_days, row.repeat_rate_pct),
            xytext=(0, 10), textcoords="offset points",
            ha="center", color=NAVY, fontweight="bold",
        )
    ax.set_title("Olist 재구매율은 관찰창이 길어져도 낮게 유지된다")
    ax.set_xlabel("첫 구매 후 관찰기간(일)")
    ax.set_ylabel("누적 재구매율")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(data["window_days"])
    ax.grid(axis="y", color=LIGHT)
    ax.text(
        0.01, -0.22,
        "주: 각 기간을 완전히 관찰할 수 있는 고객만 분모에 포함. 기간별 적격 고객 구성이 다름.",
        transform=ax.transAxes, fontsize=9, color=GRAY,
    )
    return save(fig, "01_olist_retention_curve.png")


def external_comparison():
    data = pd.read_csv(TABLES / "external_retention_comparison.csv")
    row = data.loc[data["window_days"].eq(30)].iloc[0]
    labels = ["Olist", "H&M", "REES46"]
    values = [
        row["olist_retention_rate_pct"],
        row["hm_retention_rate_pct"],
        row["rees46_retention_rate_pct"],
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.barh(labels, values, color=[BLUE, TEAL, ORANGE], height=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.7, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%", va="center", fontweight="bold", color=NAVY,
        )
    ax.set_xlim(0, 47)
    ax.set_xlabel("30일 관찰 재구매율")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("외부 공개 데이터에서 Olist의 30일 관찰값은 현저히 낮다")
    ax.grid(axis="x", color=LIGHT)
    ax.text(
        0.01, -0.22,
        "주의: 구매 이벤트·상품구성·채널·사업모델이 달라 동질 비교나 인과 해석은 불가.",
        transform=ax.transAxes, fontsize=9, color=RED, fontweight="bold",
    )
    return save(fig, "02_external_30d_comparison.png")


def opportunity_map():
    data = pd.read_csv(TABLES / "olist_improvement_opportunity_priority.csv").head(10)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sizes = 80 + 900 * (
        np.sqrt(data["eligible_customer_count"])
        / np.sqrt(data["eligible_customer_count"]).max()
    )
    scatter = ax.scatter(
        data["retention_gap_below_overall_pct_point"],
        data["median_first_payment_value"],
        s=sizes,
        c=data["balanced_priority_score"],
        cmap="YlOrRd",
        alpha=0.78,
        edgecolor="white",
        linewidth=1.3,
    )
    for row in data.itertuples():
        label = str(row.segment_value)
        ax.annotate(
            label,
            (row.retention_gap_below_overall_pct_point, row.median_first_payment_value),
            xytext=(5, 5), textcoords="offset points", fontsize=8.5,
        )
    ax.set_title("개선기회 후보는 격차·관찰가치·고객규모가 서로 다르다")
    ax.set_xlabel("전체보다 낮은 30일 재구매율 격차(%p)")
    ax.set_ylabel("첫 주문 결제금액 중앙값")
    ax.set_xlim(
        max(0, data["retention_gap_below_overall_pct_point"].min() - 0.05),
        data["retention_gap_below_overall_pct_point"].max() + 0.24,
    )
    ax.grid(color=LIGHT)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("균형형 우선순위 점수")
    ax.text(
        0.01, -0.16,
        "버블 크기=적격 고객 수. 관찰 기반 우선순위이며 캠페인 반응확률·미래 CLV가 아님.",
        transform=ax.transAxes, fontsize=9, color=GRAY,
    )
    return save(fig, "03_opportunity_priority_map.png")


def model_lift_recall():
    data = pd.read_csv(TABLES / "olist_ml_test_lift_recall.csv")
    data = data.loc[data["model_name"].eq("logistic_raw")].copy()
    x = data["top_fraction"] * 100
    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    bars = ax1.bar(
        x, data["recall_at_k_pct"],
        width=[1.1, 2.4, 3.5, 5.0],
        color=LIGHT, edgecolor=BLUE, label="Recall@K",
    )
    ax1.set_xlabel("점수 상위 고객 비율")
    ax1.set_ylabel("포착한 전체 재구매자 비율", color=BLUE)
    ax1.set_ylim(0, 28)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(x, [f"{v:.0f}%" for v in x])
    for bar, value in zip(bars, data["recall_at_k_pct"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, value + 0.8,
            f"{value:.1f}%", ha="center", color=BLUE, fontsize=9,
        )
    ax2 = ax1.twinx()
    ax2.plot(
        x, data["lift_at_k"], color=RED, marker="o",
        linewidth=2.5, markersize=7, label="Lift@K",
    )
    ax2.axhline(1, color=GRAY, linestyle="--", linewidth=1)
    ax2.set_ylabel("Lift", color=RED)
    ax2.set_ylim(0.88, 1.88)
    ax2.tick_params(axis="y", labelcolor=RED)
    for xv, value in zip(x, data["lift_at_k"]):
        ax2.annotate(
            f"{value:.2f}", (xv, value), xytext=(0, 9),
            textcoords="offset points", ha="center", color=RED,
        )
    ax1.set_title("모델은 상위 10%에서 농축되지만 재구매자 대부분을 놓친다")
    ax1.grid(axis="y", color=LIGHT)
    ax1.text(
        0.01, -0.20,
        "테스트 고객 17,355명·재구매자 108명. Lift는 캠페인 증분효과가 아님.",
        transform=ax1.transAxes, fontsize=9, color=GRAY,
    )
    return save(fig, "04_model_lift_recall.png")


def value_band_lift():
    data = pd.read_csv(TABLES / "olist_ml_subgroup_performance.csv")
    data = data.loc[
        data["subgroup_dimension"].eq("order_value_band")
        & data["subgroup_value"].isin(["Q1_low", "Q2", "Q3", "Q4_high"])
    ].copy()
    order = ["Q1_low", "Q2", "Q3", "Q4_high"]
    data["subgroup_value"] = pd.Categorical(
        data["subgroup_value"], categories=order, ordered=True
    )
    data = data.sort_values("subgroup_value")
    colors = [GRAY if value <= 1 else TEAL for value in data["within_group_lift_at_10pct"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    bars = ax.bar(
        data["subgroup_value"].astype(str),
        data["within_group_lift_at_10pct"],
        color=colors, width=0.62,
    )
    ax.axhline(1, color=RED, linestyle="--", linewidth=1.5, label="무작위 기준")
    for bar, value in zip(bars, data["within_group_lift_at_10pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.07,
            f"{value:.2f}", ha="center", fontweight="bold",
        )
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("가치구간 내부 Lift@10%")
    ax.set_xlabel("첫 주문 가치구간")
    ax.set_title("예측 순위 신호는 Q3·Q4에서 상대적으로 강했다")
    ax.grid(axis="y", color=LIGHT)
    ax.text(
        0.01, -0.18,
        "하위집단 결과는 테스트 양성이 적어 운영 위험 제한에만 사용하며 인과효과가 아님.",
        transform=ax.transAxes, fontsize=9, color=GRAY,
    )
    return save(fig, "05_value_band_lift.png")


def experiment_power():
    data = pd.read_csv(TABLES / "olist_crm_experiment_power_scenarios.csv")
    data = data.loc[data["scenario"].str.startswith("relative_uplift")].copy()
    labels = data["relative_effect"].mul(100).round().astype(int).map(
        lambda value: f"상대 +{value}%"
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.barh(labels, data["required_total_n"], color=ORANGE, alpha=0.85)
    ax.axvline(4667, color=BLUE, linestyle="--", linewidth=2, label="현재 후보 4,667명")
    for bar, value in zip(bars, data["required_total_n"]):
        ax.text(
            value + 700, bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}명", va="center", fontsize=9, fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(1000, 80000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    ax.set_xlabel("80% 검정력에 필요한 총 고객 수(로그 축)")
    ax.set_title("낮은 기준 재구매율 때문에 중간 효과 검증에는 큰 표본이 필요하다")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=LIGHT, which="both")
    ax.text(
        0.01, -0.19,
        "계획 기준율 1.768%, 양측 유의수준 5%. 현재 후보는 상대 약 +70.9%가 MDE.",
        transform=ax.transAxes, fontsize=9, color=GRAY,
    )
    return save(fig, "06_experiment_power.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    paths = [
        retention_curve(),
        external_comparison(),
        opportunity_map(),
        model_lift_recall(),
        value_band_lift(),
        experiment_power(),
    ]
    checks = []
    for path in paths:
        checks.append({
            "figure_name": path.name,
            "exists": path.exists(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "minimum_size_bytes": 20_000,
            "passed": path.exists() and path.stat().st_size >= 20_000,
        })
    summary = pd.DataFrame(checks)
    if not summary["passed"].all():
        raise AssertionError(summary.loc[~summary["passed"]].to_dict("records"))
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

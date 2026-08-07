"""Olist 첫 구매 특성과 30일 재구매의 다변량 조건부 연관성을 분석한다."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MART_PATH = ROOT / "data" / "processed" / "olist_first_purchase_customer_mart.csv"
TABLES = ROOT / "outputs" / "tables"
COEF_PATH = TABLES / "olist_multivariable_coefficients.csv"
ADJUSTED_PATH = TABLES / "olist_multivariable_adjusted_probabilities.csv"
MODEL_PATH = TABLES / "olist_multivariable_model_summary.csv"
SENSITIVITY_PATH = TABLES / "olist_multivariable_regularization_sensitivity.csv"
CHECK_PATH = TABLES / "olist_multivariable_build_summary.csv"

EXPECTED_ROWS = 86_772
EXPECTED_POSITIVES = 1_373
RARE_MIN_COUNT = 500
BASE_LAMBDA = 1.0
SENSITIVITY_LAMBDAS = [0.1, 1.0, 10.0]

ORDER_FEATURES = [
    "primary_category", "customer_state", "order_value_band",
    "item_count_group", "seller_count_group", "primary_payment_type",
    "installment_group", "first_purchase_month",
]
EXPERIENCE_FEATURES = ORDER_FEATURES + [
    "delivery_experience_group", "review_score_group",
]
EXPERIENCE_ONLY = {"delivery_experience_group", "review_score_group"}


def sigmoid(values: np.ndarray) -> np.ndarray:
    """수치적으로 안정적인 로지스틱 함수."""
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def prepare_data() -> pd.DataFrame:
    """30일 적격 고객과 분석용 범주를 준비한다."""
    if not MART_PATH.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {MART_PATH}")
    data = pd.read_csv(MART_PATH, low_memory=False)
    data = data.loc[data["eligible_30d"].eq(True)].copy()
    data["repeat_within_30d"] = data["repeat_within_30d"].astype(int)
    data["first_purchase_month"] = pd.to_datetime(
        data["order_purchase_timestamp"], errors="raise"
    ).dt.to_period("M").astype(str)
    for feature in set(EXPERIENCE_FEATURES):
        data[feature] = data[feature].astype("string").fillna("unknown")
        counts = data[feature].value_counts(dropna=False)
        rare = counts[counts.lt(RARE_MIN_COUNT)].index
        data.loc[data[feature].isin(rare), feature] = "other_rare"
    return data


def choose_reference(feature: str, series: pd.Series) -> str:
    """해석 가능한 기준 범주를 우선하고 없으면 최빈값을 사용한다."""
    preferred = {
        "item_count_group": "1",
        "seller_count_group": "single_seller",
        "order_value_band": "Q1_low",
        "primary_payment_type": "credit_card",
        "installment_group": "1",
        "delivery_experience_group": "on_time_or_early",
        "review_score_group": "5",
    }.get(feature)
    levels = set(series.unique())
    if preferred in levels:
        return preferred
    return str(series.value_counts().index[0])


def build_matrix(
    data: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, list[dict], dict[str, dict]]:
    """기준 범주를 제외한 원-핫 행렬과 열 메타데이터를 만든다."""
    columns = [np.ones(len(data), dtype=np.float64)]
    metadata = [{
        "column_index": 0, "feature": "intercept", "level": "intercept",
        "reference_level": "", "support_count": len(data),
    }]
    specs = {}
    column_index = 1
    for feature in features:
        series = data[feature].astype(str)
        reference = choose_reference(feature, series)
        levels = sorted(series.unique())
        level_to_column = {reference: None}
        for level in levels:
            if level == reference:
                continue
            columns.append(series.eq(level).to_numpy(dtype=np.float64))
            level_to_column[level] = column_index
            metadata.append({
                "column_index": column_index,
                "feature": feature,
                "level": level,
                "reference_level": reference,
                "support_count": int(series.eq(level).sum()),
            })
            column_index += 1
        specs[feature] = {
            "reference": reference,
            "levels": levels,
            "level_to_column": level_to_column,
            "current_levels": series.to_numpy(),
        }
    return np.column_stack(columns), metadata, specs


def fit_ridge_logistic(
    x: np.ndarray, y: np.ndarray, ridge_lambda: float,
    max_iter: int = 60, tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """L2 정규화 로지스틱 회귀를 Newton/IRLS로 적합한다."""
    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.full(x.shape[1], ridge_lambda, dtype=np.float64)
    penalty[0] = 0.0
    converged = False
    max_delta = np.inf
    for iteration in range(1, max_iter + 1):
        probability = sigmoid(x @ beta)
        weight = np.clip(probability * (1 - probability), 1e-8, None)
        gradient = x.T @ (y - probability) - penalty * beta
        hessian = (x.T * weight) @ x
        hessian.flat[:: hessian.shape[0] + 1] += penalty
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta += delta
        max_delta = float(np.max(np.abs(delta)))
        if max_delta < tolerance:
            converged = True
            break
    probability = sigmoid(x @ beta)
    eps = 1e-15
    log_loss = -float(np.mean(
        y * np.log(np.clip(probability, eps, 1 - eps))
        + (1 - y) * np.log(np.clip(1 - probability, eps, 1 - eps))
    ))
    brier = float(np.mean((probability - y) ** 2))
    return beta, {
        "iterations": iteration,
        "converged": converged,
        "max_abs_last_step": max_delta,
        "log_loss": log_loss,
        "brier_score": brier,
        "mean_predicted_rate_pct": float(probability.mean() * 100),
        "min_predicted_probability": float(probability.min()),
        "max_predicted_probability": float(probability.max()),
    }


def coefficient_table(
    model_name: str, beta: np.ndarray, metadata: list[dict]
) -> pd.DataFrame:
    """계수를 오즈비와 함께 정리한다."""
    result = pd.DataFrame(metadata)
    result["model_name"] = model_name
    result["coefficient"] = beta[result["column_index"].to_numpy()]
    result["odds_ratio_vs_reference"] = np.exp(
        np.clip(result["coefficient"], -20, 20)
    )
    result["feature_availability"] = np.where(
        result["feature"].isin(EXPERIENCE_ONLY), "experience_complete", "order_time"
    )
    numeric = ["coefficient", "odds_ratio_vs_reference"]
    result[numeric] = result[numeric].round(6)
    return result[[
        "model_name", "feature", "level", "reference_level", "support_count",
        "feature_availability", "coefficient", "odds_ratio_vs_reference",
    ]]


def adjusted_probability_table(
    model_name: str, data: pd.DataFrame, x: np.ndarray, beta: np.ndarray,
    specs: dict[str, dict], features: list[str],
) -> pd.DataFrame:
    """각 수준으로 표준화한 모델 기반 평균 예측확률을 계산한다."""
    base_linear = x @ beta
    rows = []
    outcome = data["repeat_within_30d"].to_numpy()
    for feature in features:
        spec = specs[feature]
        current_contribution = np.zeros(len(data), dtype=np.float64)
        for level, column in spec["level_to_column"].items():
            if column is not None:
                current_contribution[spec["current_levels"] == level] = beta[column]
        reference = spec["reference"]
        reference_probability = float(sigmoid(base_linear - current_contribution).mean() * 100)
        for level in spec["levels"]:
            column = spec["level_to_column"][level]
            level_contribution = 0.0 if column is None else beta[column]
            adjusted_rate = float(sigmoid(
                base_linear - current_contribution + level_contribution
            ).mean() * 100)
            mask = spec["current_levels"] == level
            rows.append({
                "model_name": model_name,
                "feature": feature,
                "level": level,
                "reference_level": reference,
                "support_count": int(mask.sum()),
                "observed_retention_rate_pct": float(outcome[mask].mean() * 100),
                "adjusted_retention_rate_pct": adjusted_rate,
                "adjusted_gap_vs_reference_pct_point": adjusted_rate - reference_probability,
                "feature_availability": (
                    "experience_complete" if feature in EXPERIENCE_ONLY else "order_time"
                ),
            })
    result = pd.DataFrame(rows)
    numeric = [
        "observed_retention_rate_pct", "adjusted_retention_rate_pct",
        "adjusted_gap_vs_reference_pct_point",
    ]
    result[numeric] = result[numeric].round(6)
    return result


def fit_model(
    model_name: str, data: pd.DataFrame, features: list[str], ridge_lambda: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray, list[dict]]:
    x, metadata, specs = build_matrix(data, features)
    y = data["repeat_within_30d"].to_numpy(dtype=np.float64)
    beta, diagnostics = fit_ridge_logistic(x, y, ridge_lambda)
    diagnostics.update({
        "model_name": model_name,
        "ridge_lambda": ridge_lambda,
        "customer_count": len(data),
        "positive_count": int(y.sum()),
        "feature_count_including_intercept": x.shape[1],
    })
    coefficients = coefficient_table(model_name, beta, metadata)
    adjusted = adjusted_probability_table(
        model_name, data, x, beta, specs, features
    )
    return coefficients, adjusted, diagnostics, beta, metadata


def build_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """정규화 강도별 주문 시점 모델 계수를 비교한다."""
    x, metadata, _ = build_matrix(data, ORDER_FEATURES)
    y = data["repeat_within_30d"].to_numpy(dtype=np.float64)
    meta = pd.DataFrame(metadata)
    rows = []
    diagnostics = []
    for ridge_lambda in SENSITIVITY_LAMBDAS:
        beta, diag = fit_ridge_logistic(x, y, ridge_lambda)
        diag.update({
            "model_name": "order_time_sensitivity",
            "ridge_lambda": ridge_lambda,
            "customer_count": len(data),
            "positive_count": int(y.sum()),
            "feature_count_including_intercept": x.shape[1],
        })
        diagnostics.append(diag)
        current = meta.copy()
        current = current.loc[current["feature"].ne("intercept")]
        current["ridge_lambda"] = ridge_lambda
        current["coefficient"] = beta[current["column_index"].to_numpy()]
        current["odds_ratio_vs_reference"] = np.exp(
            np.clip(current["coefficient"], -20, 20)
        )
        rows.append(current)
    result = pd.concat(rows, ignore_index=True)
    result[["coefficient", "odds_ratio_vs_reference"]] = result[[
        "coefficient", "odds_ratio_vs_reference"
    ]].round(6)
    return result[[
        "ridge_lambda", "feature", "level", "reference_level", "support_count",
        "coefficient", "odds_ratio_vs_reference",
    ]], diagnostics


def validate(
    data: pd.DataFrame, coefficients: pd.DataFrame, adjusted: pd.DataFrame,
    model_summary: pd.DataFrame, sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """분모·시점·계수·확률·수렴 조건을 검증한다."""
    checks = {
        "eligible_rows": (len(data), EXPECTED_ROWS),
        "positive_rows": (int(data["repeat_within_30d"].sum()), EXPECTED_POSITIVES),
        "duplicate_customers": (int(data["customer_unique_id"].duplicated().sum()), 0),
        "base_models": (
            int(model_summary["model_name"].isin(
                ["order_time", "experience_complete"]
            ).sum()), 2
        ),
        "model_fits_including_sensitivity": (len(model_summary), 5),
        "all_model_fits_converged": (bool(model_summary["converged"].all()), True),
        "finite_coefficients": (bool(np.isfinite(coefficients["coefficient"]).all()), True),
        "positive_odds_ratios": (bool(coefficients["odds_ratio_vs_reference"].gt(0).all()), True),
        "adjusted_rates_in_0_100": (bool(adjusted["adjusted_retention_rate_pct"].between(0, 100).all()), True),
        "experience_features_absent_from_order_model": (
            int(coefficients.loc[
                coefficients.model_name.eq("order_time"), "feature"
            ].isin(EXPERIENCE_ONLY).sum()), 0,
        ),
        "sensitivity_lambdas": (sensitivity["ridge_lambda"].nunique(), 3),
        "sensitivity_finite": (bool(np.isfinite(sensitivity["coefficient"]).all()), True),
    }
    failures = {key: value for key, value in checks.items() if value[0] != value[1]}
    if failures:
        raise AssertionError(f"다변량 연관성 분석 검증 실패: {failures}")
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
    data = prepare_data()
    order_coef, order_adjusted, order_diag, _, _ = fit_model(
        "order_time", data, ORDER_FEATURES, BASE_LAMBDA
    )
    exp_coef, exp_adjusted, exp_diag, _, _ = fit_model(
        "experience_complete", data, EXPERIENCE_FEATURES, BASE_LAMBDA
    )
    coefficients = pd.concat([order_coef, exp_coef], ignore_index=True)
    adjusted = pd.concat([order_adjusted, exp_adjusted], ignore_index=True)
    sensitivity, sensitivity_diagnostics = build_sensitivity(data)
    model_summary = pd.DataFrame(
        [order_diag, exp_diag] + sensitivity_diagnostics
    )
    checks = validate(
        data, coefficients, adjusted, model_summary, sensitivity
    )
    save(coefficients, COEF_PATH)
    save(adjusted, ADJUSTED_PATH)
    save(model_summary, MODEL_PATH)
    save(sensitivity, SENSITIVITY_PATH)
    save(checks, CHECK_PATH)


if __name__ == "__main__":
    main()

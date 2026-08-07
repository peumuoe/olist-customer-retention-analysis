"""Research appendix: validate class weighting and positive oversampling.

All imbalance treatment is applied to the training split only. Candidate
selection uses the original validation split. Test results are post-hoc
diagnostics because the project test period has already been inspected.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
spec = spec_from_file_location("ml26", ROOT / "src" / "26_train_olist_final_repurchase_models.py")
ml26 = module_from_spec(spec)
spec.loader.exec_module(ml26)

TARGET = ml26.TARGET
FEATURES = ml26.NUMERIC + ml26.CATEGORICAL
SEED = 42


def top10(y, p):
    y = np.asarray(y, dtype=int)
    n = int(np.ceil(len(y) * 0.10))
    chosen = np.argsort(-np.asarray(p), kind="stable")[:n]
    captured = int(y[chosen].sum())
    base = y.mean()
    return {
        "selected_count": n,
        "captured_positive_count": captured,
        "precision_at_10pct": captured / n,
        "recall_at_10pct": captured / y.sum(),
        "lift_at_10pct": (captured / n) / base,
        "false_positive_at_10pct": n - captured,
        "false_negative_at_10pct": int(y.sum()) - captured,
    }


def evaluate(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        "positive_count": int(y.sum()),
        "positive_rate_pct": y.mean() * 100,
        "mean_probability_pct": p.mean() * 100,
        "calibration_ratio": p.mean() / y.mean(),
        "pr_auc": average_precision_score(y, p),
        "pr_auc_over_prevalence": average_precision_score(y, p) / y.mean(),
        "roc_auc": roc_auc_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
        **top10(y, p),
    }


def random_oversample(frame, target_rate, seed=SEED):
    positive = frame.loc[frame[TARGET].astype(bool)]
    negative = frame.loc[~frame[TARGET].astype(bool)]
    desired_positive = int(np.ceil(target_rate * len(negative) / (1 - target_rate)))
    if desired_positive < len(positive):
        raise ValueError("target rate is below the original positive rate")
    extra = positive.sample(
        n=desired_positive - len(positive), replace=True, random_state=seed
    )
    return pd.concat([frame, extra], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def fit_candidate(train, strategy, value):
    fit_frame = train
    class_weight = None
    if strategy == "class_weight":
        class_weight = "balanced" if value == "balanced" else {0: 1.0, 1: float(value)}
    elif strategy == "random_oversampling":
        fit_frame = random_oversample(train, float(value))
    elif strategy != "baseline":
        raise ValueError(strategy)
    model = Pipeline([
        ("prep", ml26.make_preprocessor()),
        ("model", LogisticRegression(
            C=0.01, max_iter=3000, solver="lbfgs", class_weight=class_weight
        )),
    ])
    model.fit(fit_frame[FEATURES], fit_frame[TARGET].astype(int))
    return model, fit_frame


def candidate_specs():
    return [
        ("baseline", "baseline", None),
        ("weight_5", "class_weight", 5),
        ("weight_10", "class_weight", 10),
        ("weight_20", "class_weight", 20),
        ("weight_balanced", "class_weight", "balanced"),
        ("oversample_1pct", "random_oversampling", 0.01),
        ("oversample_2pct", "random_oversampling", 0.02),
        ("oversample_5pct", "random_oversampling", 0.05),
    ]


def compare_candidates(parts):
    rows, models, predictions = [], {}, {}
    for name, strategy, value in candidate_specs():
        model, fit_frame = fit_candidate(parts["train"], strategy, value)
        models[name] = model
        for split in ["train", "validation_select"]:
            p = model.predict_proba(parts[split][FEATURES])[:, 1]
            predictions[(name, split)] = p
            rows.append({
                "candidate": name,
                "strategy": strategy,
                "strategy_value": value,
                "evaluation_split": split,
                "fit_row_count": len(fit_frame),
                "fit_positive_count": int(fit_frame[TARGET].sum()),
                "fit_positive_rate_pct": fit_frame[TARGET].mean() * 100,
                **evaluate(parts[split][TARGET], p),
            })
    result = pd.DataFrame(rows)
    validation = result.loc[result["evaluation_split"].eq("validation_select")].copy()
    # PR-AUC is primary; Lift breaks only exact/near ties in this research appendix.
    validation["selection_rank"] = validation["pr_auc"].rank(method="min", ascending=False)
    selected = validation.sort_values(
        ["pr_auc", "lift_at_10pct", "candidate"], ascending=[False, False, True]
    ).iloc[0]["candidate"]
    return result, models, predictions, selected


def rank_stability(parts, models, selected):
    rows = []
    baseline = models["baseline"]
    for split in ["validation_select", "calibration", "test"]:
        base_p = baseline.predict_proba(parts[split][FEATURES])[:, 1]
        selected_p = models[selected].predict_proba(parts[split][FEATURES])[:, 1]
        n = int(np.ceil(len(base_p) * 0.10))
        base_ids = set(parts[split].iloc[np.argsort(-base_p, kind="stable")[:n]]["customer_unique_id"])
        selected_ids = set(parts[split].iloc[np.argsort(-selected_p, kind="stable")[:n]]["customer_unique_id"])
        rows.append({
            "selected_candidate": selected,
            "evaluation_split": split,
            "spearman_score_rank_correlation": pd.Series(base_p).corr(pd.Series(selected_p), method="spearman"),
            "top10_intersection_count": len(base_ids & selected_ids),
            "top10_union_count": len(base_ids | selected_ids),
            "top10_jaccard_overlap": len(base_ids & selected_ids) / len(base_ids | selected_ids),
        })
    return pd.DataFrame(rows)


def posthoc_test_diagnostic(parts, models, selected):
    rows = []
    for name in ["baseline", selected]:
        raw_cal = models[name].predict_proba(parts["calibration"][FEATURES])[:, 1]
        calibrator = ml26.platt_fit(raw_cal, parts["calibration"][TARGET].astype(int).to_numpy())
        raw_test = models[name].predict_proba(parts["test"][FEATURES])[:, 1]
        calibrated_test = ml26.platt_predict(calibrator, raw_test)
        for probability_type, p in [("raw", raw_test), ("platt", calibrated_test)]:
            rows.append({
                "candidate": name,
                "probability_type": probability_type,
                "evaluation_status": "posthoc_research_only_test_already_seen",
                **evaluate(parts["test"][TARGET], p),
            })
    return pd.DataFrame(rows)


def learning_curve(parts, selected):
    rows = []
    specs = {name: (strategy, value) for name, strategy, value in candidate_specs()}
    train = parts["train"].reset_index(drop=True)
    for name in list(dict.fromkeys(["baseline", selected])):
        strategy, value = specs[name]
        for fraction in [0.20, 0.40, 0.60, 0.80, 1.00]:
            subset = train.iloc[:int(len(train) * fraction)].copy()
            if subset[TARGET].sum() < 10:
                continue
            model, fit_frame = fit_candidate(subset, strategy, value)
            for split, frame in [("train_prefix_original", subset), ("validation_select", parts["validation_select"])]:
                p = model.predict_proba(frame[FEATURES])[:, 1]
                rows.append({
                    "candidate": name,
                    "train_fraction": fraction,
                    "original_train_row_count": len(subset),
                    "original_train_positive_count": int(subset[TARGET].sum()),
                    "fit_row_count_after_sampling": len(fit_frame),
                    "evaluation_split": split,
                    **evaluate(frame[TARGET], p),
                })
    return pd.DataFrame(rows)


def build_conclusion(candidate_result, test_result, selected, stability):
    val = candidate_result.loc[candidate_result["evaluation_split"].eq("validation_select")].set_index("candidate")
    base = val.loc["baseline"]
    best = val.loc[selected]
    test = test_result.loc[test_result["probability_type"].eq("raw")].set_index("candidate")
    rank_gain = best["pr_auc"] - base["pr_auc"]
    test_gain = test.loc[selected, "pr_auc"] - test.loc["baseline", "pr_auc"]
    stable_improvement = rank_gain > 0 and test_gain > 0
    conclusion = (
        "limited_rank_improvement_research_only" if stable_improvement
        else "no_stable_rank_improvement_probability_shift_dominates"
    )
    return pd.DataFrame([{
        "selected_on_validation": selected,
        "validation_pr_auc_baseline": base["pr_auc"],
        "validation_pr_auc_selected": best["pr_auc"],
        "validation_pr_auc_difference": rank_gain,
        "validation_lift10_baseline": base["lift_at_10pct"],
        "validation_lift10_selected": best["lift_at_10pct"],
        "posthoc_test_pr_auc_baseline": test.loc["baseline", "pr_auc"],
        "posthoc_test_pr_auc_selected": test.loc[selected, "pr_auc"],
        "posthoc_test_pr_auc_difference": test_gain,
        "test_top10_jaccard_vs_baseline": stability.set_index("evaluation_split").loc["test", "top10_jaccard_overlap"],
        "diagnostic_conclusion": conclusion,
        "final_model_change_authorized": False,
        "smotenc_next_step_recommended": False,
        "reason": "oversampling changes emphasis but does not create new customer behavior; test is already reused",
    }])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    data = ml26.load_data()
    parts = {name: data.loc[data["ml_split"].eq(name)].copy() for name in [
        "train", "validation_select", "calibration", "test"
    ]}
    candidates, models, predictions, selected = compare_candidates(parts)
    stability = rank_stability(parts, models, selected)
    test = posthoc_test_diagnostic(parts, models, selected)
    curves = learning_curve(parts, selected)
    conclusion = build_conclusion(candidates, test, selected, stability)

    baseline_val = candidates.query("candidate == 'baseline' and evaluation_split == 'validation_select'").iloc[0]
    checks = pd.DataFrame([
        ("eligible_rows", len(data), 86772, len(data) == 86772),
        ("original_positive_count", int(data[TARGET].sum()), 535, int(data[TARGET].sum()) == 535),
        ("candidate_count", candidates["candidate"].nunique(), 8, candidates["candidate"].nunique() == 8),
        ("candidate_result_rows", len(candidates), 16, len(candidates) == 16),
        ("validation_original_rows_unchanged", candidates.query("evaluation_split == 'validation_select'")["positive_count"].nunique(), 1, candidates.query("evaluation_split == 'validation_select'")["positive_count"].nunique() == 1),
        ("baseline_validation_positive_count", int(baseline_val["positive_count"]), 44, int(baseline_val["positive_count"]) == 44),
        ("all_probabilities_finite", int(np.isfinite(candidates[["pr_auc", "roc_auc", "brier_score", "log_loss"]].to_numpy()).all()), 1, np.isfinite(candidates[["pr_auc", "roc_auc", "brier_score", "log_loss"]].to_numpy()).all()),
        ("posthoc_test_rows", len(test), 4, len(test) == 4),
        ("rank_stability_rows", len(stability), 3, len(stability) == 3),
        ("final_model_change_authorized", int(conclusion.iloc[0]["final_model_change_authorized"]), 0, not conclusion.iloc[0]["final_model_change_authorized"]),
    ], columns=["check_name", "actual_value", "expected_value", "passed"])
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))

    candidates.to_csv(TABLES / "olist_imbalance_validation_candidates.csv", index=False, encoding="utf-8-sig")
    test.to_csv(TABLES / "olist_imbalance_posthoc_test_diagnostic.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(TABLES / "olist_imbalance_rank_stability.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(TABLES / "olist_imbalance_learning_curve.csv", index=False, encoding="utf-8-sig")
    conclusion.to_csv(TABLES / "olist_imbalance_diagnostic_conclusion.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(TABLES / "olist_imbalance_build_summary.csv", index=False, encoding="utf-8-sig")
    print("Validation candidates")
    print(candidates.query("evaluation_split == 'validation_select'").to_string(index=False))
    print("\nPost-hoc test diagnostic")
    print(test.to_string(index=False))
    print("\nConclusion")
    print(conclusion.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

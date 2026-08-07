"""Analyze over-24-hour product, seller, price, and timing follow-up patterns."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

TRANSITION_PATH = PROCESSED / "olist_customer_30d_category_transition.csv"
ITEMS_PATH = RAW / "olist_order_items_dataset.csv"

DETAIL_PATH = PROCESSED / "olist_priority_category_over_24h_followup_detail.csv"
SUMMARY_PATH = TABLES / "olist_priority_category_followup_behavior_summary.csv"
TIMING_PATH = TABLES / "olist_priority_category_followup_timing_summary.csv"
PRODUCT_PATH = TABLES / "olist_priority_category_followup_product_patterns.csv"
PRICE_PATH = TABLES / "olist_priority_category_followup_price_movement.csv"
CHECK_PATH = TABLES / "olist_priority_category_followup_build_summary.csv"

PRIORITY_CATEGORIES = ["watches_gifts", "auto", "toys"]
EXPECTED_COUNTS = {"watches_gifts": 37, "auto": 18, "toys": 18}


def order_features(order_ids: set[str]) -> pd.DataFrame:
    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["order_id", "product_id", "seller_id", "price", "freight_value"],
        dtype={"order_id": "string", "product_id": "string", "seller_id": "string"},
    )
    items = items.loc[items["order_id"].isin(order_ids)].copy()
    return items.groupby("order_id", as_index=False).agg(
        product_ids=("product_id", lambda x: tuple(sorted(set(x.dropna())))),
        seller_ids=("seller_id", lambda x: tuple(sorted(set(x.dropna())))),
        item_row_count=("product_id", "size"),
        unique_product_count=("product_id", "nunique"),
        unique_seller_count=("seller_id", "nunique"),
        item_price_sum=("price", "sum"),
        freight_value_sum=("freight_value", "sum"),
    )


def intersection_size(left: tuple, right: tuple) -> int:
    return len(set(left) & set(right))


def jaccard(left: tuple, right: tuple) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else np.nan


def build_detail() -> pd.DataFrame:
    transitions = pd.read_csv(TRANSITION_PATH, low_memory=False)
    detail = transitions.loc[
        transitions["first_category"].isin(PRIORITY_CATEGORIES)
        & transitions["days_to_second_purchase"].gt(1)
    ].copy()
    order_ids = set(detail["first_order_id"]) | set(detail["second_order_id"])
    features = order_features(order_ids)
    first = features.add_prefix("first_").rename(
        columns={"first_order_id": "first_order_id"}
    )
    second = features.add_prefix("second_").rename(
        columns={"second_order_id": "second_order_id"}
    )
    detail = detail.merge(first, on="first_order_id", validate="one_to_one")
    detail = detail.merge(second, on="second_order_id", validate="one_to_one")

    detail["shared_product_count"] = [
        intersection_size(a, b)
        for a, b in zip(detail["first_product_ids"], detail["second_product_ids"])
    ]
    detail["shared_seller_count"] = [
        intersection_size(a, b)
        for a, b in zip(detail["first_seller_ids"], detail["second_seller_ids"])
    ]
    detail["any_same_product_flag"] = detail["shared_product_count"].gt(0)
    detail["exact_product_set_match_flag"] = [
        a == b for a, b in zip(detail["first_product_ids"], detail["second_product_ids"])
    ]
    detail["product_jaccard"] = [
        jaccard(a, b)
        for a, b in zip(detail["first_product_ids"], detail["second_product_ids"])
    ]
    detail["any_same_seller_flag"] = detail["shared_seller_count"].gt(0)
    detail["exact_seller_set_match_flag"] = [
        a == b for a, b in zip(detail["first_seller_ids"], detail["second_seller_ids"])
    ]
    detail["seller_jaccard"] = [
        jaccard(a, b)
        for a, b in zip(detail["first_seller_ids"], detail["second_seller_ids"])
    ]
    detail["same_category_different_product_flag"] = (
        detail["same_category_flag"] & ~detail["any_same_product_flag"]
    )
    detail["product_behavior"] = np.select(
        [
            detail["exact_product_set_match_flag"],
            detail["any_same_product_flag"],
            detail["same_category_different_product_flag"],
            detail["same_category_flag"],
        ],
        [
            "exact_product_set_repeat", "partial_product_overlap",
            "same_category_different_product", "same_category_other_pattern",
        ],
        default="cross_category_no_product_overlap",
    )
    detail["second_to_first_item_price_ratio"] = (
        detail["second_item_price_sum"] / detail["first_item_price_sum"]
    )
    detail["item_price_change"] = (
        detail["second_item_price_sum"] - detail["first_item_price_sum"]
    )
    detail["price_movement"] = np.select(
        [detail["item_price_change"].gt(0), detail["item_price_change"].lt(0)],
        ["higher", "lower"],
        default="equal",
    )
    detail["followup_day_band"] = pd.cut(
        detail["days_to_second_purchase"],
        bins=[1, 3, 7, 14, 30],
        labels=["over_1_to_3_days", "over_3_to_7_days", "over_7_to_14_days", "over_14_to_30_days"],
        include_lowest=False,
    )
    for column in ["first_product_ids", "second_product_ids", "first_seller_ids", "second_seller_ids"]:
        detail[column] = detail[column].map(lambda x: "|".join(x))
    detail[["product_jaccard", "seller_jaccard", "second_to_first_item_price_ratio", "item_price_change"]] = detail[[
        "product_jaccard", "seller_jaccard", "second_to_first_item_price_ratio", "item_price_change"
    ]].round(6)
    return detail


def summarize_scope(frame: pd.DataFrame, category: str, scope: str) -> dict:
    return {
        "first_category": category,
        "analysis_scope": scope,
        "customer_count": len(frame),
        "same_category_count": int(frame["same_category_flag"].sum()),
        "same_category_pct": frame["same_category_flag"].mean() * 100,
        "any_same_product_count": int(frame["any_same_product_flag"].sum()),
        "any_same_product_pct": frame["any_same_product_flag"].mean() * 100,
        "exact_product_set_match_count": int(frame["exact_product_set_match_flag"].sum()),
        "exact_product_set_match_pct": frame["exact_product_set_match_flag"].mean() * 100,
        "same_category_different_product_count": int(frame["same_category_different_product_flag"].sum()),
        "same_category_different_product_pct": frame["same_category_different_product_flag"].mean() * 100,
        "any_same_seller_count": int(frame["any_same_seller_flag"].sum()),
        "any_same_seller_pct": frame["any_same_seller_flag"].mean() * 100,
        "exact_seller_set_match_count": int(frame["exact_seller_set_match_flag"].sum()),
        "exact_seller_set_match_pct": frame["exact_seller_set_match_flag"].mean() * 100,
        "median_days_to_second_purchase": frame["days_to_second_purchase"].median(),
        "median_first_item_price_sum": frame["first_item_price_sum"].median(),
        "median_second_item_price_sum": frame["second_item_price_sum"].median(),
        "median_second_to_first_price_ratio": frame["second_to_first_item_price_ratio"].median(),
    }


def build_summaries(detail: pd.DataFrame):
    rows = []
    for category in PRIORITY_CATEGORIES:
        category_frame = detail.loc[detail["first_category"].eq(category)]
        rows.append(summarize_scope(category_frame, category, "all_over_24h_followups"))
        rows.append(summarize_scope(
            category_frame.loc[category_frame["same_category_flag"]],
            category,
            "same_category_over_24h_followups",
        ))
    summary = pd.DataFrame(rows)
    numeric = summary.select_dtypes(include="number").columns
    summary[numeric] = summary[numeric].round(6)

    timing = (
        detail.groupby(["first_category", "followup_day_band"], observed=False)
        .agg(
            customer_count=("customer_unique_id", "size"),
            same_category_count=("same_category_flag", "sum"),
            same_product_count=("any_same_product_flag", "sum"),
        )
        .reset_index()
    )
    denominators = detail.groupby("first_category").size().rename("category_customer_count")
    timing = timing.merge(denominators, left_on="first_category", right_index=True)
    timing["category_customer_share_pct"] = (
        timing["customer_count"] / timing["category_customer_count"] * 100
    ).round(6)

    product = (
        detail.groupby(["first_category", "product_behavior"], as_index=False)
        .size()
        .rename(columns={"size": "customer_count"})
        .merge(denominators, left_on="first_category", right_index=True)
    )
    product["category_customer_share_pct"] = (
        product["customer_count"] / product["category_customer_count"] * 100
    ).round(6)
    price_rows = []
    for category in PRIORITY_CATEGORIES:
        category_frame = detail.loc[detail["first_category"].eq(category)]
        for scope, scope_frame in [
            ("all_over_24h_followups", category_frame),
            ("same_category_over_24h_followups", category_frame.loc[category_frame["same_category_flag"]]),
        ]:
            counts = scope_frame["price_movement"].value_counts()
            for movement in ["higher", "equal", "lower"]:
                count = int(counts.get(movement, 0))
                price_rows.append({
                    "first_category": category,
                    "analysis_scope": scope,
                    "price_movement": movement,
                    "customer_count": count,
                    "scope_customer_count": len(scope_frame),
                    "scope_customer_share_pct": count / len(scope_frame) * 100,
                })
    price = pd.DataFrame(price_rows)
    price["scope_customer_share_pct"] = price["scope_customer_share_pct"].round(6)
    return summary, timing, product, price


def validate(detail, summary, timing, product, price):
    actual_counts = detail.groupby("first_category").size().to_dict()
    checks = [
        ("detail_customer_count", len(detail), sum(EXPECTED_COUNTS.values()), len(detail) == sum(EXPECTED_COUNTS.values())),
        ("unique_customer_count", detail["customer_unique_id"].nunique(), len(detail), detail["customer_unique_id"].is_unique),
        ("first_order_unique_count", detail["first_order_id"].nunique(), len(detail), detail["first_order_id"].is_unique),
        ("second_order_unique_count", detail["second_order_id"].nunique(), len(detail), detail["second_order_id"].is_unique),
        ("category_counts_match", str(actual_counts), str(EXPECTED_COUNTS), actual_counts == EXPECTED_COUNTS),
        ("invalid_followup_day_count", int((~detail["days_to_second_purchase"].between(1, 30, inclusive="right")).sum()), 0, detail["days_to_second_purchase"].gt(1).all() and detail["days_to_second_purchase"].le(30).all()),
        ("missing_order_feature_count", int(detail[["first_product_ids", "second_product_ids", "first_seller_ids", "second_seller_ids"]].isna().sum().sum()), 0, detail[["first_product_ids", "second_product_ids", "first_seller_ids", "second_seller_ids"]].notna().all().all()),
        ("exact_product_without_overlap_count", int((detail["exact_product_set_match_flag"] & ~detail["any_same_product_flag"]).sum()), 0, (~detail["exact_product_set_match_flag"] | detail["any_same_product_flag"]).all()),
        ("exact_seller_without_overlap_count", int((detail["exact_seller_set_match_flag"] & ~detail["any_same_seller_flag"]).sum()), 0, (~detail["exact_seller_set_match_flag"] | detail["any_same_seller_flag"]).all()),
        ("same_category_behavior_partition_count", int((detail["same_category_different_product_flag"] | detail["any_same_product_flag"]).sum()), int(detail["same_category_flag"].sum()), (detail["same_category_different_product_flag"] | detail["any_same_product_flag"]).eq(detail["same_category_flag"]).all()),
        ("summary_row_count", len(summary), 6, len(summary) == 6),
        ("timing_count_sum", int(timing["customer_count"].sum()), len(detail), timing["customer_count"].sum() == len(detail)),
        ("product_pattern_count_sum", int(product["customer_count"].sum()), len(detail), product["customer_count"].sum() == len(detail)),
        ("price_all_scope_count_sum", int(price.loc[price["analysis_scope"].eq("all_over_24h_followups"), "customer_count"].sum()), len(detail), price.loc[price["analysis_scope"].eq("all_over_24h_followups"), "customer_count"].sum() == len(detail)),
    ]
    return pd.DataFrame(checks, columns=["check_name", "actual_value", "expected_value", "passed"])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    detail = build_detail()
    summary, timing, product, price = build_summaries(detail)
    checks = validate(detail, summary, timing, product, price)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    timing.to_csv(TIMING_PATH, index=False)
    product.to_csv(PRODUCT_PATH, index=False)
    price.to_csv(PRICE_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    print(checks.to_string(index=False))
    print("\nBehavior summary")
    print(summary.to_string(index=False))
    print("\nProduct patterns")
    print(product.to_string(index=False))
    print("\nTiming")
    print(timing.to_string(index=False))
    print("\nPrice movement")
    print(price.to_string(index=False))


if __name__ == "__main__":
    main()

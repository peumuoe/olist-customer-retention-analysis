"""Build Olist first-to-second category transitions for 30-day repeat customers."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

MART_PATH = PROCESSED / "olist_first_purchase_customer_mart.csv"
ORDERS_PATH = PROCESSED / "retention_orders.csv"
ITEMS_PATH = RAW / "olist_order_items_dataset.csv"
PRODUCTS_PATH = RAW / "olist_products_dataset.csv"
TRANSLATION_PATH = RAW / "product_category_name_translation.csv"

CUSTOMER_PATH = PROCESSED / "olist_customer_30d_category_transition.csv"
PAIR_PATH = TABLES / "olist_30d_category_transition_pairs.csv"
SUMMARY_PATH = TABLES / "olist_30d_category_transition_summary.csv"
PRIORITY_PATH = TABLES / "olist_30d_priority_category_transition_candidates.csv"
CHECK_PATH = TABLES / "olist_30d_category_transition_build_summary.csv"
TIMING_PATH = TABLES / "olist_30d_category_transition_timing_sensitivity.csv"
FOLLOWUP_PATH = TABLES / "olist_30d_over_24h_category_transition_pairs.csv"

EXPECTED_ELIGIBLE = 86_772
EXPECTED_REPEAT = 1_373
PRIORITY_CATEGORIES = {
    "cool_stuff", "watches_gifts", "office_furniture", "auto", "toys"
}


def representative_categories(order_ids: set[str]) -> pd.DataFrame:
    """Choose one order category by item count, breaking ties alphabetically."""
    items = pd.read_csv(
        ITEMS_PATH, usecols=["order_id", "product_id"], dtype="string"
    )
    items = items.loc[items["order_id"].isin(order_ids)]
    products = pd.read_csv(
        PRODUCTS_PATH,
        usecols=["product_id", "product_category_name"],
        dtype="string",
    )
    translation = pd.read_csv(TRANSLATION_PATH, dtype="string")
    items = items.merge(products, on="product_id", how="left", validate="many_to_one")
    items = items.merge(
        translation, on="product_category_name", how="left", validate="many_to_one"
    )
    items["category"] = (
        items["product_category_name_english"]
        .fillna(items["product_category_name"])
        .fillna("unknown")
    )
    counts = (
        items.groupby(["order_id", "category"], as_index=False)
        .size()
        .rename(columns={"size": "category_item_count"})
    )
    counts = counts.sort_values(
        ["order_id", "category_item_count", "category"],
        ascending=[True, False, True],
    )
    representative = counts.drop_duplicates("order_id").rename(
        columns={"category": "representative_category"}
    )
    category_count = counts.groupby("order_id").size().rename("category_count")
    return representative.merge(category_count, on="order_id", validate="one_to_one")


def evidence_tier(count: int) -> str:
    if count >= 20:
        return "exploratory_signal_count_20_plus"
    if count >= 5:
        return "small_sample_count_5_to_19"
    return "very_sparse_count_below_5"


def build():
    mart = pd.read_csv(MART_PATH, low_memory=False)
    eligible = mart.loc[mart["eligible_30d"].eq(True)].copy()
    repeaters = eligible.loc[eligible["repeat_within_30d"].eq(True)].copy()

    orders = pd.read_csv(ORDERS_PATH, low_memory=False)
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="raise"
    )
    second = orders.loc[
        orders["customer_order_number"].eq(2),
        ["customer_unique_id", "order_id", "order_purchase_timestamp"],
    ].rename(columns={
        "order_id": "second_order_id",
        "order_purchase_timestamp": "second_purchase_timestamp",
    })
    customer = repeaters[[
        "customer_unique_id", "order_id", "order_purchase_timestamp",
        "primary_category", "category_count",
    ]].rename(columns={
        "order_id": "first_order_id",
        "order_purchase_timestamp": "first_purchase_timestamp",
        "primary_category": "first_category",
        "category_count": "first_category_count",
    }).merge(second, on="customer_unique_id", how="left", validate="one_to_one")
    customer["first_purchase_timestamp"] = pd.to_datetime(
        customer["first_purchase_timestamp"], errors="raise"
    )
    customer["days_to_second_purchase"] = (
        customer["second_purchase_timestamp"] - customer["first_purchase_timestamp"]
    ).dt.total_seconds() / 86_400

    all_ids = set(customer["first_order_id"]) | set(customer["second_order_id"].dropna())
    categories = representative_categories(all_ids)
    first_categories = categories.rename(columns={
        "order_id": "first_order_id",
        "representative_category": "recomputed_first_category",
        "category_item_count": "first_primary_category_item_count",
        "category_count": "recomputed_first_category_count",
    })
    second_categories = categories.rename(columns={
        "order_id": "second_order_id",
        "representative_category": "second_category",
        "category_item_count": "second_primary_category_item_count",
        "category_count": "second_category_count",
    })
    customer = customer.merge(first_categories, on="first_order_id", validate="one_to_one")
    customer = customer.merge(second_categories, on="second_order_id", validate="one_to_one")
    customer["first_category_match_flag"] = customer["first_category"].eq(
        customer["recomputed_first_category"]
    )
    customer["same_category_flag"] = customer["first_category"].eq(
        customer["second_category"]
    )
    customer["first_order_multi_category_flag"] = customer[
        "first_category_count"
    ].gt(1)
    customer["second_order_multi_category_flag"] = customer[
        "second_category_count"
    ].gt(1)
    customer["followup_timing_band"] = pd.cut(
        customer["days_to_second_purchase"],
        bins=[-0.000001, 0, 1 / 24, 1, 7, 30],
        labels=[
            "same_timestamp", "over_0_to_1_hour", "over_1_to_24_hours",
            "over_1_to_7_days", "over_7_to_30_days",
        ],
        include_lowest=True,
    )

    eligible_by_first = eligible.groupby("primary_category").size().rename(
        "first_category_eligible_customers"
    )
    repeat_by_first = customer.groupby("first_category").size().rename(
        "first_category_repeat_customers"
    )
    second_base = customer.groupby("second_category").size().rename(
        "overall_second_category_customers"
    )
    pairs = (
        customer.groupby(["first_category", "second_category"], as_index=False)
        .agg(
            transition_customer_count=("customer_unique_id", "size"),
            median_days_to_second_purchase=("days_to_second_purchase", "median"),
            same_category_flag=("same_category_flag", "first"),
        )
        .merge(eligible_by_first, left_on="first_category", right_index=True)
        .merge(repeat_by_first, left_on="first_category", right_index=True)
        .merge(second_base, left_on="second_category", right_index=True)
    )
    pairs["conversion_rate_among_all_first_category_pct"] = (
        pairs["transition_customer_count"]
        / pairs["first_category_eligible_customers"] * 100
    )
    pairs["transition_share_among_repeaters_pct"] = (
        pairs["transition_customer_count"]
        / pairs["first_category_repeat_customers"] * 100
    )
    pairs["overall_second_category_share_pct"] = (
        pairs["overall_second_category_customers"] / len(customer) * 100
    )
    pairs["lift_vs_overall_second_category"] = (
        pairs["transition_share_among_repeaters_pct"]
        / pairs["overall_second_category_share_pct"]
    )
    pairs["evidence_tier"] = pairs["transition_customer_count"].map(evidence_tier)
    pairs["priority_first_category_flag"] = pairs["first_category"].isin(
        PRIORITY_CATEGORIES
    )
    pairs["rank_within_first_category"] = pairs.groupby("first_category")[
        "transition_customer_count"
    ].rank(method="min", ascending=False).astype(int)

    summary = customer.groupby("first_category", as_index=False).agg(
        repeat_customer_count=("customer_unique_id", "size"),
        same_category_repeat_count=("same_category_flag", "sum"),
        distinct_second_category_count=("second_category", "nunique"),
        median_days_to_second_purchase=("days_to_second_purchase", "median"),
        first_multi_category_count=("first_order_multi_category_flag", "sum"),
        second_multi_category_count=("second_order_multi_category_flag", "sum"),
    ).merge(eligible_by_first, left_on="first_category", right_index=True)
    summary["repeat_rate_pct"] = (
        summary["repeat_customer_count"] / summary["first_category_eligible_customers"] * 100
    )
    summary["same_category_share_among_repeaters_pct"] = (
        summary["same_category_repeat_count"] / summary["repeat_customer_count"] * 100
    )
    summary["cross_category_repeat_count"] = (
        summary["repeat_customer_count"] - summary["same_category_repeat_count"]
    )
    summary["cross_category_share_among_repeaters_pct"] = (
        summary["cross_category_repeat_count"] / summary["repeat_customer_count"] * 100
    )
    summary["priority_first_category_flag"] = summary["first_category"].isin(
        PRIORITY_CATEGORIES
    )

    priority = pairs.loc[
        pairs["priority_first_category_flag"]
        & pairs["rank_within_first_category"].le(5)
    ].sort_values(["first_category", "rank_within_first_category", "second_category"])

    timing = customer.groupby("followup_timing_band", observed=False).agg(
        transition_customer_count=("customer_unique_id", "size"),
        same_category_count=("same_category_flag", "sum"),
    ).reset_index()
    timing["share_of_30d_transitions_pct"] = (
        timing["transition_customer_count"] / len(customer) * 100
    )
    timing["same_category_share_pct"] = (
        timing["same_category_count"] / timing["transition_customer_count"] * 100
    )

    followup_customer = customer.loc[customer["days_to_second_purchase"].gt(1)].copy()
    followup_by_first = followup_customer.groupby("first_category").size().rename(
        "first_category_over_24h_followup_customers"
    )
    followup_second_base = followup_customer.groupby("second_category").size().rename(
        "overall_over_24h_second_category_customers"
    )
    followup = (
        followup_customer.groupby(["first_category", "second_category"], as_index=False)
        .agg(
            transition_customer_count=("customer_unique_id", "size"),
            median_days_to_second_purchase=("days_to_second_purchase", "median"),
            same_category_flag=("same_category_flag", "first"),
        )
        .merge(eligible_by_first, left_on="first_category", right_index=True)
        .merge(followup_by_first, left_on="first_category", right_index=True)
        .merge(followup_second_base, left_on="second_category", right_index=True)
    )
    followup["conversion_rate_among_all_first_category_pct"] = (
        followup["transition_customer_count"]
        / followup["first_category_eligible_customers"] * 100
    )
    followup["transition_share_among_over_24h_followups_pct"] = (
        followup["transition_customer_count"]
        / followup["first_category_over_24h_followup_customers"] * 100
    )
    followup["overall_over_24h_second_category_share_pct"] = (
        followup["overall_over_24h_second_category_customers"]
        / len(followup_customer) * 100
    )
    followup["lift_vs_overall_over_24h_second_category"] = (
        followup["transition_share_among_over_24h_followups_pct"]
        / followup["overall_over_24h_second_category_share_pct"]
    )
    followup["evidence_tier"] = followup["transition_customer_count"].map(evidence_tier)
    followup["priority_first_category_flag"] = followup["first_category"].isin(
        PRIORITY_CATEGORIES
    )
    followup["rank_within_first_category"] = followup.groupby("first_category")[
        "transition_customer_count"
    ].rank(method="min", ascending=False).astype(int)

    numeric_cols = [
        "median_days_to_second_purchase",
        "conversion_rate_among_all_first_category_pct",
        "transition_share_among_repeaters_pct",
        "overall_second_category_share_pct",
        "lift_vs_overall_second_category",
    ]
    pairs[numeric_cols] = pairs[numeric_cols].round(6)
    priority[numeric_cols] = priority[numeric_cols].round(6)
    summary[[
        "median_days_to_second_purchase", "repeat_rate_pct",
        "same_category_share_among_repeaters_pct",
        "cross_category_share_among_repeaters_pct",
    ]] = summary[[
        "median_days_to_second_purchase", "repeat_rate_pct",
        "same_category_share_among_repeaters_pct",
        "cross_category_share_among_repeaters_pct",
    ]].round(6)
    timing[["share_of_30d_transitions_pct", "same_category_share_pct"]] = timing[[
        "share_of_30d_transitions_pct", "same_category_share_pct"
    ]].round(6)
    followup_numeric = [
        "median_days_to_second_purchase",
        "conversion_rate_among_all_first_category_pct",
        "transition_share_among_over_24h_followups_pct",
        "overall_over_24h_second_category_share_pct",
        "lift_vs_overall_over_24h_second_category",
    ]
    followup[followup_numeric] = followup[followup_numeric].round(6)
    return len(eligible), customer, pairs, summary, priority, timing, followup


def validate(eligible_count, customer, pairs, summary, timing, followup) -> pd.DataFrame:
    checks = [
        ("eligible_customer_count", eligible_count, EXPECTED_ELIGIBLE, eligible_count == EXPECTED_ELIGIBLE),
        ("transition_customer_count", len(customer), EXPECTED_REPEAT, len(customer) == EXPECTED_REPEAT),
        ("unique_transition_customer_count", customer["customer_unique_id"].nunique(), EXPECTED_REPEAT, customer["customer_unique_id"].is_unique),
        ("missing_second_order_count", int(customer["second_order_id"].isna().sum()), 0, customer["second_order_id"].notna().all()),
        ("invalid_days_to_second_count", int((~customer["days_to_second_purchase"].between(0, 30)).sum()), 0, customer["days_to_second_purchase"].between(0, 30).all()),
        ("first_category_mismatch_count", int((~customer["first_category_match_flag"]).sum()), 0, customer["first_category_match_flag"].all()),
        ("pair_count_sum", int(pairs["transition_customer_count"].sum()), EXPECTED_REPEAT, pairs["transition_customer_count"].sum() == EXPECTED_REPEAT),
        ("summary_count_sum", int(summary["repeat_customer_count"].sum()), EXPECTED_REPEAT, summary["repeat_customer_count"].sum() == EXPECTED_REPEAT),
        ("timing_count_sum", int(timing["transition_customer_count"].sum()), EXPECTED_REPEAT, timing["transition_customer_count"].sum() == EXPECTED_REPEAT),
        ("over_24h_followup_pair_count_sum", int(followup["transition_customer_count"].sum()), int(customer["days_to_second_purchase"].gt(1).sum()), followup["transition_customer_count"].sum() == customer["days_to_second_purchase"].gt(1).sum()),
        ("priority_category_count", int(summary.loc[summary["priority_first_category_flag"], "first_category"].nunique()), len(PRIORITY_CATEGORIES), set(summary.loc[summary["priority_first_category_flag"], "first_category"]) == PRIORITY_CATEGORIES),
    ]
    return pd.DataFrame(checks, columns=["check_name", "actual_value", "expected_value", "passed"])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    eligible_count, customer, pairs, summary, priority, timing, followup = build()
    checks = validate(eligible_count, customer, pairs, summary, timing, followup)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_dict("records"))
    customer.to_csv(CUSTOMER_PATH, index=False)
    pairs.sort_values(
        ["first_category", "transition_customer_count", "second_category"],
        ascending=[True, False, True],
    ).to_csv(PAIR_PATH, index=False)
    summary.sort_values("repeat_customer_count", ascending=False).to_csv(SUMMARY_PATH, index=False)
    priority.to_csv(PRIORITY_PATH, index=False)
    timing.to_csv(TIMING_PATH, index=False)
    followup.sort_values(
        ["first_category", "transition_customer_count", "second_category"],
        ascending=[True, False, True],
    ).to_csv(FOLLOWUP_PATH, index=False)
    checks.to_csv(CHECK_PATH, index=False)
    print(checks.to_string(index=False))
    print("\nPriority category summary")
    print(summary.loc[summary["priority_first_category_flag"]].to_string(index=False))
    print("\nPriority transition candidates")
    print(priority[[
        "first_category", "second_category", "transition_customer_count",
        "transition_share_among_repeaters_pct", "lift_vs_overall_second_category",
        "evidence_tier", "rank_within_first_category",
    ]].to_string(index=False))
    print("\nTiming sensitivity")
    print(timing.to_string(index=False))
    print("\nOver-24-hour priority transitions")
    print(followup.loc[followup["priority_first_category_flag"], [
        "first_category", "second_category", "transition_customer_count",
        "transition_share_among_over_24h_followups_pct",
        "lift_vs_overall_over_24h_second_category", "evidence_tier",
    ]].sort_values(
        ["first_category", "transition_customer_count"], ascending=[True, False]
    ).to_string(index=False))


if __name__ == "__main__":
    main()

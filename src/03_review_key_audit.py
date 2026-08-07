from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    reviews_path = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"

    reviews = pd.read_csv(
        reviews_path,
        low_memory=False,
    )

    # review_id별 행 수와 연결된 주문 수
    review_id_summary = (
        reviews.groupby("review_id", as_index=False)
        .agg(
            row_count=("review_id", "size"),
            unique_order_count=("order_id", "nunique"),
            first_order_id=("order_id", "first"),
        )
    )

    duplicated_review_ids = review_id_summary.loc[
        review_id_summary["row_count"] > 1
    ].copy()

    # review_id + order_id 조합의 고유성 확인
    composite_duplicate_rows = int(
        reviews.duplicated(
            subset=["review_id", "order_id"],
            keep=False,
        ).sum()
    )

    # 주문별 리뷰 행 수 확인
    order_review_summary = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_row_count=("review_id", "size"),
            unique_review_id_count=("review_id", "nunique"),
        )
    )

    multiple_review_orders = order_review_summary.loc[
        order_review_summary["review_row_count"] > 1
    ].copy()

    overall_summary = pd.DataFrame(
        [
            {
                "check_name": "total_review_rows",
                "value": len(reviews),
            },
            {
                "check_name": "unique_review_ids",
                "value": reviews["review_id"].nunique(),
            },
            {
                "check_name": "duplicated_review_id_count",
                "value": len(duplicated_review_ids),
            },
            {
                "check_name": "review_ids_linked_to_multiple_orders",
                "value": int(
                    (
                        duplicated_review_ids["unique_order_count"] > 1
                    ).sum()
                ),
            },
            {
                "check_name": "duplicate_review_id_order_rows",
                "value": composite_duplicate_rows,
            },
            {
                "check_name": "orders_with_multiple_review_rows",
                "value": len(multiple_review_orders),
            },
        ]
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "review_key_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duplicated_review_ids.to_csv(
        OUTPUT_DIR / "duplicated_review_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    multiple_review_orders.to_csv(
        OUTPUT_DIR / "multiple_review_orders.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[리뷰 키 상세 검증]")
    print(overall_summary.to_string(index=False))

    print("\n저장 완료:")
    print("- review_key_overall_summary.csv")
    print("- duplicated_review_ids.csv")
    print("- multiple_review_orders.csv")


if __name__ == "__main__":
    main()
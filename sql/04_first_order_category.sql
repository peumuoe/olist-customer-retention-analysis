-- 04. 첫 주문 카테고리별 재구매 (발표 2.3)
-- 주문 대표 카테고리 = 상품 수가 가장 많은 카테고리, 동률이면 알파벳 순 (src/23과 같은 규칙)
-- 재구매 = 24시간 이후 ~ 30일 안의 두 번째 주문 (같은 날 재결제 제외)

CREATE OR REPLACE TABLE order_representative_category AS
WITH item_category AS (
    SELECT
        i.order_id,
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category
    FROM raw_order_items AS i
    LEFT JOIN raw_products AS p ON i.product_id = p.product_id
    LEFT JOIN raw_category_translation AS t ON p.product_category_name = t.product_category_name
),
category_counts AS (
    SELECT order_id, category, COUNT(*) AS category_item_count
    FROM item_category
    GROUP BY order_id, category
)
SELECT order_id, category
FROM category_counts
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY order_id
    ORDER BY category_item_count DESC, category ASC
) = 1;

CREATE OR REPLACE TABLE first_category_repeat AS
WITH cutoff AS (
    SELECT MAX(order_purchase_timestamp) AS observation_cutoff FROM retention_orders
),
eligible AS (
    SELECT
        c.customer_unique_id,
        COALESCE(f.category, 'unknown') AS first_category,
        s.category AS second_category,
        c.days_to_second_purchase
    FROM customer_first_second AS c
    CROSS JOIN cutoff AS k
    LEFT JOIN order_representative_category AS f ON c.first_order_id = f.order_id
    LEFT JOIN order_representative_category AS s ON c.second_order_id = s.order_id
    WHERE c.first_purchase_timestamp <= k.observation_cutoff - INTERVAL 30 DAY
)
SELECT
    first_category,
    COUNT(*) AS eligible_customers,
    SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END)
        AS over_24h_repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END)
        / COUNT(*), 2) AS over_24h_repeat_rate_pct,
    SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30
              AND second_category = first_category THEN 1 ELSE 0 END) AS same_category_repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30
              AND second_category = first_category THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END), 0), 1)
        AS same_category_share_pct
FROM eligible
GROUP BY first_category
ORDER BY eligible_customers DESC;

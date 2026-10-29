-- 06. 구매 간격 분석 (LAG)
-- 같은 고객의 직전 주문과 현재 주문 사이 시간을 구해, 몇 번째 구매로 넘어갈 때 간격이 어떻게 달라지는지 본다.
-- 첫 → 두 번째 구매의 30%가 24시간 안이라면 '나눠 결제', 그 이후 구매는 '진짜 재방문'일 가능성이 크다.

CREATE OR REPLACE TABLE purchase_interval_by_step AS
WITH ordered AS (
    SELECT
        customer_unique_id,
        customer_order_number,
        order_purchase_timestamp,
        LAG(order_purchase_timestamp) OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp, order_id
        ) AS previous_purchase_timestamp
    FROM retention_orders
),
intervals AS (
    SELECT
        CASE
            WHEN customer_order_number = 2 THEN '1 → 2'
            WHEN customer_order_number = 3 THEN '2 → 3'
            ELSE '3 → 4+'
        END AS purchase_step,
        (EPOCH(order_purchase_timestamp) - EPOCH(previous_purchase_timestamp)) / 86400.0 AS gap_days
    FROM ordered
    WHERE previous_purchase_timestamp IS NOT NULL
)
SELECT
    purchase_step,
    COUNT(*) AS transitions,
    ROUND(100.0 * AVG(CASE WHEN gap_days <= 1 / 24.0 THEN 1 ELSE 0 END), 1) AS within_1h_pct,
    ROUND(100.0 * AVG(CASE WHEN gap_days > 1 / 24.0 AND gap_days <= 1 THEN 1 ELSE 0 END), 1) AS over_1h_to_24h_pct,
    ROUND(100.0 * AVG(CASE WHEN gap_days > 1 AND gap_days <= 30 THEN 1 ELSE 0 END), 1) AS over_1d_to_30d_pct,
    ROUND(100.0 * AVG(CASE WHEN gap_days > 30 AND gap_days <= 180 THEN 1 ELSE 0 END), 1) AS over_30d_to_180d_pct,
    ROUND(100.0 * AVG(CASE WHEN gap_days > 180 THEN 1 ELSE 0 END), 1) AS over_180d_pct,
    ROUND(MEDIAN(gap_days), 1) AS median_gap_days
FROM intervals
GROUP BY purchase_step
ORDER BY purchase_step;

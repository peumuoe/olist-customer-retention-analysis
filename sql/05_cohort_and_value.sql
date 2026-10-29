-- 05. 첫 구매 월 코호트별 30일 재구매율 + 첫 주문 금액 4분위별 30일 재구매율
-- 코호트: 최근 고객일수록 덜 돌아오는지 확인 (부록 A1)
-- 금액 4분위: 첫 주문 결제 금액 기준 (실험 대상을 금액 상위로 좁힌 근거, 발표 3.3)

CREATE OR REPLACE TABLE monthly_cohort_30d AS
WITH cutoff AS (
    SELECT MAX(order_purchase_timestamp) AS observation_cutoff FROM retention_orders
)
SELECT
    DATE_TRUNC('month', c.first_purchase_timestamp) AS cohort_month,
    COUNT(*) AS eligible_customers,
    SUM(CASE WHEN c.days_to_second_purchase <= 30 THEN 1 ELSE 0 END) AS repeat_30d,
    ROUND(100.0 * SUM(CASE WHEN c.days_to_second_purchase <= 30 THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS repeat_30d_pct
FROM customer_first_second AS c, cutoff AS k
WHERE c.first_purchase_timestamp <= k.observation_cutoff - INTERVAL 30 DAY
GROUP BY 1
HAVING COUNT(*) >= 100
ORDER BY 1;

CREATE OR REPLACE TABLE first_order_value_quartile_30d AS
WITH cutoff AS (
    SELECT MAX(order_purchase_timestamp) AS observation_cutoff FROM retention_orders
),
first_order_value AS (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM raw_payments
    GROUP BY order_id
),
eligible AS (
    SELECT
        c.*,
        v.payment_value,
        NTILE(4) OVER (ORDER BY v.payment_value, c.customer_unique_id) AS value_quartile
    FROM customer_first_second AS c
    CROSS JOIN cutoff AS k
    JOIN first_order_value AS v ON c.first_order_id = v.order_id
    WHERE c.first_purchase_timestamp <= k.observation_cutoff - INTERVAL 30 DAY
)
SELECT
    'Q' || value_quartile AS first_order_value_quartile,
    COUNT(*) AS eligible_customers,
    ROUND(MEDIAN(payment_value), 2) AS median_first_payment,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase <= 30 THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS business_kpi_30d_pct,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END)
        / COUNT(*), 2) AS model_target_over_24h_30d_pct
FROM eligible
GROUP BY value_quartile
ORDER BY value_quartile;

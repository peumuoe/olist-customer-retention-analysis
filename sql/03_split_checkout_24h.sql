-- 03. 24시간 안 재주문(나눠 결제 가능성) 분리 (발표 1.2)
-- (a) 두 번째 주문까지 걸린 시간 분포: 전체 재구매 고객 기준
-- (b) 사업 지표(30일 재주문) vs 예측 모델용 지표(24시간 이후 ~ 30일 재주문)

CREATE OR REPLACE TABLE second_order_timing AS
SELECT
    COUNT(*) AS second_order_customers,
    SUM(CASE WHEN days_to_second_purchase <= 1 THEN 1 ELSE 0 END) AS within_24h_customers,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase <= 1 THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS within_24h_share_pct
FROM customer_first_second
WHERE second_purchase_timestamp IS NOT NULL;

CREATE OR REPLACE TABLE kpi_vs_model_target AS
WITH cutoff AS (
    SELECT MAX(order_purchase_timestamp) AS observation_cutoff FROM retention_orders
),
eligible AS (
    SELECT c.*
    FROM customer_first_second AS c, cutoff AS k
    WHERE c.first_purchase_timestamp <= k.observation_cutoff - INTERVAL 30 DAY
)
SELECT
    COUNT(*) AS eligible_30d_customers,
    SUM(CASE WHEN days_to_second_purchase <= 30 THEN 1 ELSE 0 END) AS repeat_30d,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase <= 30 THEN 1 ELSE 0 END) / COUNT(*), 4)
        AS business_kpi_30d_pct,
    SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END)
        AS repeat_over_24h_30d,
    ROUND(100.0 * SUM(CASE WHEN days_to_second_purchase > 1 AND days_to_second_purchase <= 30 THEN 1 ELSE 0 END)
        / COUNT(*), 4) AS model_target_over_24h_30d_pct
FROM eligible;

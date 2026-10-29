-- 02. 고정 관찰창 누적 재구매율 (발표 2.1: 30일 1.58% … 365일 4.53%)
-- 관찰 마감일 = 배송 완료 주문의 마지막 구매 시각
-- N일 창의 대상 = 첫 구매가 (마감일 - N일) 이전인 고객 → 관찰기간이 끝까지 보장된 고객만 분모에 넣는다.
-- 재구매 = 두 번째 주문이 첫 주문 후 N일 이내

CREATE OR REPLACE TABLE retention_window_summary AS
WITH cutoff AS (
    SELECT MAX(order_purchase_timestamp) AS observation_cutoff
    FROM retention_orders
),
windows AS (
    SELECT UNNEST([30, 60, 90, 180, 365]) AS window_days
)
SELECT
    w.window_days,
    COUNT(*) AS eligible_customer_count,
    SUM(CASE
            WHEN c.second_purchase_timestamp IS NOT NULL
             AND c.second_purchase_timestamp
                 <= c.first_purchase_timestamp + TO_DAYS(CAST(w.window_days AS INTEGER))
            THEN 1 ELSE 0
        END) AS repeat_customer_count,
    ROUND(100.0 * SUM(CASE
            WHEN c.second_purchase_timestamp IS NOT NULL
             AND c.second_purchase_timestamp
                 <= c.first_purchase_timestamp + TO_DAYS(CAST(w.window_days AS INTEGER))
            THEN 1 ELSE 0
        END) / COUNT(*), 4) AS repeat_rate_pct
FROM windows AS w
CROSS JOIN cutoff AS k
JOIN customer_first_second AS c
    ON c.first_purchase_timestamp <= k.observation_cutoff - TO_DAYS(CAST(w.window_days AS INTEGER))
GROUP BY w.window_days
ORDER BY w.window_days;

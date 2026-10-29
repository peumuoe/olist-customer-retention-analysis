-- 01. 재구매 분석용 주문 마트 (src/06_build_retention_orders.py와 같은 정의)
-- 재구매 = 같은 고객(customer_unique_id)의 서로 다른 '배송 완료' 주문
-- 고객별 구매 순서는 구매 시각, 주문 ID 순으로 매긴다.

CREATE OR REPLACE TABLE retention_orders AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_purchase_timestamp,
    ROW_NUMBER() OVER (
        PARTITION BY c.customer_unique_id
        ORDER BY o.order_purchase_timestamp, o.order_id
    ) AS customer_order_number,
    MIN(o.order_purchase_timestamp) OVER (
        PARTITION BY c.customer_unique_id
    ) AS customer_first_purchase_timestamp
FROM raw_orders AS o
JOIN raw_customers AS c
    ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered';

-- 고객 1행: 첫 주문, 두 번째 주문, 두 주문 사이 경과일
CREATE OR REPLACE TABLE customer_first_second AS
SELECT
    customer_unique_id,
    COUNT(*) AS total_order_count,
    MAX(CASE WHEN customer_order_number = 1 THEN order_id END) AS first_order_id,
    MAX(CASE WHEN customer_order_number = 1 THEN order_purchase_timestamp END) AS first_purchase_timestamp,
    MAX(CASE WHEN customer_order_number = 2 THEN order_id END) AS second_order_id,
    MAX(CASE WHEN customer_order_number = 2 THEN order_purchase_timestamp END) AS second_purchase_timestamp
FROM retention_orders
GROUP BY customer_unique_id;

CREATE OR REPLACE TABLE customer_first_second AS
SELECT
    *,
    (EPOCH(second_purchase_timestamp) - EPOCH(first_purchase_timestamp)) / 86400.0
        AS days_to_second_purchase
FROM customer_first_second;

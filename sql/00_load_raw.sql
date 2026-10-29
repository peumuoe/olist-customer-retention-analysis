-- 00. 원천 CSV를 뷰로 등록
-- 실행: python sql/run_sql.py  (DuckDB, 프로젝트 루트 기준 상대 경로)

CREATE OR REPLACE VIEW raw_orders AS
SELECT
    order_id,
    customer_id,
    order_status,
    CAST(order_purchase_timestamp AS TIMESTAMP) AS order_purchase_timestamp
FROM read_csv_auto('data/raw/olist_orders_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_customers AS
SELECT customer_id, customer_unique_id
FROM read_csv_auto('data/raw/olist_customers_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_order_items AS
SELECT order_id, product_id
FROM read_csv_auto('data/raw/olist_order_items_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_products AS
SELECT product_id, product_category_name
FROM read_csv_auto('data/raw/olist_products_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_category_translation AS
SELECT product_category_name, product_category_name_english
FROM read_csv_auto('data/raw/product_category_name_translation.csv', header = TRUE, all_varchar = TRUE);

-- 쿼리 최적화 예시(sql/optimization)용
CREATE OR REPLACE VIEW raw_customers_geo AS
SELECT customer_id, customer_unique_id, customer_zip_code_prefix, customer_state
FROM read_csv_auto('data/raw/olist_customers_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_geolocation AS
SELECT
    geolocation_zip_code_prefix,
    CAST(geolocation_lat AS DOUBLE) AS geolocation_lat,
    CAST(geolocation_lng AS DOUBLE) AS geolocation_lng
FROM read_csv_auto('data/raw/olist_geolocation_dataset.csv', header = TRUE, all_varchar = TRUE);

CREATE OR REPLACE VIEW raw_payments AS
SELECT order_id, CAST(payment_value AS DOUBLE) AS payment_value
FROM read_csv_auto('data/raw/olist_order_payments_dataset.csv', header = TRUE, all_varchar = TRUE);

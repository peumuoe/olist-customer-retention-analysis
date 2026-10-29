-- 최적화 후: 위치 테이블을 우편번호 앞자리별로 먼저 합계·건수로 줄인 뒤 조인한다.
-- 평균을 다시 '합계 / 건수'로 계산하므로, 최적화 전과 결과가 정확히 같다.
WITH geo_by_zip AS (
    SELECT
        geolocation_zip_code_prefix,
        SUM(geolocation_lat) AS lat_sum,
        SUM(geolocation_lng) AS lng_sum,
        COUNT(*) AS geo_rows
    FROM raw_geolocation
    GROUP BY geolocation_zip_code_prefix
)
SELECT
    c.customer_state,
    COUNT(DISTINCT c.customer_unique_id) AS customers,
    ROUND(SUM(z.lat_sum) / SUM(z.geo_rows), 4) AS avg_lat,
    ROUND(SUM(z.lng_sum) / SUM(z.geo_rows), 4) AS avg_lng
FROM raw_customers_geo AS c
JOIN geo_by_zip AS z
    ON c.customer_zip_code_prefix = z.geolocation_zip_code_prefix
GROUP BY c.customer_state
ORDER BY c.customer_state

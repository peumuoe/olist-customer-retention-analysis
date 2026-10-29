-- 최적화 전: 원본 위치 테이블(약 100만 행)을 고객과 먼저 조인한 뒤 평균을 낸다.
-- 같은 우편번호 앞자리에 위치 행이 수십 개씩 있어, 조인 단계에서 행이 수백만 개로 불어난다.
SELECT
    c.customer_state,
    COUNT(DISTINCT c.customer_unique_id) AS customers,
    ROUND(AVG(g.geolocation_lat), 4) AS avg_lat,
    ROUND(AVG(g.geolocation_lng), 4) AS avg_lng
FROM raw_customers_geo AS c
JOIN raw_geolocation AS g
    ON c.customer_zip_code_prefix = g.geolocation_zip_code_prefix
GROUP BY c.customer_state
ORDER BY c.customer_state

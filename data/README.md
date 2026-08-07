# Data availability

기준일: 2026-08-07

## GitHub에 포함하는 데이터

`portfolio_data/`에는 Tableau, Power BI, Looker Studio용 공개 집계 데이터만 포함합니다.

- 고객·주문·판매자·리뷰 식별자 없음
- 개인 단위 모델 점수 없음
- 외부 비교의 인과 해석 제한 포함
- CRM 수용량의 비증분 관찰값 경고 포함
- 파일별 행·열·SHA-256 manifest 포함

## GitHub에 포함하지 않는 데이터

- `raw/`: Olist 원본 CSV
- `external/`: H&M·REES46 원본과 대용량 외부 데이터
- `processed/`: 고객·주문 단위 분석 마트와 모델 점수

원본 데이터는 파일 크기와 재배포·라이선스 관리 때문에 저장소에 커밋하지 않습니다. 프로젝트 코드는 사용자가 합법적으로 내려받은 원본 데이터에서 분석 산출물을 재생성하는 구조로 정리합니다.

## BI 데이터 사전

- 데이터셋 안내: `portfolio_data/README.md`
- 컬럼 단위 사전: `portfolio_data/data_dictionary.csv`
- 파일 manifest: `portfolio_data/data_manifest.csv`
- 상세 모델링 원칙: `../docs/portfolio_bi_data_dictionary.md`

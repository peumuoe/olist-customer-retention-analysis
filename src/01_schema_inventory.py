from pathlib import Path

import pandas as pd


# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """CSV 파일을 기본 인코딩으로 읽고, 실패하면 대체 인코딩을 사용합니다."""

    try:
        return pd.read_csv(file_path, low_memory=False)

    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"파일을 읽지 못했습니다: {file_path.name}"
        ) from error


def get_sample_values(series: pd.Series, sample_size: int = 3) -> str:
    """컬럼의 대표값을 최대 3개까지 문자열로 반환합니다."""

    sample_values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .head(sample_size)
        .tolist()
    )

    return " | ".join(sample_values)


def create_schema_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """파일별 요약과 컬럼별 구조 정보를 생성합니다."""

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {RAW_DATA_DIR}"
        )

    file_records: list[dict[str, object]] = []
    column_records: list[dict[str, object]] = []

    for file_path in csv_files:
        df = read_csv_safely(file_path)

        file_records.append(
            {
                "file_name": file_path.name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "duplicated_row_count": int(df.duplicated().sum()),
                "total_missing_count": int(df.isna().sum().sum()),
            }
        )

        for column_order, column_name in enumerate(df.columns, start=1):
            series = df[column_name]

            column_records.append(
                {
                    "file_name": file_path.name,
                    "column_order": column_order,
                    "column_name": column_name,
                    "dtype": str(series.dtype),
                    "row_count": len(df),
                    "missing_count": int(series.isna().sum()),
                    "missing_rate_pct": round(
                        series.isna().mean() * 100,
                        2,
                    ),
                    "unique_count": int(
                        series.nunique(dropna=False)
                    ),
                    "sample_values": get_sample_values(series),
                }
            )

    file_summary = pd.DataFrame(file_records)
    schema_inventory = pd.DataFrame(column_records)

    return file_summary, schema_inventory


def main() -> None:
    file_summary, schema_inventory = create_schema_inventory()

    file_summary_path = OUTPUT_DIR / "file_summary.csv"
    schema_inventory_path = OUTPUT_DIR / "schema_inventory.csv"

    file_summary.to_csv(
        file_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    schema_inventory.to_csv(
        schema_inventory_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[파일별 요약]")
    print(file_summary.to_string(index=False))

    print("\n[컬럼별 구조]")
    print(schema_inventory.to_string(index=False))

    print("\n저장 완료")
    print(f"- {file_summary_path}")
    print(f"- {schema_inventory_path}")


if __name__ == "__main__":
    main()
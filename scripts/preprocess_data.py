"""
다이캐스팅 공정 데이터를 전처리하는 스크립트

1. 입력 데이터 형식:
   - 31개의 공정 변수 컬럼
   - 결함 유형별 컬럼들 (defect_1, defect_2, ...)

2. 전처리 과정:
   1. 결함 유형 컬럼들을 하나의 defect 컬럼으로 통합(defect 컬럼은 어떤 결함이라도 있으면 1, 없으면 0)
   2. 결측값 제거
   3. 표준편차가 0인 컬럼 제거

실행 방법:
```
python scripts/preprocess_data.py --input-path data/DieCasting_Quality_Raw_Data.csv --output-path data/DieCasting_Quality_Data_processed.csv
```
"""

from pathlib import Path

import pandas as pd
import typer
from typer import Option


def main(
    input_path: Path = Option(
        "data/DieCasting_Quality_Raw_Data.csv",
        "--input-path",
        help="파일을 저장할 경로",
    ),
    output_path: Path = Option(
        "data/DieCasting_Quality_Data_processed.csv",
        "--output-path",
        help="파일을 저장할 경로",
    ),
):
    print("데이터 불러오기")
    df = pd.read_csv(input_path, skiprows=1)

    print("전처리 실행")
    df["defect"] = df[df.columns[31:]].max(axis=1).astype(bool).astype(int)
    df = df[df.columns[:31].tolist() + ["defect"]]

    missing_rows = df.isnull().sum(axis=1)
    df = df[missing_rows == 0]

    zero_std_cols = (
        df.describe().loc["std"][df.describe().loc["std"] == 0].index.tolist()
    )
    df = df.drop(columns=zero_std_cols)

    print("데이터 저장")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)

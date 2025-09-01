"""
실험 2에서 사용할 데이터 분할
- id가 6000000보다 작거나 같은 데이터를 훈련 데이터로 사용
- id가 6000000보다 큰 데이터를 테스트 데이터로 사용

실행 방법:
```
python notebooks/exp02/create_train_test.py
```
"""

from pathlib import Path

import pandas as pd
import typer
from typer import Option


def main(
    input_path: Path = Option(
        "data/DieCasting_Quality_Data_processed.csv",
        "--input-path",
        help="전처리 데이터 파일 경로",
    ),
    output_dir: Path = Option(
        "data/exp02",
        "--output-dir",
        help="파일을 저장할 경로",
    ),
):
    df = pd.read_csv(input_path)
    df_train = df[df["id"] <= 6000000]
    df_test = df[df["id"] > 6000000]

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / "train.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

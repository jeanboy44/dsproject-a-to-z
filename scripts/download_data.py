"""
구글 드라이브에서 원천 데이터를 다운로드하는 스크립트

실행 방법:
```
python scripts/download_data.py --url https://drive.google.com/file/d/1Bh_Q1WROysdeF25BazhCkddKs-8bjJCi/view?usp=sharing --output-path data/DieCasting_Quality_Raw_Data.csv
```
"""

from pathlib import Path

import gdown
import typer
from typer import Option


def main(
    url: str = Option(..., "--url", help="구글 드라이브 파일의 URL"),
    output_path: Path = Option(..., "--output-path", help="파일을 저장할 경로"),
):
    print(f"저장 경로: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"구글 드라이브에서 파일 다운로드 중: {url}")
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        print("다운로드 성공!")
    except Exception as e:
        print(f"다운로드 중 오류 발생: {e}")


if __name__ == "__main__":
    typer.run(main)

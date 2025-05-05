import os
from pathlib import Path

import gdown
import typer
from typer import Option


def main(
    url: str = Option(..., "--url", help="구글 드라이브 파일의 URL"),
    output_path: Path = Option("data/", "--output-path", help="파일을 저장할 경로"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"구글 드라이브에서 파일 다운로드 중: {url}")
    print(f"저장 경로: {output_path}")

    try:
        # Download the file using gdown
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        print("다운로드 성공!")
    except Exception as e:
        print(f"다운로드 중 오류 발생: {e}")


if __name__ == "__main__":
    typer.run(main)

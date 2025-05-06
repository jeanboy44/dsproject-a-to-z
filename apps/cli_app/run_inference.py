"""
이 파일은 주기적으로 모델을 로드하고 예측을 수행하는 모델 워치독 애플리케이션입니다.

주요 기능:
1. 모델 로드: 지정된 모델 이름, 별칭 및 추적 URI를 사용하여 MLflow에서 LightGBM 모델을 로드합니다.
2. 예측 수행: 주기적으로 새로운 데이터가 생성되면 예측을 수행합니다.
3. 예측 저장: 예측 결과를 지정된 디렉토리에 저장합니다.

사용 방법 예시:
```bash
python apps/cli_app/run_inference.py
```
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from lightgbm import Booster
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.inference import load_model, preprocess, run_inference, save_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize(
    model_name: str,
    model_alias: str,
    tracking_uri: str,
    src_dir: Path,
    dst_dir: Path,
) -> Optional[Booster]:
    logger.info(f"Initializing model: {model_name}@{model_alias} from {tracking_uri}")
    try:
        model = load_model(model_name, model_alias, tracking_uri)
        logger.info(f"Model initialized: {model}")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        return

    if src_dir.is_dir():
        logger.info(f"Source directory: {src_dir}")
    else:
        logger.error(f"Source directory does not exist: {src_dir}")
        return

    if dst_dir.is_dir():
        logger.info(f"Destination directory: {dst_dir}")
    else:
        logger.error(f"Destination directory does not exist: {dst_dir}")
        return

    if src_dir == dst_dir:
        logger.error("Source and destination directories cannot be the same")
        return

    return model


def run(
    model: Booster,
    src_path: Path,
    dst_path: Path,
):
    logger.info(f"Running inference on '{src_path}'")
    try:
        input_data = pd.read_csv(src_path)
    except Exception as e:
        logger.error(f"Error during input data loading: {e}")
        return

    try:
        processed_data = preprocess(input_data)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    try:
        predictions = run_inference(processed_data, model)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return

    try:
        save_predictions(predictions, dst_path)
    except Exception as e:
        logger.error(f"Error during saving predictions: {e}")
        return

    logger.info(f"Inference completed successfully. Predictions saved to '{dst_path}'")


class ModelEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        model_name: str,
        model_alias: str,
        tracking_uri: str,
        src_dir: Path,
        dst_dir: Path,
    ):
        self.model = initialize(model_name, model_alias, tracking_uri, src_dir, dst_dir)
        self.src_dir = src_dir
        self.dst_dir = dst_dir

    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            dst_path = self.dst_dir / file_path.name
            run(self.model, file_path, dst_path)


if __name__ == "__main__":
    config = load_config("apps/cli_app/config.yaml")
    model_name = config["model_name"]
    model_alias = config["model_alias"]
    tracking_uri = config["tracking_uri"]
    src_dir = Path(config["src_dir"])
    dst_dir = Path(config["dst_dir"])
    reset = config["reset"]

    logger.info(config)

    if reset:
        if src_dir.is_dir():
            shutil.rmtree(src_dir)
        src_dir.mkdir(parents=True, exist_ok=True)

        if dst_dir.is_dir():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

    model_handler = ModelEventHandler(
        model_name, model_alias, tracking_uri, src_dir, dst_dir
    )
    observer = Observer()
    observer.schedule(model_handler, path=src_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from lightgbm import Booster
from typer import Option
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.inference import load_model, preprocess, run_inference, save_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        save_predictions(predictions, dst_path)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
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


def main(
    model_name: str = Option(
        "diecast",
        "--model-name",
        help="The name of the model.",
    ),
    model_alias: str = Option(
        "production",
        "--model-alias",
        help="The alias of the model.",
    ),
    tracking_uri: str = Option(
        "http://localhost:5001",
        "--tracking-uri",
        help="The tracking URI of the model.",
    ),
    src_dir: Path = Option(
        "data/src",
        "--src-dir",
        help="The path to the source directory.",
    ),
    dst_dir: Path = Option(
        "data/dst",
        "--dst-dir",
        help="The path to the destination directory.",
    ),
    reset: bool = Option(
        False,
        "--reset",
        help="Reset the source and destination directories.",
    ),
):
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


if __name__ == "__main__":
    typer.run(main)

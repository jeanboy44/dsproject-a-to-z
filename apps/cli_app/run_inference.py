import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from lightgbm import Booster
from typer import Option

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
    logger.info(f"Running inference on '{src_path}' and saving to '{dst_path}'")
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
):
    model = initialize(model_name, model_alias, tracking_uri, src_dir, dst_dir)
    if not model:
        logger.error("Failed to initialize model")
        return

    seen = set(f.name for f in src_dir.iterdir() if f.is_file())
    while True:
        current = set(f.name for f in src_dir.iterdir() if f.is_file())
        new_files = current - seen
        if new_files:
            for file in new_files:
                run(model, src_dir / file, dst_dir / file)
        seen = current
        time.sleep(1)


if __name__ == "__main__":
    typer.run(main)

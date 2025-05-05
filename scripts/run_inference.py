from pathlib import Path

import pandas as pd
import typer
from typer import Option

from src.inference import load_model, preprocess, run_inference, save_predictions


def main(
    sample_path: Path = Option(
        "data/sample.csv",
        "--sample-path",
        help="The path to the sample data.",
    ),
    output_path: Path = Option(
        "data/predictions.csv",
        "--output-path",
        help="The path to the output file.",
    ),
):
    model = load_model(
        model_name="diecast",
        model_alias="production",
        tracking_uri="http://localhost:5001",
    )
    input_data = pd.read_csv(sample_path)
    processed_data = preprocess(input_data)
    predictions = run_inference(processed_data, model)
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    typer.run(main)

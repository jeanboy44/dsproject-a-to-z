import os
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import Booster


def download_sample_data(url: str, output_path: Path) -> None:
    """Download sample data from MLflow."""
    try:
        print(f"Downloading sample data from {url}...")
        os.makedirs(output_path.parent, exist_ok=True)
        mlflow.artifacts.download_artifacts(artifact_uri=url, dst_path=output_path)
        print(f"Sample data downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading sample data: {e}")
        raise


def load_model(model_name: str, model_alias: str, tracking_uri: str) -> Booster:
    """Load the model from MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.lightgbm.load_model(f"models:/{model_name}@{model_alias}")


def preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
    processed_data = input_data[input_data.columns[:31].tolist()]
    return processed_data


def run_inference(input_data: pd.DataFrame, model: Booster) -> pd.DataFrame:
    """
    Runs inference on the provided DataFrame using the loaded MLflow model.

    Args:
        input_data: The pandas DataFrame to run inference on.
        model: The LightGBM model to use for inference.
    Returns:
        A pandas DataFrame containing the predictions.
    """
    print("Running inference...")
    try:
        predictions = model.predict(input_data)
        print("Inference completed.")
        # Ensure predictions are returned as a DataFrame
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(
                predictions, columns=["predictions"]
            )  # Adjust column name if needed
        return predictions
    except Exception as e:
        print(f"Error during inference: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def save_predictions(predictions: pd.DataFrame, output_path: Path = "predictions.csv"):
    """
    Saves the predictions DataFrame to a CSV file.

    Args:
        predictions: The pandas DataFrame containing predictions.
        output_path: The path to save the CSV file.
    """
    if predictions.empty:
        print("No predictions to save.")
        return

    try:
        print(f"Saving predictions to {output_path}...")
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved successfully to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

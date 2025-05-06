import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.models
import pandas as pd
from icecream import ic
from lightgbm import Booster

ic.configureOutput(
    includeContext=True,
    prefix=lambda: f"{datetime.now().strftime('%H:%M:%S')} | ",
)


def validate_data_schema(
    input_data: pd.DataFrame, model_name: str, model_alias: str, tracking_uri: str
) -> bool:
    """
    Validates the input DataFrame schema against the MLflow model's input signature.

    Args:
        input_data: The pandas DataFrame whose schema needs validation.
        model_name: The name of the model in MLflow registry.
        model_alias: The alias of the model version (e.g., "champion").
        tracking_uri: The MLflow tracking server URI.

    Returns:
        True if the input data schema matches the model signature, False otherwise.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{model_alias}"
    ic(model_uri)
    model_info = mlflow.models.get_model_info(model_uri)

    if not model_info.signature or not model_info.signature.inputs:
        ic(f"Warning: Model signature or input schema not found for {model_uri}.")
        return False
    expected_columns = model_info.signature.inputs.input_names()
    actual_columns = input_data.columns.tolist()

    ic(expected_columns)
    ic(actual_columns)

    # Simple check: Does the input data contain all expected columns?
    # More robust checks could compare types or enforce column order if necessary.
    missing_columns = set(expected_columns) - set(actual_columns)
    extra_columns = set(actual_columns) - set(expected_columns)

    if not missing_columns and not extra_columns:
        ic("Data schema validation successful.")
        return True
    else:
        if missing_columns:
            ic("Validation Failed: Missing columns in input data:")
            ic(missing_columns)
        if extra_columns:
            ic("Validation Failed: Extra columns found in input data:")
            ic(extra_columns)
        return False


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
    predictions = model.predict(input_data)
    # Ensure predictions are returned as a DataFrame
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(
            predictions, columns=["probability"]
        )  # Adjust column name if needed
    return predictions


def save_predictions(predictions: pd.DataFrame, output_path: Path = "predictions.csv"):
    """
    Saves the predictions DataFrame to a CSV file.

    Args:
        predictions: The pandas DataFrame containing predictions.
        output_path: The path to save the CSV file.
    """
    predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_data = pd.read_csv("data/sample.csv")
    validate_data_schema(
        input_data=preprocess(input_data),
        model_name="diecast",
        model_alias="production",
        tracking_uri="http://localhost:5001",
    )

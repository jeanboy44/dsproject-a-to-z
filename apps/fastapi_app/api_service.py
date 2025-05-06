# To run (example):
# PYTHONPATH=. uvicorn apps.fastapi_app.api_service:app --reload --port 8001


import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import load_model, preprocess, run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class InputData(BaseModel):
    """Request body structure."""

    raw: dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "raw": {
                    "id": 3649127,
                    "Product_Type": 1,
                    "Shot": 127,
                    "Velocity_1": 0.142,
                    "Velocity_2": 0.166,
                    "Velocity_3": 0.178,
                    "High_Velocity": 2.129,
                    "Cylinder_Pressure": 239,
                    "Rapid_Rise_Time": 0.008,
                    "Biscuit_Thickness ": 12,
                    "Clamping_Force ": 259,
                    "Cycle_Time": 22.6,
                    " Pressure_Rise_Time": 0.035,
                    "Casting_Pressure": 1156,
                    "Spray_Time": 9.7,
                    "Spray_1_Time": 1.2,
                    "Spray_2_Time": 0.7,
                    "Melting_Furnace_Temp": 726.2,
                    "Air_Pressure": 4.7,
                    "Air_Pressure_Min": 3,
                    "Air_Pressure_Max": 9,
                    "Coolant_Temp": 26.5,
                    "Coolant_Temp_Min": 10,
                    "Coolant_Temp_Max": 50,
                    "Coolant_Pressure": 2.62,
                    "Factory_Temp": 33.0,
                    "Factory_Temp_Min": 18.0,
                    "Factory_Temp_Max": 22.0,
                    "Factory_Humidity": 62.3,
                    "Factory_Humidity_Min": 18.0,
                    "Factory_Humidity_Max": 22.0,
                    "Short_Shot_1": 0,
                    "Bubble_1": 0,
                    "Exfoliation_1": 0,
                    "Blow_Hole_1": 0,
                    "Stain_1": 0,
                    "Dent_1": 0,
                    "Deformation_1": 0,
                    "Contamination_1": 0,
                    "Impurity_1": 0,
                    "Crack_1": 0,
                    "Scratch_1": 0,
                    "Buring_Mark_1": 0,
                    "Inclusions_1": 0,
                    "Short_Shot_2": 0,
                    "Bubble_2": 0,
                    "Exfoliation_2": 0,
                    "Blow_Hole_2": 0,
                    "Stain_2": 0,
                    "Dent_2": 0,
                    "Deformation_2": 0,
                    "Contamination_2": 0,
                    "Impurity_2": 0,
                    "Crack_2": 0,
                    "Scratch_2": 0,
                    "Buring_Mark_2": 0,
                    "Inclusions_2": 0,
                }
            }
        }
    }


class Response(BaseModel):
    """Response body structure."""

    probability: float


# --- Globals ---
TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "diecast"
MODEL_ALIAS = "production"
MODEL = None  # Loaded model

# --- FastAPI App ---
app = FastAPI(title="ML Inference API")

logger.info(f"Attempting to load model: {MODEL_NAME}@{MODEL_ALIAS} from {TRACKING_URI}")
try:
    MODEL = load_model(MODEL_NAME, MODEL_ALIAS, TRACKING_URI)
    logger.info(f"Successfully loaded model: {MODEL_NAME}@{MODEL_ALIAS}")
except Exception as e:
    error_message = f"Fatal error during application startup: Failed to load model '{MODEL_NAME}@{MODEL_ALIAS}' from '{TRACKING_URI}'. Reason: {e}"
    logger.exception(error_message)


@app.post("/predict", response_model=Response)
async def predict(input_data: InputData):
    """Receive data, preprocess, run inference, return predictions."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_NAME}:{MODEL_ALIAS}' is not available.",
        )

    # 1. Convert to DataFrame
    try:
        input_records = [input_data.raw]
        if not input_records:
            raise ValueError("Empty input list.")
        input_df = pd.DataFrame(input_records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating DataFrame: {e}")

    # 2. Preprocess
    try:
        processed_data = preprocess(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during preprocessing: {e}")

    # 3. Run inference
    try:
        predictions_raw = run_inference(processed_data, MODEL)
        probability = predictions_raw["probability"][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

    # 4. Return predictions
    return Response(probability=probability)


@app.get("/", summary="Health Check")
async def read_root():
    """Basic health check."""
    return {
        "message": "Inference API status",
        "model_status": "loaded" if MODEL is not None else "not loaded",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }

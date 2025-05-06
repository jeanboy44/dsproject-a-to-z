from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.models
import pandas as pd
from lightgbm import Booster


def load_model(model_name: str, model_alias: str, tracking_uri: str) -> Booster:
    """지정된 모델 이름, 별칭 및 추적 URI를 사용하여 MLflow에서 LightGBM 모델을 로드합니다.

    Args:
        model_name (str): 로드할 모델의 이름입니다.
        model_alias (str): 로드할 모델의 별칭입니다.
        tracking_uri (str): MLflow 추적 서버의 URI입니다.

    Returns:
        Booster: 로드된 LightGBM 모델입니다.
    """
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.lightgbm.load_model(f"models:/{model_name}@{model_alias}")


def preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
    """입력 데이터프레임을 사전 처리합니다.
    이 함수는 불필요한 열을 제거합니다.

    Args:
        input_data (pd.DataFrame): 사전 처리할 입력 데이터프레임입니다.

    Returns:
        pd.DataFrame: 사전 처리된 데이터프레임입니다.
    """
    drop_cols = ["defect"]
    processed_data = input_data.drop(columns=drop_cols, axis=1)
    return processed_data


def run_inference(input_data: pd.DataFrame, model: Booster) -> pd.DataFrame:
    """입력 데이터에 대해 추론을 실행하고 예측을 반환합니다.

    Args:
        input_data (pd.DataFrame): 추론을 실행할 입력 데이터입니다.
        model (Booster): 추론에 사용할 사전 훈련된 LightGBM 모델입니다.

    Returns:
        pd.DataFrame: "probability" 열이 있는 예측이 포함된 데이터프레임입니다.
    """
    predictions = model.predict(input_data)
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions, columns=["probability"])
    return predictions


def save_predictions(
    predictions: pd.DataFrame, output_path: Path = "predictions.csv"
) -> None:
    """예측 데이터프레임을 CSV 파일에 저장합니다.

    Args:
        predictions (pd.DataFrame): 저장할 예측이 포함된 데이터프레임입니다.
        output_path (Path, optional): 예측을 저장할 경로입니다.
            기본값은 "predictions.csv"입니다.
    """
    predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    model = load_model("diecasting_base", "production", "http://localhost:5001")
    print(model)
    df = pd.read_csv("data/exp01/test.csv").sample(n=1, random_state=42)
    df.to_csv("data/sample.csv", index=False)
    print(df)
    df_prep = preprocess(df)
    print(df_prep)
    predictions = run_inference(df_prep, model)
    print(predictions)
    save_predictions(predictions, "data/predictions.csv")

import json

import mlflow
import mlflow.lightgbm
import pandas as pd

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5002")
    model_name = "my-model"
    model_version = 1
    model_uri = f"models:/{model_name}/{model_version}"

    # 1. MLflow에 등록된 예제 데이터 가져오기
    # run id 추출
    client = mlflow.MlflowClient()
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )
    run_id = model_version_details.run_id
    # run에 등록된 예제 데이터 가져오기
    client.download_artifacts(run_id, "model/input_example.json", "data/")
    with open("data/model/input_example.json", "r") as f:
        input_example = json.load(f)
        input_example = pd.DataFrame(
            data=input_example["data"], columns=input_example["columns"]
        )

    # 2. 모델 로드
    model = mlflow.lightgbm.load_model(model_uri)

    # 3. 예측
    y_pred = model.predict(input_example)
    print(y_pred)

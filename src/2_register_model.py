import mlflow

if __name__ == "__main__":
    run_id = "9c65b053cd654d5185b09393c8d6cf07"  # 1_base_model_with_mlflow.py에서 생성한 run id
    model_name = "my-model"
    model_uri = f"runs:/{run_id}/model"

    # 1. 모델 등록
    mlflow.set_tracking_uri("http://localhost:5002")
    registered_model_version = mlflow.register_model(
        model_uri=model_uri, name=model_name
    )

    # 2. 모델 별칭 등록
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias="production", version=registered_model_version.version
    )

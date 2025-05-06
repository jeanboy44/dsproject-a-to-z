"""
MLflow 실험을 생성하는 스크립트

실행 방법:
```
python notebooks/create_mlflow_experiment.py --experiment-name exp01
```
"""

import mlflow
import typer


def main(
    experiment_name: str = typer.Option(..., "--experiment-name", help="실험 이름"),
):
    mlflow.set_tracking_uri("http://localhost:5001")

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(
            f"MLflow experiment '{experiment_name}' created successfully with ID: {experiment_id}"
        )
    except Exception:
        print(
            f"Could not create MLflow experiment '{experiment_name}'. It might already exist."
        )


if __name__ == "__main__":
    typer.run(main)

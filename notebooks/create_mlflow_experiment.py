import mlflow

# Define the experiment name
mlflow.set_tracking_uri("http://localhost:5001")
experiment_name = "exp01"

try:
    # Create the MLflow experiment
    experiment_id = mlflow.create_experiment(experiment_name)
    print(
        f"MLflow experiment '{experiment_name}' created successfully with ID: {experiment_id}"
    )
except Exception:
    print(
        f"Could not create MLflow experiment '{experiment_name}'. It might already exist."
    )

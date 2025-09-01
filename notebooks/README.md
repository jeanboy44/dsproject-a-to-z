# 노트북 실행하기

## MLFlow 설정 방법
### 서버 실행
```sh
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

### 실험 생성
```sh
python notebooks/create_mlflow_experiment.py --experiment-name exp01
```
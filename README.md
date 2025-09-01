# 제조 데이터 활용 분석 실습
제조 데이터를 활용한 ML모델 및 데모 App 개발 실습 프로젝트입니다. 본 프로젝트에서 사용하는 데이터는 인공지능 제조 플랫폼 KAMP에서 제공되었습니다. 데이터에 대한 자세한 내용은 [링크](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=%EC%A3%BC%EC%A1%B0+%ED%92%88%EC%A7%88&page=1&DATASET_SEQ=55&DISPLAY_MODE_SEL=CARD&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=)를 참고하세요.

## 설치
### 환경 설치
#### (option 1) with conda
1. create and activate conda environment
    ```
    conda create -n dsproject-a-to-z python=3.12
    conda activate dsproject-a-to-z
    ```
1. install required packages
    ```
    pip install -r requirements.txt
    ```

#### (option 2) with uv
1. install uv(version: uv 0.7.2)
   - mac: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
1. setup python envrionment
    ```
    uv sync
    ```
1. activate venv
    - mac: `source .venv/bin/activate`
    - windows: `.venv\Scripts\activate`

### 환경 변수 설정
1. 환경 변수 설정을 위한 `.env` 파일 생성
    1. `.env.example` 파일을 복사하여 `.env` 생성
        ```
        PYTHONPATH="."
        ```

## 테스트
```
pytest tests
```

## 실습 순서
1. 다운로드 데이터
```
python scripts/download_data.py --url https://drive.google.com/file/d/1Bh_Q1WROysdeF25BazhCkddKs-8bjJCi/view?usp=sharing --output-path data/DieCasting_Quality_Raw_Data.csv
```

2. EDA
`notebooks/01_eda.ipynb` 실행

3. 데이터 전처리
```
python scripts/preprocess_data.py --input-path data/DieCasting_Quality_Raw_Data.csv --output-path data/DieCasting_Quality_Data_processed.csv
```

4. MLFLOW 셋업
```
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
```

5. 실험 1. MLFlow 실험 생성
```
python notebooks/create_mlflow_experiment.py --experiment-name exp01
```

6. 실험 1. 학습, 테스트 데이터 생성
```
python notebooks/exp01/create_train_test.py
```

7. 실험 1. 모델 분석 실행
    - `notebooks/exp01/01_base.ipynb`
    - `notebooks/exp01/02_base_mlflow.ipynb`
    - `notebooks/exp01/03_base_mlflow_autolog.ipynb`
    - `notebooks/exp01/04_add_your_code.ipynb`

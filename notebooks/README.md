# 노트북 실행하기

## 데이터 준비
### 원천 데이터 다운로드
```sh
python notebooks/download_data.py --url https://drive.google.com/file/d/1Bh_Q1WROysdeF25BazhCkddKs-8bjJCi/view?usp=sharing --output-path data/DieCasting_Quality_Raw_Data.csv
```



## MLFlow 설정
```
mlflow server \
  --backend-store-uri file:/Users/jeanboy/workspace/dsproject-a-to-z/mlruns \
  --default-artifact-root file:/Users/jeanboy/workspace/dsproject-a-to-z/mlruns \
  --host 0.0.0.0 \
  --port 5001
```
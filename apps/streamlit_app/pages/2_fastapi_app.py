import shutil
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh


@st.cache_resource
def reset_dirs(src_dir: Path, dst_dir: Path):
    if src_dir.is_dir():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    if dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


def get_results_to_csv(src_dir: Path, dst_dir: Path, threshold: float) -> pd.DataFrame:
    """특징과 예측 결과를 병합하여 DataFrame으로 반환합니다.

    Args:
        src_dir (Path): 특징 데이터가 저장된 디렉토리 경로
        dst_dir (Path): 예측 결과가 저장된 디렉토리 경로
        threshold (float): 결함 판정을 위한 임계값

    Returns:
        pd.DataFrame: 특징과 예측 결과가 병합된 DataFrame
    """
    features = []
    for file in src_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df["file_name"] = file.stem
        features.append(df)

    if not features:
        return pd.DataFrame()

    features = pd.concat(features)

    predictions = []
    for file in dst_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df["file_name"] = file.stem
        predictions.append(df)

    predictions = pd.concat(predictions)
    predictions["y_hat"] = predictions["probability"] > threshold

    results = pd.merge(features, predictions, on="file_name", how="left")
    return results


def load_config(config_path: str) -> dict:
    """설정 파일을 로드합니다.

    Args:
        config_path (str): 설정 파일 경로

    Returns:
        dict: CLI 앱 설정 정보
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("fastapi_app")


def count_production(dst_dir: Path) -> int:
    """예측된 파일 개수로 전체 생산량을 계산합니다.

    Args:
        dst_dir (Path): 예측 결과가 저장된 디렉토리 경로

    Returns:
        int: 전체 생산량
    """
    return len(list(dst_dir.glob("*")))


def count_defect(dst_dir: Path, threshold: float) -> int:
    """예측된 파일 중 결함 수를 계산합니다.

    Args:
        dst_dir (Path): 예측 결과가 저장된 디렉토리 경로
        threshold (float): 결함 판정을 위한 임계값

    Returns:
        int: 결함 수
    """
    num_defect = 0
    for file in dst_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if df["probability"].iloc[0] > threshold:
            num_defect += 1
    return num_defect


def get_files_not_processed(src_dir: Path, dst_dir: Path) -> list[Path]:
    return list(set(src_dir.glob("*.csv")) - set(dst_dir.glob("*.csv")))


def predict_files(host: str, port: int, files: list[Path], dst_dir: Path):
    for file in files:
        input_data = {}
        df = pd.read_csv(file)
        input_data["raw"] = df.to_dict(orient="records")[0]
        try:
            response = requests.post(f"http://{host}:{port}/predict", json=input_data)
            pd.DataFrame([response.json()]).to_csv(dst_dir / file.name, index=False)
        except Exception:
            pd.DataFrame([{"probability": None}]).to_csv(
                dst_dir / file.name, index=False
            )


# 0.5초에 한 번씩 리프레시
st_autorefresh(interval=500)

# 설정 파일 로드
config = load_config("apps/streamlit_app/config.yaml")
src_dir = Path(config["src_dir"])
dst_dir = Path(config["dst_dir"])
threshold = config["threshold"]
results_dir = Path(config["results_dir"])
target_production = config["target_production"]
target_defect_rate = config["target_defect_rate"]
host = config["host"]
port = config["port"]
reset_dirs(src_dir, dst_dir)

files_to_predict = get_files_not_processed(src_dir, dst_dir)
predict_files(host, port, files_to_predict, dst_dir)

production_amount = count_production(dst_dir)  # 생산량 계산
defect_amount = count_defect(dst_dir, threshold)  # 결함 수 계산
nondefect_amount = production_amount - defect_amount  # 양품 수 계산
defect_rate = (
    defect_amount / production_amount if production_amount > 0 else 0
)  # 결함률 계산

# 화면
st.title("다이캐스팅 주조 설비 결함 검사")
tab1, tab2 = st.tabs(["생산 현황", "생산 이력"])
# 생산 현황 탭
with tab1:
    st.metric(
        label="생산량",
        value=f"{production_amount} / {target_production}",
        border=True,
        delta=f"{production_amount - target_production}",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="양품 수", value=nondefect_amount, border=True)
    with col2:
        st.metric(label="결함 수", value=defect_amount, border=True)

    st.metric(
        label="결함률",
        value=f"{defect_rate * 100}%",
        border=True,
        delta_color="inverse",
        delta=f"{(defect_rate - target_defect_rate) * 100}",
    )
    st.markdown("---")
# 생산 이력 탭
with tab2:
    results = get_results_to_csv(src_dir, dst_dir, threshold)
    if results.empty:
        st.warning("생산 이력이 없습니다.")
        st.stop()

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        y_hat = st.multiselect(
            "결함 여부",
            options=[True, False],
            default=True,
        )
        data = results[results["y_hat"].isin(y_hat)]
    with col2:
        st.download_button(
            label="데이터 저장",
            data=data.to_csv(index=False),
            file_name="results.csv",
            icon=":material/download:",
        )
    st.dataframe(
        data,
        column_config=None,
        hide_index=False,
        use_container_width=True,
        column_order=None,
    )

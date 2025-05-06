from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh


def get_results_to_csv(src_dir: Path, dst_dir: Path, threshold: float):
    features = []
    for file in src_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df["file_name"] = file.stem
        features.append(df)

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


src_dir = Path("data/cli_app/src")
dst_dir = Path("data/cli_app/dst")
results_dir = Path("data/cli_app/results")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("No configuration found")

    if not config.get("cli_app"):
        raise ValueError("No CLI app configuration found")

    return config.get("cli_app")


def count_production(dst_dir: Path):
    return len(list(dst_dir.glob("*")))


def count_defect(dst_dir: Path, threshold: float):
    num_defect = 0
    for file in dst_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if df["probability"].iloc[0] > threshold:
            num_defect += 1
    return num_defect


config = load_config("apps/streamlit_app/config.yaml")
src_dir = Path(config["src_dir"])
dst_dir = Path(config["dst_dir"])

st_autorefresh(interval=500)
production_amount = count_production(dst_dir)
defect_amount = count_defect(dst_dir, 0.5)
nondefect_amount = production_amount - defect_amount
defect_rate = defect_amount / production_amount if production_amount > 0 else 0

st.title("다이캐스팅 주조 설비 결함 검사")
st.metric(
    label="생산량",
    value=f"{production_amount} / {config['target_production']}",
    border=True,
    delta=f"{production_amount - config['target_production']}",
)
# 양품 수
col1, col2 = st.columns(2)
with col1:
    st.metric(label="양품 수", value=nondefect_amount, border=True)
# 결함 수
with col2:
    st.metric(label="결함 수", value=defect_amount, border=True)
# 결함률
st.metric(
    label="결함률",
    value=f"{defect_rate * 100}%",
    border=True,
    delta_color="inverse",
    delta=f"{(defect_rate - config['target_defect_rate']) * 100}",
)

if production_amount == config["target_production"]:
    results = get_results_to_csv(src_dir, dst_dir, config["threshold"])
    results.to_csv(results_dir / "results.csv", index=False)

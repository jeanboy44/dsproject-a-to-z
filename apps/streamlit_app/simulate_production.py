import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
import yaml
from tqdm import tqdm
from typer import Option

with open("apps/streamlit_app/config.yaml", "r") as f:
    config = yaml.safe_load(f)


app = typer.Typer()


def simulate_production(
    src_path: Path,
    dst_dir: Path,
    target_production: int,
    cycle_time: int,
    seed: int,
):
    if not src_path.is_file():
        raise typer.BadParameter("src_path must be a csv file.")

    if not dst_dir.is_dir():
        raise typer.BadParameter("dst_dir must be a directory.")

    df = pd.read_csv(src_path, skiprows=1)

    # Generate random indices for sampling
    indices = df.index.to_list()
    random_indices = (
        pd.Series(indices).sample(n=target_production, random_state=seed).to_list()
    )

    for idx, random_idx in tqdm(enumerate(random_indices), total=target_production):
        samples = df.iloc[[random_idx], :]
        samples.to_csv(
            dst_dir
            / f"{idx + 1:04d}_{random_idx:04d}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            index=False,
        )

        time.sleep(cycle_time)


@app.command("cli")
def simulate_cli_production(
    src_path: Path = Option(
        "data/DieCasting_Quality_Raw_Data.csv",
        "--src-path",
        help="The path to the input CSV file.",
    ),
    cycle_time: int = Option(
        1,
        "--cycle-time",
        help="The cycle time for each production. in seconds.",
    ),
    seed: int = Option(
        42,
        "--seed",
        help="The random seed for reproducibility.",
    ),
):
    app_config = config["cli_app"]
    src_dir = Path(app_config["src_dir"])

    simulate_production(
        src_path=src_path,
        dst_dir=src_dir,
        target_production=app_config["target_production"],
        cycle_time=cycle_time,
        seed=seed,
    )


@app.command("fastapi")
def simulate_fastapi_production(
    src_path: Path = Option(
        "data/DieCasting_Quality_Raw_Data.csv",
        "--src-path",
        help="The path to the input CSV file.",
    ),
    cycle_time: int = Option(
        1,
        "--cycle-time",
        help="The cycle time for each production. in seconds.",
    ),
    seed: int = Option(
        42,
        "--seed",
        help="The random seed for reproducibility.",
    ),
):
    app_config = config["fastapi_app"]
    src_dir = Path(app_config["src_dir"])

    simulate_production(
        src_path=src_path,
        dst_dir=src_dir,
        target_production=app_config["target_production"],
        cycle_time=cycle_time,
        seed=seed,
    )


if __name__ == "__main__":
    app()

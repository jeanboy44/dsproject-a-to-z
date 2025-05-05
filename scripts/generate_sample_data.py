from pathlib import Path

import pandas as pd
import typer
from typer import Option


def main(
    src_path: Path = Option(
        "data/DieCasting_Quality_Raw_Data.csv",
        "--src-path",
        help="The path to the input CSV file.",
    ),
    dst_path: Path = Option(
        "data/sample.csv",
        "--dst-path",
        help="The path to the output file.",
        callback=lambda p: p
        if p.suffix in [".csv", ".json"]
        else typer.BadParameter("File must have .csv or .json extension"),
    ),
    seed: int = Option(42, "--seed", help="The random seed for reproducibility."),
):
    """
    Reads a CSV file, sets the random seed, and prints a specified number of random samples.

    Args:
        src_path (Path): The path to the input CSV file.
        dst_path (Path): The path to the output file.(csv for json, tex)
        seed (int): The random seed for reproducibility.
    """
    df = pd.read_csv(src_path, skiprows=1)

    samples = df.sample(n=1, random_state=seed)

    if dst_path.suffix == ".csv":
        samples.to_csv(dst_path, index=False)
    elif dst_path.suffix == ".json":
        samples.to_json(dst_path, orient="records")
    else:
        raise ValueError(f"Unsupported file extension: {dst_path.suffix}")


if __name__ == "__main__":
    typer.run(main)

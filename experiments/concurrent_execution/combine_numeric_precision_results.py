import sys
from pathlib import Path

import pandas as pd
import glob, re


def combine_numeric_precision_results(input_directory: str, output_directory: Path) -> None:
    INPUT_GLOB = f"{input_directory}/solving_combined_statistics_*_digit*.csv"
    OUTPUT_CSV = output_directory / "numeric_precision_out.csv"
    KEEP_ALGO = "numeric_sam"

    paths = sorted(glob.glob(INPUT_GLOB))
    dfs = []

    for p in paths:
        df = pd.read_csv(p)
        m = re.search(r"_(\d+)_digits\.csv", p)
        if "num_digits" not in df.columns and m:
            df["num_digits"] = int(m.group(1))
        if "learning_algorithm" in df.columns:
            df = df[df["learning_algorithm"].astype(str) == KEEP_ALGO].copy()
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    out = (
        all_df.groupby(["num_digits", "num_trajectories"], as_index=False)["percent_ok"]
        .mean()
        .sort_values(["num_digits", "num_trajectories"])
    )

    out = out.rename(columns={"percent_ok": "percent_solved"})
    out = out[["num_trajectories", "percent_solved", "num_digits"]]
    out.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = Path(sys.argv[2])
    combine_numeric_precision_results(input_dir, output_dir)

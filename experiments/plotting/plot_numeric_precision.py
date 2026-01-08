import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Load data
def plot_numeric_precision(input_path: Path, output_folder_path: Path) -> None:
    df = pd.read_csv(input_path)

    plt.figure(figsize=(12, 8))

    digits = sorted(df["num_digits"].unique())
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

    for i, d in enumerate(digits):
        subset = df[df["num_digits"] == d].sort_values("num_trajectories")
        plt.plot(
            subset["num_trajectories"],
            subset["percent_solved"],
            label=f"{int(d)} digits",
            linewidth=8,
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
        )

    plt.xlabel("# Trajectories", fontsize=44)
    plt.ylabel("AVG % of solved", fontsize=44)
    plt.ylim(0, 100)
    plt.xticks(fontsize=44)
    plt.yticks(fontsize=44)
    plt.legend(fontsize=40)
    plt.grid(True)
    plt.tight_layout()

    output_path = output_folder_path / "numeric_precision.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    output_folder_path = Path(sys.argv[2])
    plot_numeric_precision(input_path, output_folder_path)

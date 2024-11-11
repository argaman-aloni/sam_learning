import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_results(results_directory_path: Path):
    """Plot the results of the experiments."""
    for file_path in results_directory_path.glob("*solving_aggregated*.csv"):
        df = pd.read_csv(file_path)

        # Define color-blind friendly palette
        color_palette = sns.color_palette("colorblind", n_colors=len(df["learning_algorithm"].unique()))
        line_styles = ["-", "--", "-.", ":", "--", "-.", ":"]
        marker_options = ["o", "s", "D", "v", "^", ">", "<", "p", "P", "*", "X", "d"]
        line_styles_iterator = list(line_styles)
        color_cycle = list(color_palette)
        markers = list(marker_options)

        # Group the data by 'num_trajectories', 'learning_algorithm' and calculate the mean and std of 'percent_ok'
        df = df[~df["learning_algorithm"].isin(["incremental_nsam", "naive_nsam_no_dependency_removal"])]  # Remove max_percent_ok from the plot
        grouped_data = (
            df.groupby(["num_trajectories", "learning_algorithm"])
            .agg(
                avg_max_percent_ok=("max_percent_ok", "mean"),
                std_max_percent_ok=("max_percent_ok", "std"),
                goal_not_achieved=("percent_goal_not_achieved", "first"),
            )
            .reset_index()
        )

        labels = {
            "numeric_sam": "NSAM*",
            "naive_nsam": "NSAM",
        }

        # Plotting
        sns.set(style="whitegrid")
        stand_alone_fig = plt.figure(figsize=(12, 8))

        legend_order = [1, 0]

        # Plot a line for each learning algorithm
        for index, algo in enumerate(df["learning_algorithm"].unique()):
            algo_data = grouped_data[grouped_data["learning_algorithm"] == algo]

            plt.plot(
                algo_data["num_trajectories"],
                algo_data["avg_max_percent_ok"],
                linestyle=line_styles_iterator[index],
                label=labels[algo],
                marker=markers[index],
                color=color_cycle[index],
                linewidth=8,
            )

            plt.fill_between(
                algo_data["num_trajectories"],
                np.clip(algo_data["avg_max_percent_ok"] - algo_data["std_max_percent_ok"], 0, 100),
                np.clip(algo_data["avg_max_percent_ok"] + algo_data["std_max_percent_ok"], 0, 100),
                alpha=0.2,
            )

        plt.xlabel("# Trajectories", fontsize=44)
        plt.ylabel("AVG % of solved", fontsize=44)
        plt.ylim(0, 100)
        plt.tick_params(axis="both", which="major", labelsize=46)
        plt.grid(True)
        handles1, legend_labels1 = plt.gca().get_legend_handles_labels()
        stand_alone_fig.legend(
            [handles1[idx] for idx in legend_order],
            [legend_labels1[idx] for idx in legend_order],
            fontsize=48,
            loc='upper right',  # Inside the grid, on the right
            frameon=True  # Optional: frame around legend
        )

        output_file_path = file_path.parent / f"{file_path.stem}_plot_solo.png"
        plt.tight_layout()  # Ensures no overlap between plot elements
        plt.savefig(output_file_path, bbox_inches="tight")


if __name__ == "__main__":
    results_path = Path(sys.argv[1])
    plot_results(results_path)

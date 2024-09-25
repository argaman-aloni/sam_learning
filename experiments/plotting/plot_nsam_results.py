import sys
from itertools import cycle
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
        line_styles_iterator = cycle(line_styles)
        color_cycle = cycle(color_palette)
        markers = cycle(marker_options)

        # Group the data by 'num_trajectories', 'learning_algorithm' and calculate the mean and std of 'percent_ok'
        df = df[df["learning_algorithm"] != "incremental_nsam"]  # Remove max_percent_ok from the plot
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
            "naive_nsam": "NSAM with DR",
            "naive_nsam_no_dependency_removal": "NSAM without DR",
        }

        # Plotting
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        legend_order = [2, 0, 1]

        # Plot a line for each learning algorithm
        for algo in df["learning_algorithm"].unique():
            algo_data = grouped_data[grouped_data["learning_algorithm"] == algo]
            plt.plot(
                algo_data["num_trajectories"],
                algo_data["avg_max_percent_ok"],
                linestyle=next(line_styles_iterator),
                label=labels[algo],
                marker=next(markers),
                color=next(color_cycle),
                linewidth=3,
            )

            # Plot standard deviation as shaded area around the mean line
            plt.fill_between(
                algo_data["num_trajectories"],
                np.clip(algo_data["avg_max_percent_ok"] - algo_data["std_max_percent_ok"], 0, 100),
                np.clip(algo_data["avg_max_percent_ok"] + algo_data["std_max_percent_ok"], 0, 100),
                alpha=0.2,
            )

        # Set plot labels and title
        plt.xlabel("# Observations", fontsize=24)
        plt.ylabel("AVG % of solved", fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(ticks=list(range(0, 101, 10)), fontsize=24)

        # Add a legend
        handles, legend_labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[idx] for idx in legend_order], [legend_labels[idx] for idx in legend_order], fontsize=24)
        plt.grid(True)

        output_file_path = file_path.parent / f"{file_path.stem}_plot.png"
        plt.savefig(output_file_path, bbox_inches="tight")

        # Show the plot
        plt.show()

        # Second Plot (with x-axis limited to 0-10)
        line_styles_iterator = cycle(line_styles)
        color_cycle = cycle(color_palette)
        markers = cycle(marker_options)
        plt.figure(figsize=(10, 6))

        for algo in df["learning_algorithm"].unique():
            algo_data = grouped_data[grouped_data["learning_algorithm"] == algo]
            plt.plot(
                algo_data["num_trajectories"],
                algo_data["avg_max_percent_ok"],
                linestyle=next(line_styles_iterator),
                label=labels[algo],
                marker=next(markers),
                color=next(color_cycle),
                linewidth=3,
            )

            # Plot standard deviation as shaded area around the mean line
            plt.fill_between(
                algo_data["num_trajectories"],
                np.clip(algo_data["avg_max_percent_ok"] - algo_data["std_max_percent_ok"], 0, 100),
                np.clip(algo_data["avg_max_percent_ok"] + algo_data["std_max_percent_ok"], 0, 100),
                alpha=0.2,
            )

        plt.xlabel("# Observations (0-10)", fontsize=24)
        plt.ylabel("AVG % of solved", fontsize=24)
        plt.xlim(0, 10)
        plt.xticks(fontsize=24)
        plt.yticks(ticks=list(range(0, 101, 10)), fontsize=24)

        handles, legend_labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[idx] for idx in legend_order], [legend_labels[idx] for idx in legend_order], fontsize=24)
        plt.grid(True)

        output_file_path_limited = file_path.parent / f"{file_path.stem}_plot_limited.png"
        plt.savefig(output_file_path_limited, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    results_path = Path(sys.argv[1])
    plot_results(results_path)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

markers = ["o", "s", "^", "D", "v", "P", "X", "H"]  # Marker styles for line differentiation
line_styles = ["-", "--", "-.", ":", "--", "-.", ":"]


def plot_solving_results(file_path: Path, output_file_path: Path, using_triplets: bool = False):
    # Adjusting the x-axis to start from 0 and end at the maximum value of 'num_trajectories'
    data = pd.read_csv(file_path)
    debug_columns = [column for column in data.columns if "problems" in column]
    data = data.drop(columns=["solver", *debug_columns])
    # Prepare the data: group by and average 'percent_ok' over 'fold'
    group_by_columns = (
        ["num_trajectories", "learning_algorithm", "policy"] if not using_triplets else ["num_trajectory_triplets", "learning_algorithm", "policy"]
    )
    grouped_data = data.groupby(group_by_columns, as_index=False).agg(avg_percent_ok=("percent_ok", "mean"), std_percent_ok=("percent_ok", "std"),)

    # Label mappings for readability
    algorithm_label_map = {"sam_learning": "SAM", "ma_sam": "MA-SAM", "ma_sam_plus": "MA-SAM+"}
    grouped_data["learning_algorithm"] = grouped_data["learning_algorithm"].replace(algorithm_label_map)
    grouped_data["policy"] = grouped_data["policy"].str.replace("_", " ")

    # Create the figure with subplots for each unique policy
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    legend_order = [2, 0, 1]
    # Reset color index for consistent color usage across all subplots
    colorblind_palette = sns.color_palette("colorblind", n_colors=len(grouped_data["learning_algorithm"].unique()))

    for i, policy in enumerate(["no remove", "soft", "hard"]):
        ax = axes[i]
        policy_data = grouped_data[grouped_data["policy"] == policy]
        unique_combinations = policy_data.groupby(["learning_algorithm"])
        for idx, (algo, group) in enumerate(unique_combinations):
            plot_x_axis = group["num_trajectories"] if not using_triplets else group["num_trajectory_triplets"]
            ax.plot(
                plot_x_axis,
                group["avg_percent_ok"],
                label=algo[0],
                color=colorblind_palette[idx % len(colorblind_palette)],
                marker=markers[idx % len(markers)],
                markersize=6,
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=4,
            )
            ax.fill_between(
                plot_x_axis,
                np.clip(group["avg_percent_ok"] - group["std_percent_ok"], 0, 100),
                np.clip(group["avg_percent_ok"] + group["std_percent_ok"], 0, 100),
                alpha=0.2,
            )
            ax.set_yticks(range(0, 110, 10))
            ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontsize=20)
            ax.set_yticklabels(ax.get_yticks(), fontsize=20)

        # Individual plot for the current policy
        plt.figure(figsize=(10, 6))
        for idx, (algo, group) in enumerate(unique_combinations):
            plot_x_axis = group["num_trajectories"] if not using_triplets else group["num_trajectory_triplets"]
            plt.plot(
                plot_x_axis,
                group["avg_percent_ok"],
                label=algo,
                color=colorblind_palette[idx % len(colorblind_palette)],
                marker=markers[idx % len(markers)],
                markersize=6,
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=4,
            )
            plt.fill_between(
                plot_x_axis,
                np.clip(group["avg_percent_ok"] - group["std_percent_ok"], 0, 100),
                np.clip(group["avg_percent_ok"] + group["std_percent_ok"], 0, 100),
                alpha=0.2,
            )
        plt.title(f"Policy: {policy}", fontsize=28)
        plt.xlabel(f"# {'Trajectories' if not using_triplets else 'Triplets'}", fontsize=28)
        plt.ylabel("Average % Solved", fontsize=28)
        plt.tick_params(axis="both", which="major", labelsize=28)
        plt.ylim(0, 100)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(
            [handles[idx] for idx in legend_order], ["SAM", "MA-SAM", "MA-SAM+"], fontsize=28,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_file_path.parent / f"solving_statistics_{policy.replace(' ', '_')}_plot{'_with_triplets' if using_triplets else ''}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        ax.set_title(f"Policy: {policy}", fontsize=24)
        ax.set_xlabel(f"# {'Trajectories' if not using_triplets else 'Triplets'}", fontsize=24)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average % Solved", fontsize=24)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    axes[2].legend(
        [handles[idx] for idx in legend_order], [legend_labels[idx] for idx in legend_order], fontsize=24,
    )

    plt.tight_layout()  # Adjust layout for legend
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    args = sys.argv
    plot_solving_results(Path(args[1]), Path(args[2]))

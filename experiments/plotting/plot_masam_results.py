from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_solving_results(file_path: Path, output_file_path: Path):
    # Adjusting the x-axis to start from 0 and end at the maximum value of 'num_trajectories'
    data = pd.read_csv(file_path)
    debug_columns = [column for column in data.columns if "problems" in column]
    data = data.drop(columns=["solver", *debug_columns])
    # Prepare the data: group by and average 'percent_ok' over 'fold'
    grouped_data = data.groupby(["num_trajectories", "learning_algorithm", "policy"], as_index=False).mean()

    # Label mappings for readability
    algorithm_label_map = {"sam_learning": "SAM", "ma_sam": "MA-SAM", "ma_sam_plus": "MA-SAM+"}
    grouped_data["learning_algorithm"] = grouped_data["learning_algorithm"].replace(algorithm_label_map)
    grouped_data["policy"] = grouped_data["policy"].str.replace("_", " ")

    # Create the figure with subplots for each unique policy
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    legend_order = [2, 0, 1]
    # Reset color index for consistent color usage across all subplots
    colorblind_palette = sns.color_palette("colorblind", n_colors=len(grouped_data["learning_algorithm"].unique()))
    markers = ["o", "s", "^", "D", "v", "P", "X", "H"]  # Marker styles for line differentiation
    line_styles = ["-", "--", "-.", ":", "--", "-.", ":"]

    for i, policy in enumerate(["no remove", "soft", "hard"]):
        ax = axes[i]
        policy_data = grouped_data[grouped_data["policy"] == policy]
        unique_combinations = policy_data.groupby(["learning_algorithm"])

        for idx, (algo, group) in enumerate(unique_combinations):
            ax.plot(
                group["num_trajectories"],
                group["percent_ok"],
                label=algo[0],
                color=colorblind_palette[idx % len(colorblind_palette)],
                marker=markers[idx % len(markers)],
                markersize=6,
                linestyle=line_styles[idx % len(line_styles)],
                linewidth=3,
            )

        ax.set_title(f"Policy: {policy}")
        ax.set_xlabel("# Trajectories")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average % Solved")
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        [handles[idx] for idx in legend_order],
        [legend_labels[idx] for idx in legend_order],
        title="Algorithm",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()  # Adjust layout for legend
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
    plt.show()

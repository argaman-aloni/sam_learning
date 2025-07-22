import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

from pandas import DataFrame

from utilities import LearningAlgorithmType


def plot_online_learning_statistics(
    working_directory: Path,
    learning_algorithm: LearningAlgorithmType = LearningAlgorithmType.semi_online,
    unified_df: DataFrame = None,
    domain_name: str = "unknown_domain",
):
    """Plots the statistics of model solution statuses per episode."""
    averaged = unified_df.groupby("episode_number")[["not_solved", "optimistic_not_applicable", "optimistic_solved", "safe_solved"]].mean()
    smoothed = averaged.rolling(window=5, min_periods=1, center=True).mean()
    categories = ["Not Solved", "SVR-AM Inapplicable", "SVR-AM Solved", "NSAM Solved"]
    # Plotting the graph
    fig, ax = plt.subplots(figsize=(12, 8))
    stacks = ax.stackplot(
        smoothed.index,
        smoothed["not_solved"],
        smoothed["optimistic_not_applicable"],
        smoothed["optimistic_solved"],
        smoothed["safe_solved"],
        labels=["Not Solved", "SVR-AM Inapplicable", "SVR-AM Solved", "NSAM Solved"],
        colors=["#F20000", "#56B4E9", "#E0FA5C", "#71FA5C"],  # distinct colorblind-friendly palette
        zorder=1,
    )
    hatches = ["o", "*", ".", "x"]  # distinct hatches for each stack
    for stack, hatch in zip(stacks, hatches):
        stack.set_hatch(hatch)

    x = smoothed.index.to_numpy()
    y_values = [
        smoothed["not_solved"].to_numpy(),
        smoothed["optimistic_not_applicable"].to_numpy(),
        smoothed["optimistic_solved"].to_numpy(),
        smoothed["safe_solved"].to_numpy(),
    ]

    # Draw separating boundary lines
    cumulative = np.zeros_like(y_values[0])
    for y in y_values[:-1]:  # skip top layer
        cumulative += y
        ax.plot(x, cumulative, color="black", linewidth=1.5, zorder=2)

    ax.set_xlabel("Episode", fontsize=36)
    ax.set_ylabel("Solving Rate", fontsize=36)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 80)
    ax.tick_params(axis="both", which="major", labelsize=36)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(
        working_directory / "results_directory" / f"{domain_name}_{learning_algorithm.name}_model_solution_status_percentages.pdf",
        bbox_inches="tight",
        dpi=300,
    )


def plot_exploration_time_statistics(
    working_directory: Path,
    learning_algorithm: LearningAlgorithmType = LearningAlgorithmType.semi_online,
    unified_df: DataFrame = None,
    domain_name: str = "unknown_domain",
):
    """Plots the exploration time statistics."""
    plt.figure(figsize=(12, 8))
    averaged = unified_df.groupby("episode_number")[["exploration_time"]].mean()
    smoothed = averaged.rolling(window=5, min_periods=1, center=True).mean()
    plt.plot(smoothed.index, smoothed["exploration_time"], color="#000000", linewidth=3)

    plt.xlabel("Episode", fontsize=36)
    plt.ylabel("Exploration Time (sec)", fontsize=36)
    plt.tick_params(axis="both", which="major", labelsize=36)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        working_directory / "results_directory" / f"{domain_name}_{learning_algorithm.name}_exploration_time_statistics.pdf",
        bbox_inches="tight",
        dpi=300,
    )


def plot_legend(working_directory: Path, domain_name: str = "unknown_domain"):
    """Plots the legend for the online learning statistics."""
    patches = [
        Patch(facecolor="#F20000", label="Not Solved", hatch="o"),
        Patch(facecolor="#56B4E9", label="Inapplicable", hatch="*"),
        Patch(facecolor="#E0FA5C", label="Solved", hatch="."),
        Patch(facecolor="#71FA5C", label="Solved Safe", hatch="x"),
    ]

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", ncol=len(patches), fontsize=24, frameon=False)
    fig.savefig(
        working_directory / "results_directory" / f"{domain_name}_legend.pdf",
        bbox_inches="tight",
        dpi=300,
    )

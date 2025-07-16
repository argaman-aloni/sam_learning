import matplotlib.pyplot as plt
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

    # Plotting the graph
    plt.figure(figsize=(12, 8))
    plt.stackplot(
        smoothed.index,
        smoothed["not_solved"],
        smoothed["optimistic_not_applicable"],
        smoothed["optimistic_solved"],
        smoothed["safe_solved"],
        labels=["Not Solved", "SVR-AM Inapplicable", "SVR-AM Solved", "NSAM Solved"],
        colors=["#F20000", "#FA915C", "#E0FA5C", "#71FA5C"],  # distinct colorblind-friendly palette
    )

    plt.xlabel("Episode", fontsize=36)
    plt.ylabel("Solving Rate", fontsize=36)
    plt.ylim(0, 1)
    plt.xlim(0, 80)
    # plt.title("Model Solution Status Percentages per Episode")
    # plt.legend(fontsize=36)
    plt.tick_params(axis="both", which="major", labelsize=36)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
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
        Patch(facecolor="#F20000", label="Not Solved"),
        Patch(facecolor="#FA915C", label="Inapplicable"),
        Patch(facecolor="#E0FA5C", label="Solved"),
        Patch(facecolor="#71FA5C", label="Solved Safe"),
    ]

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", ncol=len(patches), fontsize=24, frameon=False)
    fig.savefig(
        working_directory / "results_directory" / f"{domain_name}_legend.pdf",
        bbox_inches="tight",
        dpi=300,
    )

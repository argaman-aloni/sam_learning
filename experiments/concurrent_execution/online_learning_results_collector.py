from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from experiments.plotting.plot_online_learning_results import plot_online_learning_statistics
from utilities import LearningAlgorithmType


def export_unified_statistics_to_csv(
    working_directory: Path, filename: str, domain_name: str, learning_algorithm: LearningAlgorithmType
) -> DataFrame:
    """Exports the unified statistics DataFrame to a CSV file."""
    # Load the new CSV files for the Counters Exploration
    path_format = working_directory / "results_directory" / f"{learning_algorithm.name}_exploration_statistics_fold_*.csv"
    statistics_file_paths = sorted(glob(str(path_format)))
    fold_data = [pd.read_csv(fp) for fp in statistics_file_paths]

    # Add a 'fold' column to each new DataFrame
    for i, df in enumerate(fold_data):
        df["fold"] = i
        df["not_solved"] = ~df["safe_model_solution_status"].isin(["ok", "not_applicable", "irrelevant"]) & ~df[
            "optimistic_model_solution_status"
        ].isin(["ok", "not_applicable", "irrelevant"]).astype(int)
        df["optimistic_not_applicable"] = (df["optimistic_model_solution_status"] == "not_applicable").astype(int)
        df["optimistic_solved"] = (df["optimistic_model_solution_status"] == "ok").astype(int)
        df["safe_solved"] = (df["safe_model_solution_status"] == "ok").astype(int)

    unified_df = pd.concat(fold_data, ignore_index=True)
    unified_df.to_csv(working_directory / "results_directory" / f"{domain_name}_{learning_algorithm.name}_{filename}", index=False)
    return unified_df


def plot_statistics(
    working_directory: Path, output_csv: str, learning_algorithm: LearningAlgorithmType = LearningAlgorithmType.semi_online
):
    """Plots the statistics of model solution statuses per episode."""
    unified_df = export_unified_statistics_to_csv(working_directory, output_csv, learning_algorithm)

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
        labels=["Not Solved", "Optimistic Not Applicable", "Optimistic Solved", "Safe Solved"],
        colors=["#F20000", "#FA915C", "#E0FA5C", "#71FA5C"],  # distinct colorblind-friendly palette
    )

    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Solving Rate", fontsize=18)
    plt.ylim(0, 1)
    plt.xlim(0, 80)
    # plt.title("Model Solution Status Percentages per Episode")
    plt.legend(fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        working_directory / "results_directory" / f"{learning_algorithm.name}_model_solution_status_percentages.pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot model solution status percentages per episode.")
    parser.add_argument(
        "--working_directory", type=Path, required=True, help="Path to the working directory containing the statistics files."
    )
    parser.add_argument("--output_csv", type=str, default="unified_statistics.csv", help="Output CSV file name for unified statistics.")
    parser.add_argument(
        "--learning_algorithm", type=int, help="The index representing the learning algorithm used in the experiment.", required=True
    )
    parser.add_argument("--domain_name", type=str, required=True, help="The name of the domain the results belong to.")
    args = parser.parse_args()
    unified_df = export_unified_statistics_to_csv(
        Path(args.working_directory), args.output_csv, args.domain_name, LearningAlgorithmType(args.learning_algorithm)
    )
    plot_online_learning_statistics(
        Path(args.working_directory), LearningAlgorithmType(args.learning_algorithm), unified_df, args.domain_name
    )

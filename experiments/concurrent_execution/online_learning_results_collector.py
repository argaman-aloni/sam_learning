from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from typing import List

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
        df["not_solved"] = (
            df["safe_model_solution_status"].isin(["no_solution", "timeout", "solver_error", "goal_not_achieved"])
            & df["optimistic_model_solution_status"].isin(["no_solution", "timeout", "solver_error", "goal_not_achieved"])
            | df["safe_model_solution_status"].isin(["irrelevant"])
            & df["optimistic_model_solution_status"].isin(["no_solution", "timeout", "solver_error", "goal_not_achieved"])
            | df["optimistic_model_solution_status"].isin(["irrelevant"])
            & df["safe_model_solution_status"].isin(["no_solution", "timeout", "solver_error", "goal_not_achieved"]).astype(int)
        )
        df["optimistic_not_applicable"] = (df["optimistic_model_solution_status"] == "not_applicable").astype(int)
        df["optimistic_solved"] = (df["optimistic_model_solution_status"] == "ok").astype(int)
        df["safe_solved"] = (df["safe_model_solution_status"] == "ok").astype(int)

    unified_df = pd.concat(fold_data, ignore_index=True)
    unified_df.to_csv(working_directory / "results_directory" / f"{domain_name}_{learning_algorithm.name}_{filename}", index=False)
    return unified_df


def collect_results_for_all_algorithms(
    working_directory: Path, learning_algorithms: List[LearningAlgorithmType], domain_name: str, output_csv: str = "unified_statistics.csv"
):
    """Collects results for all specified learning algorithms and plots the statistics."""
    for algorithm in learning_algorithms:
        print(f"Processing {algorithm.name}...")
        unified_df = export_unified_statistics_to_csv(working_directory, output_csv, domain_name, algorithm)
        plot_online_learning_statistics(working_directory, algorithm, unified_df, domain_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot model solution status percentages per episode.")
    parser.add_argument(
        "--working_directory", type=Path, required=True, help="Path to the working directory containing the statistics files."
    )
    parser.add_argument("--output_csv", type=str, default="unified_statistics.csv", help="Output CSV file name for unified statistics.")
    parser.add_argument("--learning_algorithms", required=True, help="the list of algorithms that will run in parallel")
    parser.add_argument("--domain_name", type=str, required=True, help="The name of the domain the results belong to.")
    args = parser.parse_args()
    experiment_learning_algorithms = args.learning_algorithms.split(",")
    input_learning_algorithms = [LearningAlgorithmType(int(e)) for e in experiment_learning_algorithms]
    collect_results_for_all_algorithms(Path(args.working_directory), input_learning_algorithms, args.domain_name, args.output_csv)

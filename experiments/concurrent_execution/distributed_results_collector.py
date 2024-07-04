import argparse
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser

from statistics.numeric_performance_calculator import NUMERIC_PERFORMANCE_STATS
from utilities import LearningAlgorithmType
from validators.safe_domain_validator import SOLVING_STATISTICS

FOLD_FIELD = "fold"
MAX_SOLVED_FIELD = "max_percent_solved"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates the folds directories for the slurm tasks to work on.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithms", required=True, help="the list of algorithms that will run in parallel")
    parser.add_argument("--num_folds", required=True, help="the number of folds to that were created", type=int, default=5)
    parser.add_argument("--internal_iterations", required=True, help="The internal iterations that the algorithm will run in parallel.")
    args = parser.parse_args()
    return args


class StatisticsCollector:
    logger: logging.Logger

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learning_algorithms: List[int] = None,
        num_folds: int = 5,
        iterations: List[int] = None,
    ):
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.learning_algorithms = [LearningAlgorithmType(int(e)) for e in learning_algorithms]
        self.num_folds = num_folds
        self.iterations = iterations
        self.logger = logging.getLogger("ClusterRunner")

    @staticmethod
    def _process_combined_data(combined_statistics_data: List[dict]) -> pd.DataFrame:
        # Load the CSV file into a DataFrame
        df = pd.DataFrame(combined_statistics_data)
        numeric_columns = ["percent_ok", "fold", "num_trajectories"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        aggragated_df = df.groupby(["fold", "num_trajectories", "learning_algorithm"]).agg(max_percent_ok=("percent_ok", "max"))
        return aggragated_df

    def _combine_statistics_data(self, file_path_template: str, combined_statistics_data: List[dict]) -> None:
        """

        :param file_path_template:
        :param combined_statistics_data:
        :return:
        """
        results_directory = self.working_directory_path / "results_directory"
        for fold in range(self.num_folds):
            for iteration in self.iterations:
                for learning_algorithm in self.learning_algorithms:
                    solving_statistics_file_path = results_directory / file_path_template.format(
                        fold=fold, iteration=iteration, learning_algorithm=learning_algorithm.name
                    )
                    with open(solving_statistics_file_path, "rt") as statistics_file:
                        reader = csv.DictReader(statistics_file)
                        combined_statistics_data.extend([{FOLD_FIELD: fold, **row} for row in reader])

    def _collect_solving_statistics(self) -> None:
        """Collects the statistics from the statistics files in the results directory and combines them."""
        self.logger.info("Collecting the solving statistics from the results directory.")
        combined_statistics_file_path = self.working_directory_path / "results_directory" / "solving_combined_statistics.csv"
        combined_aggragated_stats = self.working_directory_path / "results_directory" / "solving_aggregated_statistics.csv"
        combined_statistics_data = []
        file_path_template = "{learning_algorithm}_problem_solving_stats_fold_{fold}_{iteration}_trajectories.csv"
        self._combine_statistics_data(file_path_template, combined_statistics_data)
        combined_and_augmented_df = self._process_combined_data(combined_statistics_data)
        combined_and_augmented_df.to_csv(combined_aggragated_stats)
        with open(combined_statistics_file_path, "wt") as combined_statistics_file:
            writer = csv.DictWriter(combined_statistics_file, fieldnames=[FOLD_FIELD, *SOLVING_STATISTICS])
            writer.writeheader()
            writer.writerows(combined_statistics_data)

        self.logger.info("Done collecting the solving statistics from the results directory!")

    def _collect_numeric_performance_statistics(self) -> None:
        self.logger.info("Collecting the numeric performance statistics from the results directory.")
        domain = DomainParser(self.working_directory_path / self.domain_file_name).parse_domain()
        combined_statistics_file_path = self.working_directory_path / "results_directory" / "numeric_performance_combined_statistics.csv"
        combined_statistics_data = []
        file_path_template = "{learning_algorithm}_" + domain.name + "_numeric_learning_performance_stats_fold_{fold}_{iteration}.csv"
        self._combine_statistics_data(file_path_template, combined_statistics_data)
        with open(combined_statistics_file_path, "wt") as combined_statistics_file:
            writer = csv.DictWriter(combined_statistics_file, fieldnames=[FOLD_FIELD, *NUMERIC_PERFORMANCE_STATS])
            writer.writeheader()
            writer.writerows(combined_statistics_data)
        self.logger.info("Done collecting the statistics from the results directory!")

    def collect_statistics(self) -> None:
        self._collect_solving_statistics()
        self._collect_numeric_performance_statistics()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    args = parse_arguments()
    experiment_learning_algorithms = args.learning_algorithms.split(",")
    internal_iterations = [int(val) for val in args.internal_iterations.split(",")]
    StatisticsCollector(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        learning_algorithms=experiment_learning_algorithms,
        num_folds=args.num_folds,
        iterations=internal_iterations,
    ).collect_statistics()

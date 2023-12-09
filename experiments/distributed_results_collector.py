import argparse
import csv
import logging
from pathlib import Path
from typing import List

from utilities import LearningAlgorithmType
from validators.safe_domain_validator import SOLVING_STATISTICS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates the folds directories for the slurm tasks to work on.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithms", required=True,
                        help="the list of algorithms that will run in parallel")
    parser.add_argument("--num_folds", required=True, help="the number of folds to that were created",
                        type=int, default=5)
    parser.add_argument("--internal_iterations", required=True,
                        help="The internal iterations that the algorithm will run in parallel.")
    args = parser.parse_args()
    return args


class StatisticsCollector:
    logger: logging.Logger

    def __init__(
            self, working_directory_path: Path, domain_file_name: str, learning_algorithms: List[int] = None,
            num_folds: int = 5, iterations: List[int] = None):
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.learning_algorithms = [LearningAlgorithmType(int(e)) for e in learning_algorithms]
        self.num_folds = num_folds
        self.iterations = iterations
        self.logger = logging.getLogger("ClusterRunner")

    def collect_statistics(self) -> None:
        """Collects the statistics from the statistics files in the results directory and combines them."""
        self.logger.info("Collecting the statistics from the results directory.")
        results_directory = self.working_directory_path / "results_directory"
        combined_statistics_file_path = results_directory / "solving_combined_statistics.csv"
        combined_statistics_data = []
        for fold in range(self.num_folds):
            for iteration in self.iterations:
                for learning_algorithm in self.learning_algorithms:
                    statistics_file_path = results_directory / f"{learning_algorithm.name}_problem_solving_stats_{fold}_{iteration}.csv"
                    with open(statistics_file_path, "rt") as statistics_file:
                        reader = csv.DictReader(statistics_file)
                        combined_statistics_data.extend([{"fold": fold, **row} for row in reader])

        with open(combined_statistics_file_path, "wt") as combined_statistics_file:
            writer = csv.DictWriter(combined_statistics_file, fieldnames=SOLVING_STATISTICS)
            writer.writeheader()
            writer.writerows(combined_statistics_data)
        self.logger.info("Done collecting the statistics from the results directory!")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    args = parse_arguments()
    experiment_learning_algorithms = args.learning_algorithms.split(",")
    internal_iterations = [int(val) for val in args.internal_iterations.split(",")]
    StatisticsCollector(working_directory_path=Path(args.working_directory_path),
                        domain_file_name=args.domain_file_name,
                        learning_algorithms=experiment_learning_algorithms,
                        num_folds=args.num_folds,
                        iterations=internal_iterations).collect_statistics()

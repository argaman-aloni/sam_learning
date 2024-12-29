import argparse
import csv
import logging
from pathlib import Path
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from experiments.concurrent_execution.distributed_results_collector import DistributedResultsCollector, FOLD_FIELD
from experiments.plotting.plot_masam_results import plot_solving_results
from statistics.learning_statistics_manager import LEARNED_ACTIONS_STATS_COLUMNS
from statistics.semantic_performance_calculator import SEMANTIC_PRECISION_STATS
from utilities import LearningAlgorithmType


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates the folds directories for the slurm tasks to work on.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithms", required=True, help="the list of algorithms that will run in parallel")
    parser.add_argument("--num_folds", required=True, help="the number of folds to that were created", type=int, default=5)
    parser.add_argument("--internal_iterations", required=True, help="The internal iterations that the algorithm will run in parallel.")
    args = parser.parse_args()
    return args


class MultiAgentExperimentsResultsCollector(DistributedResultsCollector):
    logger: logging.Logger

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learning_algorithms: List[int] = None,
        num_folds: int = 5,
        iterations: List[int] = None,
    ):
        super().__init__(working_directory_path, domain_file_name, learning_algorithms, num_folds, iterations)

    def _collect_semantic_performance_statistics(self, domain: Domain) -> None:
        """Collects the semantic performance statistics from the results directory.

        :param domain: the domain to collect the statistics for.
        """
        self.logger.info("Collecting the semantic performance statistics from the results directory.")
        combined_semantic_performance_file_path = self.working_directory_path / "results_directory" / "semantic_performance_combined_statistics.csv"
        combined_semantic_performance_statistics_data = []
        file_path_template = "{learning_algorithm}_" + domain.name + "_{fold}_{iteration}_semantic_performance.csv"
        self._combine_statistics_data(
            file_path_template, combined_semantic_performance_statistics_data, exclude_algorithm=LearningAlgorithmType.ma_sam_plus
        )
        with open(combined_semantic_performance_file_path, "wt") as combined_statistics_file:
            writer = csv.DictWriter(combined_statistics_file, fieldnames=[FOLD_FIELD, *SEMANTIC_PRECISION_STATS])
            writer.writeheader()
            writer.writerows(combined_semantic_performance_statistics_data)

    def _collect_syntactic_performance_statistics(self, domain: Domain) -> None:
        """Collects the syntactic performance statistics from the results directory.

        :param domain: the domain to collect the statistics for.
        """
        self.logger.info("Collecting the syntactic performance statistics from the results directory.")
        combined_syntactic_performance_file_path = self.working_directory_path / "results_directory" / "syntactic_performance_combined_statistics.csv"
        combined_action_performance_statistics_data = []
        action_performance_path_template = "{learning_algorithm}_" + domain.name + "_action_stats_fold_{fold}_{iteration}.csv"
        self._combine_statistics_data(
            action_performance_path_template, combined_action_performance_statistics_data, exclude_algorithm=LearningAlgorithmType.ma_sam_plus
        )
        with open(combined_syntactic_performance_file_path, "wt") as combined_statistics_file:
            writer = csv.DictWriter(combined_statistics_file, fieldnames=[FOLD_FIELD, *LEARNED_ACTIONS_STATS_COLUMNS])
            writer.writeheader()
            writer.writerows(combined_action_performance_statistics_data)

    def _collect_performance_statistics(self) -> None:
        domain = DomainParser(self.working_directory_path / self.domain_file_name).parse_domain()
        self._collect_semantic_performance_statistics(domain)
        self._collect_syntactic_performance_statistics(domain)
        self.logger.info("Done collecting the statistics from the results directory!")

    def collect_statistics(self) -> None:
        self._collect_solving_statistics()
        results_directory = self.working_directory_path / "results_directory"
        combined_statistics_file_path = results_directory / "solving_combined_statistics.csv"
        plot_solving_results(combined_statistics_file_path, results_directory / "solving_combined_statistics.pdf")
        self._collect_performance_statistics()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    args = parse_arguments()
    experiment_learning_algorithms = args.learning_algorithms.split(",")
    internal_iterations = [int(val) for val in args.internal_iterations.split(",")]
    MultiAgentExperimentsResultsCollector(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        learning_algorithms=experiment_learning_algorithms,
        num_folds=args.num_folds,
        iterations=internal_iterations,
    ).collect_statistics()

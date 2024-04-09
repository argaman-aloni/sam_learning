import argparse
import logging
from pathlib import Path

from typing import List

from utilities import DistributedKFoldSplit


DEFAULT_EXPERIMENT_SIZE = 100


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates the folds directories for the slurm tasks to work on.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithms", required=True, help="the list of algorithms that will run in parallel")
    parser.add_argument("--internal_iterations", required=True, help="The internal iterations that the algorithm will run in parallel.")
    args = parser.parse_args()
    return args


class FoldsCreator:
    k_fold: DistributedKFoldSplit
    logger: logging.Logger

    def __init__(
        self, working_directory_path: Path, domain_file_name: str, learning_algorithms: List[int] = None, internal_iterations: List[int] = None
    ):
        self.k_fold = DistributedKFoldSplit(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            n_split=5,
            learning_algorithms=learning_algorithms,
            internal_iterations=internal_iterations,
        )
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.logger = logging.getLogger("cluster-folder-creator")

    def create_folds_from_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        (self.working_directory_path / "results_directory").mkdir(exist_ok=True)
        self.logger.info("Removing the old folds directories if exist.")
        self.k_fold.remove_created_directories()
        self.logger.info("Done removing the old folds directories!")
        self.logger.info("Creating the folds directories.")
        self.k_fold.create_k_fold(max_items=DEFAULT_EXPERIMENT_SIZE)
        self.logger.info("Done creating the folds directories!")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    args = parse_arguments()
    experiment_learning_algorithms = args.learning_algorithms.split(",")
    internal_iterations = [int(val) for val in args.internal_iterations.split(",")]
    if len(internal_iterations) > 0:
        print(f"Internal iterations: {internal_iterations}")

    FoldsCreator(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        learning_algorithms=experiment_learning_algorithms,
        internal_iterations=internal_iterations,
    ).create_folds_from_cross_validation()

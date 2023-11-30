import argparse
import logging
from pathlib import Path

from utilities import DistributedKFoldSplit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creates the folds directories for the slurm tasks to work on.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    args = parser.parse_args()
    return args


class FoldsCreator:
    k_fold: DistributedKFoldSplit
    logger: logging.Logger

    def __init__(self, working_directory_path: Path, domain_file_name: str):
        self.k_fold = DistributedKFoldSplit(
            working_directory_path=working_directory_path, domain_file_name=domain_file_name, n_split=5)
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.logger = logging.getLogger("ClusterRunner")

    def create_folds_from_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.k_fold.create_k_fold()
        self.logger.info("Done creating the folds directories!")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    args = parse_arguments()
    FoldsCreator(working_directory_path=Path(args.working_directory_path),
                 domain_file_name=args.domain_file_name).create_folds_from_cross_validation()

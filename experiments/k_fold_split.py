"""Create a train and test set split for the input directory for the action model learning algorithms to use."""
import logging
import random
import shutil
from pathlib import Path
from typing import Tuple, List, NoReturn, Iterator


def create_test_set_indices(array_size: int, n_split: int) -> Iterator[List[int]]:
    """Creates the indices to use to create the train and the test set of the problems.

    :param array_size: the size of the array containing the problems.
    :param n_split: the number of splits to divide the dataset to.
    :return: a list containing the indices of the problems that will be used in the test set in each fold.
    """
    random.seed(42)  # it is always good to seed something according to the most important number in the world!
    indices = list(range(array_size))
    random.shuffle(indices)
    for i in range(n_split):
        yield indices[i::n_split]


class KFoldSplit:
    """Split the working directory into a train and test set directories."""

    logger: logging.Logger
    working_directory_path: Path
    n_split: int
    train_set_dir_path: Path
    test_set_dir_path: Path
    domain_file_path: Path
    validation_directory_path: Path

    def __init__(self, working_directory_path: Path, n_split: int, domain_file_name: str):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.n_split = n_split
        self.train_set_dir_path = working_directory_path / "train"
        self.test_set_dir_path = working_directory_path / "test"
        self.validation_directory_path = working_directory_path / "validation_set"
        self.domain_file_path = working_directory_path / domain_file_name

    def _copy_domain(self) -> NoReturn:
        """Copies the domain to the train set directory so that it'd be used in the learning process."""
        self.logger.debug("Copying the domain to the train set directory.")
        shutil.copy(self.domain_file_path, self.train_set_dir_path / self.domain_file_path.name)

    def remove_created_directories(self) -> NoReturn:
        """Deletes the train and test directories."""
        self.logger.debug("Deleting the train set directory!")
        shutil.rmtree(self.train_set_dir_path)
        self.logger.debug("Deleting the test set directory!")
        shutil.rmtree(self.test_set_dir_path)

    def create_k_fold(self) -> Iterator[Tuple[Path, Path]]:
        """Creates a generator that will be used for the next algorithm to know where the train and test set
            directories reside.

        :return: a generator for the train and test set directories.
        """
        self.logger.info("Starting to create the folds for the cross validation process.")
        trajectory_paths = []
        problem_paths = []
        for trajectory_file_path in self.working_directory_path.glob("*.trajectory"):
            trajectory_paths.append(trajectory_file_path)
            problem_paths.append(self.working_directory_path / f"{trajectory_file_path.stem}.pddl")

        for test_set_indices in create_test_set_indices(len(problem_paths), self.n_split):
            self.train_set_dir_path.mkdir(exist_ok=True)
            self.test_set_dir_path.mkdir(exist_ok=True)
            self.validation_directory_path.mkdir(exist_ok=True)

            self._copy_domain()
            test_set_problems = [problem_paths[i] for i in test_set_indices]
            train_set_problems = list(filter(lambda x: x not in test_set_problems, problem_paths))
            train_set_trajectories = [trajectory_paths[i] for i in range(len(trajectory_paths)) if
                                      i not in test_set_indices]
            self.logger.info(f"Created a new fold - train set has the following problems: {train_set_problems} "
                             f"and test set had the following problems: {test_set_problems}")
            for problem in test_set_problems:
                shutil.copy(problem, self.test_set_dir_path / problem.name)

            for trajectory, problem in zip(train_set_trajectories, train_set_problems):
                shutil.copy(trajectory, self.train_set_dir_path / trajectory.name)
                shutil.copy(problem, self.train_set_dir_path / problem.name)

            self.logger.debug("Finished creating fold!")
            yield self.train_set_dir_path, self.test_set_dir_path

            self.remove_created_directories()

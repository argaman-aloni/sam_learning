"""Create a train and test set split for the input directory for the action model learning algorithms to use."""
import logging
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

from utilities.k_fold_split import create_test_set_indices, save_fold_settings, FOLDS_LABEL, load_fold_settings


class DistributedKFoldSplit:
    """Split the working directory into a train and test set directories."""

    logger: logging.Logger
    working_directory_path: Path
    n_split: int
    train_set_dir_path: Path
    test_set_dir_path: Path
    domain_file_path: Path
    only_train_test: bool

    def __init__(self, working_directory_path: Path, domain_file_name: str, n_split: int = 0,
                 only_train_test: bool = False, learning_algorithms: List[int] = None,
                 internal_iterations: List[int] = None):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.n_split = n_split
        self.train_set_dir_path = working_directory_path / "train"
        self.test_set_dir_path = working_directory_path / "test"
        self.domain_file_path = working_directory_path / domain_file_name
        self.only_train_test = only_train_test
        self._learning_algorithms = learning_algorithms
        self._internal_iterations = internal_iterations

    def _copy_domain(self, directory_path: Path) -> None:
        """Copies the domain to the train set directory so that it'd be used in the learning process."""
        self.logger.debug("Copying the domain to the train set directory.")
        shutil.copy(self.domain_file_path, directory_path / self.domain_file_path.name)

    def remove_created_directories(self) -> None:
        """Deletes the train and test directories."""

        self.logger.debug("Deleting the train set directory!")
        shutil.rmtree(self.train_set_dir_path, ignore_errors=True)
        self.logger.debug("Deleting the test set directory!")
        shutil.rmtree(self.test_set_dir_path, ignore_errors=True)

    def create_directories_content(
            self, train_set_problems: List[Path], test_set_problems: List[Path], trajectory_paths: List[Path],
            fold_index: int, learning_algorithm: int, internal_iteration: Optional[int] = None,
            selected_training_trajectories: Optional[List[Path]] = None) -> Tuple[Path, Path]:
        """Creates the content of the train and test set directories."""
        fold_train_dir_path = (
                self.train_set_dir_path / f"fold_{fold_index}_{learning_algorithm}"
                                          f"{f'_{internal_iteration}' if internal_iteration is not None else ''}")
        fold_test_dir_path = (
                self.test_set_dir_path / f"fold_{fold_index}_{learning_algorithm}"
                                         f"{f'_{internal_iteration}' if internal_iteration is not None else ''}")
        fold_train_dir_path.mkdir(exist_ok=True)
        fold_test_dir_path.mkdir(exist_ok=True)
        self._copy_domain(fold_train_dir_path)

        for problem in test_set_problems:
            shutil.copy(problem, fold_test_dir_path / problem.name)

        if selected_training_trajectories is not None:
            for trajectory in selected_training_trajectories:
                shutil.copy(trajectory, fold_train_dir_path / trajectory.name)
                problem_path = self.working_directory_path / f"{trajectory.stem}.pddl"
                shutil.copy(problem_path, fold_train_dir_path / problem_path.name)
        else:
            for trajectory, problem in zip(trajectory_paths, train_set_problems):
                shutil.copy(trajectory, fold_train_dir_path / trajectory.name)
                shutil.copy(problem, fold_train_dir_path / problem.name)

        return fold_train_dir_path, fold_test_dir_path

    def create_k_fold(self, trajectory_suffix: str = "*.trajectory",
                      max_items: int = 0, load_configuration: bool = True) -> List[Tuple[Path, Path]]:
        """Creates a generator that will be used for the next algorithm to know where the train and test set
            directories reside.

        :param trajectory_suffix: the suffix of the trajectory files to be used.
        :param max_items: the maximum number of items to be used in the train and test set together.
        :param load_configuration: whether to load folds settings from the configuration file.
        :return: a generator for the train and test set directories.
        """
        self.logger.info("Starting to create the folds for the cross validation process.")
        self.train_set_dir_path.mkdir(exist_ok=True)
        self.test_set_dir_path.mkdir(exist_ok=True)
        folds_data = load_fold_settings(self.working_directory_path)
        train_test_paths = []
        if load_configuration and len(folds_data) > 0:
            self.logger.info("Loading the folds settings from the configuration file.")
            for index, (fold_name, fold_content) in enumerate(folds_data.items()):
                train_set_trajectories = [
                    self.working_directory_path / f"{problem_path.stem}.trajectory" for
                    problem_path in fold_content["train"]]
                if self._internal_iterations is not None:
                    for internal_iteration in self._internal_iterations:
                        selected_training_trajectories = random.sample(
                            train_set_trajectories, k=internal_iteration if internal_iteration > 0 else 1)
                        for learning_algorithm in self._learning_algorithms:
                            self.logger.debug("Creating fold directories content for each learning algorithm.")
                            train_test_paths.append(self.create_directories_content(
                                fold_content["train"], fold_content["test"], train_set_trajectories, index,
                                learning_algorithm, internal_iteration, selected_training_trajectories))

                    else:
                        for learning_algorithm in self._learning_algorithms:
                            train_test_paths.append(self.create_directories_content(
                                fold_content["train"], fold_content["test"], train_set_trajectories, index,
                                learning_algorithm))

                self.logger.debug("Finished creating fold!")

            return train_test_paths

        problem_paths = []
        trajectory_paths = list(self.working_directory_path.glob(trajectory_suffix))
        trajectory_paths.sort()  # sort the trajectories so that the same order is used each time the algorithm runs
        items_per_fold = max_items if (0 < max_items <= len(trajectory_paths)) else len(trajectory_paths)
        trajectory_paths = random.sample(trajectory_paths, k=items_per_fold)
        for trajectory_file_path in trajectory_paths:
            problem_paths.append(self.working_directory_path / f"{trajectory_file_path.stem}.pddl")

        num_splits = len(trajectory_paths) if self.n_split == 0 else self.n_split
        for fold_index, (train_set_indices, test_set_indices) in enumerate(create_test_set_indices(
                len(problem_paths), num_splits, self.only_train_test)):
            test_set_problems = [problem_paths[i] for i in test_set_indices]
            train_set_problems = [problem_paths[i] for i in train_set_indices]
            train_set_trajectories = [trajectory_paths[i] for i in range(len(trajectory_paths)) if
                                      i not in test_set_indices]
            self.logger.info(f"Created a new fold - train set has the following problems: {train_set_problems} "
                             f"and test set had the following problems: {test_set_problems}")
            for learning_algorithm in self._learning_algorithms:
                train_test_paths.append(
                    self.create_directories_content(train_set_problems, test_set_problems, train_set_trajectories,
                                                    fold_index, learning_algorithm))

            folds_data[f"{FOLDS_LABEL}_{fold_index}"] = {
                "train": [str(p.absolute()) for p in train_set_problems],
                "test": [str(p.absolute()) for p in test_set_problems]
            }
            self.logger.debug("Finished creating fold!")

        save_fold_settings(self.working_directory_path, folds_data)
        return train_test_paths

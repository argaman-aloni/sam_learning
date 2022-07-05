"""Module responsible for running the experiments using the PlanMiner algorithm."""
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, List

from pddl_plus_parser.lisp_parsers import DomainParser

from experiments.k_fold_split import KFoldSplit
from utilities import LearningAlgorithmType
from validators import DomainValidator

DEFAULT_SPLIT = 5
PLAN_MINER_FILE_PATH = "path/to/plan_miner.py"  # TODO: Fix this.


class PlanMinerExperimentRunner:
    """Runs the experiment using PlanMiner algorithm."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    domain_validator: DomainValidator

    def __init__(self, working_directory_path: Path, domain_file_name: str):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path,
                                 domain_file_name=domain_file_name,
                                 n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self.domain_validator = DomainValidator(
            self.working_directory_path, LearningAlgorithmType.plan_miner,
            self.working_directory_path / domain_file_name)

    def concatenate_trajectories(self, train_set_dir_path: Path, allowed_observations: List[Path]) -> Path:
        """Concatenates the trajectories into a single file.

        :param train_set_dir_path: the directory containing the trajectories.
        :param allowed_observations: the paths to the trajectories that are to be concatenated.
        :return: the path to the concatenated trajectories file.
        """
        self.logger.info("Concatenating the trajectories into a single file!")
        observations_content = []
        for observation_path in allowed_observations:
            with open(observation_path, "r") as observation_file:
                observations_content.append(observation_file.read())

        output_file_path = train_set_dir_path / "concatenated_trajectories.pts"
        with open(output_file_path, "wt") as output_file:
            output_file.write("\n".join(observations_content))

        return output_file_path

    def run_plan_miner(self, concatenated_trajectory_path: Path, domain_name: str) -> Path:
        """Runs the PlanMiner algorithm and learns an action model from the trajectories.

        :param concatenated_trajectory_path: the path to the concatenated trajectories file.
        :param domain_name: the name of the domain that will be given to both the file and the domain PDDL object.
        :return: the path to the newly learned domain file.
        """
        self.logger.info("Running PlanMiner on the input trajectories!")
        process = subprocess.run([PLAN_MINER_FILE_PATH, concatenated_trajectory_path, domain_name], capture_output=True)
        if process.returncode != 0:
            self.logger.error(f"PlanMiner failed with the following error: {process.stderr}")
            raise Exception("PlanMiner failed!")

        self.logger.info("PlanMiner finished successfully!")
        return Path(os.getcwd()) / f"{domain_name}.pddl"

    def copy_domain(self, learned_domain_path: Path, test_set_dir_path: Path) -> Path:
        """Copies the learned domain to the test set directory.

        :param learned_domain_path: the path to the learned domain file.
        :param test_set_dir_path: the path to the test set directory.
        :return: the path to the copied domain file.
        """
        self.logger.info("Copying the learned domain to the test set directory!")
        shutil.copy(learned_domain_path, test_set_dir_path)
        learned_domain_path.unlink()
        return test_set_dir_path / learned_domain_path.name

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> NoReturn:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_observations = []
        for trajectory_file_path in train_set_dir_path.glob("*.pts"):
            allowed_observations.append(trajectory_file_path)
            concatenated_trajectory_path = self.concatenate_trajectories(allowed_observations)
            self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
            learned_domain_path = self.run_plan_miner(concatenated_trajectory_path, domain_name=partial_domain.name)
            self.validate_learned_domain(allowed_observations, learned_domain_path, test_set_dir_path)

        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(
            self, allowed_observations: List[Path], learned_domain_path: Path,
            test_set_dir_path: Path) -> NoReturn:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_domain_path: the path to the domain learned by the algorithm.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        """
        domain_file_path = self.copy_domain(learned_domain_path, test_set_dir_path)
        self.logger.debug("Checking that the test set problems can solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations)

        domain_file_path.unlink()

    def run_cross_validation(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()


def main():
    args = sys.argv
    working_directory_path = Path(args[1])
    domain_file_name = args[2]
    PlanMinerExperimentRunner(working_directory_path=working_directory_path,
                              domain_file_name=domain_file_name).run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

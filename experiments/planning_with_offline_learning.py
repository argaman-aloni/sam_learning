"""The POL main framework - Compile, Learn and Plan."""
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import NoReturn, Dict, Union, Type

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser

from sam_learning.learners import SAMLearner, NumericSAMLearner
from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from solvers.metric_ff_solver import MetricFFSolver

DEFAULT_SPLIT = 3


class LearningAlgorithmType(Enum):
    sam_learning = 1
    esam_learning = 2
    numeric_sam = 3


class SolverType(Enum):
    fast_downward = 1
    metric_ff = 2


class POL:
    """Class that represents the POL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learning_algorithm: LearningAlgorithmType
    learning_statistics_manager: LearningStatisticsManager
    _learning_algorithm_options: Dict[LearningAlgorithmType, Type[Union[SAMLearner, NumericSAMLearner]]]

    def __init__(self, working_directory_path: Path, domain_file_name: str, learning_algorithm: LearningAlgorithmType):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path, n_split=DEFAULT_SPLIT,
                                 domain_file_name=domain_file_name)
        self.domain_file_name = domain_file_name
        self.learning_algorithm = learning_algorithm
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm.name)
        self._learning_algorithm_options = {
            LearningAlgorithmType.sam_learning: SAMLearner,
            LearningAlgorithmType.numeric_sam: NumericSAMLearner
        }

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path):
        """

        :param fold_num:
        :param learning_algorithm:
        :return:
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_observations = []
        for trajectory_file_path in train_set_dir_path.glob("*.trajectory"):
            problem_name = trajectory_file_path.stem
            problem = ProblemParser(train_set_dir_path / f"{problem_name}.pddl", partial_domain).parse_problem()
            new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
            allowed_observations.append(new_observation)
            learner = self._learning_algorithm_options[self.learning_algorithm](partial_domain=partial_domain)
            learned_model, learning_report = learner.learn_action_model(allowed_observations)
            self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model)

    def run_cross_validation(self, solver_type: SolverType) -> NoReturn:
        """

        :param solver_type:
        :return:
        """
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")



if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(level=logging.INFO)
    offline_learner = POL(working_directory_path=Path("C:\Argaman\Planning\Minecraft\IPC3\Tests1\Depots\\Numeric"),
                          domain_file_name="depot_numeric.pddl",
                          learning_algorithm=LearningAlgorithmType.numeric_sam)
    offline_learner.run_cross_validation(None)

"""The POL main framework - Compile, Learn and Plan."""
import logging
import sys
from pathlib import Path
from typing import NoReturn, List

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser

from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from validators.safe_domain_validator import SafeDomainValidator
from experiments.util_types import LearningAlgorithmType, SolverType
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner

DEFAULT_SPLIT = 3

LEARNING_ALGORITHMS = {
    LearningAlgorithmType.sam_learning: SAMLearner,
    LearningAlgorithmType.numeric_sam: NumericSAMLearner
}

SAFE_LEARNER_TYPES = [LearningAlgorithmType.sam_learning, LearningAlgorithmType.numeric_sam]


class POL:
    """Class that represents the POL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learning_statistics_manager: LearningStatisticsManager
    _learning_algorithm: LearningAlgorithmType
    _solver: SolverType
    domain_validator: SafeDomainValidator  # TODO: add unsafe domain validators.

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 learning_algorithm: LearningAlgorithmType, solver: SolverType):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path, n_split=DEFAULT_SPLIT,
                                 domain_file_name=domain_file_name)
        self.domain_file_name = domain_file_name
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm)
        self._learning_algorithm = learning_algorithm
        self._solver = solver
        if learning_algorithm in SAFE_LEARNER_TYPES:
            self.domain_validator = SafeDomainValidator(self.working_directory_path, solver, learning_algorithm)
        else:
            self.domain_validator = None
            # TODO: add unsafe domain validation.

    def export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        """
        domain_path = test_set_path / self.domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

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
        validation_problems = []
        for trajectory_file_path in train_set_dir_path.glob("*.trajectory"):
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            validation_problems.append(problem_path)
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
            allowed_observations.append(new_observation)
            learner = LEARNING_ALGORITHMS[self._learning_algorithm](partial_domain=partial_domain)
            learned_model, learning_report = learner.learn_action_model(allowed_observations)
            self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model, learning_report)
            self.validate_learned_domain(allowed_observations, learned_model, test_set_dir_path, validation_problems)

        if self._learning_algorithm == LearningAlgorithmType.numeric_sam:
            self.learning_statistics_manager.export_numeric_learning_statistics(fold_number=fold_num)

        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(self, allowed_observations: List[Observation], learned_model,
                                test_set_dir_path, validation_problems) -> NoReturn:
        """

        :param allowed_observations:
        :param learned_model:
        :param test_set_dir_path:
        :param validation_problems:
        :return:
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        self.domain_validator.copy_validation_problems(domain_file_path, validation_problems)
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              used_observations=allowed_observations,
                                              test_set_directory_path=self.working_directory_path / "validation_set",
                                              is_validation=True)
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations)

    def run_cross_validation(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")


if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(level=logging.INFO)
    offline_learner = POL(
        working_directory_path=Path("/sise/home/mordocha/numeric_planning/domains/IPC3/Tests1/Depots"),
        domain_file_name="depot_numeric.pddl",
        learning_algorithm=LearningAlgorithmType.numeric_sam,
        solver=SolverType.metric_ff)
    offline_learner.run_cross_validation()

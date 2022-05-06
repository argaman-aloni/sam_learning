"""The POL main framework - Compile, Learn and Plan."""
import json
import logging
import sys
from pathlib import Path
from typing import NoReturn, List, Optional, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation

from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from experiments.util_types import LearningAlgorithmType, SolverType
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner
from validators import DomainValidator

DEFAULT_SPLIT = 4

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
    debug: bool
    domain_validator: DomainValidator
    fluents_map: Dict[str, List[str]]

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 learning_algorithm: LearningAlgorithmType, solver: SolverType, fluents_map_path: Optional[Path]):
        self.logger = logging.getLogger(__name__)
        self.debug = False
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path,
                                 domain_file_name=domain_file_name)
        self.domain_file_name = domain_file_name
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm)
        self._learning_algorithm = learning_algorithm
        self._solver = solver
        if fluents_map_path is not None:
            with open(fluents_map_path, "rt") as json_file:
                self.fluents_map = json.load(json_file)

        self.domain_validator = DomainValidator(
            self.working_directory_path, solver, learning_algorithm, self.working_directory_path / domain_file_name)

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
            self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
            learner = LEARNING_ALGORITHMS[self._learning_algorithm](partial_domain=partial_domain,
                                                                    preconditions_fluent_map=self.fluents_map)
            learned_model, learning_report = learner.learn_action_model(allowed_observations)
            self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model, learning_report)
            self.validate_learned_domain(allowed_observations, learned_model, test_set_dir_path, validation_problems)

        if self._learning_algorithm == LearningAlgorithmType.numeric_sam:
            self.learning_statistics_manager.export_numeric_learning_statistics(fold_number=fold_num)

        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(self, allowed_observations: List[Observation], learned_model: LearnerDomain,
                                test_set_dir_path: Path, validation_problems: List[Path]) -> NoReturn:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param validation_problems: the problems to use as validation set.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        if self.debug:
            self.domain_validator.copy_validation_problems(domain_file_path, validation_problems)
            self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                                  used_observations=allowed_observations,
                                                  test_set_directory_path=self.working_directory_path / "validation_set",
                                                  is_validation=True)
            self.domain_validator.clear_validation_problems()

        self.logger.debug("Checking that the test set problems can solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations,
                                              is_validation=False)

    def run_cross_validation(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()


def main():
    args = sys.argv
    working_directory_path = Path(args[1])
    domain_file_name = args[2]
    learning_algorithm = LearningAlgorithmType.numeric_sam
    solver = SolverType.metric_ff
    fluents_map_path = Path(args[3])
    offline_learner = POL(working_directory_path=working_directory_path,
                          domain_file_name=domain_file_name,
                          learning_algorithm=learning_algorithm,
                          solver=solver,
                          fluents_map_path=fluents_map_path)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

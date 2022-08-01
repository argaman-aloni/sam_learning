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
from experiments.numeric_performance_calculator import NumericPerformanceCalculator
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner, PolynomialSAMLearning
from utilities import LearningAlgorithmType
from validators import DomainValidator

DEFAULT_SPLIT = 5
MAX_FOLD_SIZE = 65

NUMERIC_ALGORITHMS = [LearningAlgorithmType.numeric_sam, LearningAlgorithmType.plan_miner,
                      LearningAlgorithmType.polynomial_sam, LearningAlgorithmType.raw_numeric_sam]

LEARNING_ALGORITHMS = {
    LearningAlgorithmType.sam_learning: SAMLearner,
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    # difference is that the learner is not given any fluents to assist in learning
    LearningAlgorithmType.raw_numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.polynomial_sam: PolynomialSAMLearning,
}


class POL:
    """Class that represents the POL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learning_statistics_manager: LearningStatisticsManager
    _learning_algorithm: LearningAlgorithmType
    domain_validator: DomainValidator
    fluents_map: Dict[str, List[str]]
    numeric_performance_calc: NumericPerformanceCalculator

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 learning_algorithm: LearningAlgorithmType, fluents_map_path: Optional[Path],
                 use_metric_ff: bool = False):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path,
                                 domain_file_name=domain_file_name,
                                 n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm)
        self._learning_algorithm = learning_algorithm
        if fluents_map_path is not None:
            with open(fluents_map_path, "rt") as json_file:
                self.fluents_map = json.load(json_file)

        else:
            self.fluents_map = None

        self.numeric_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name,
            use_metric_ff=use_metric_ff)

    def _init_numeric_performance_calculator(self) -> NoReturn:
        """Initializes the algorithm of the numeric precision / recall calculator."""
        if self._learning_algorithm not in NUMERIC_ALGORITHMS:
            return

        domain_path = self.working_directory_path / self.domain_file_name
        model_domain = partial_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
        observations = []
        for trajectory_file_path in self.working_directory_path.glob("*.trajectory"):
            problem_path = self.working_directory_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
            observations.append(new_observation)

        self.numeric_performance_calc = NumericPerformanceCalculator(model_domain=model_domain,
                                                                     observations=observations,
                                                                     working_directory_path=self.working_directory_path,
                                                                     learning_algorithm=self._learning_algorithm)

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
        observed_objects = {}
        learned_domain_path = None
        for index, trajectory_file_path in enumerate(train_set_dir_path.glob("*.trajectory")):
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            observed_objects.update(problem.objects)
            new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
            allowed_observations.append(new_observation)
            if index % 2 != 0:
                self.logger.info(f"Skipping the iteration {index} to save the total amount of time!")
                continue

            self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
            learner = LEARNING_ALGORITHMS[self._learning_algorithm](partial_domain=partial_domain,
                                                                    preconditions_fluent_map=self.fluents_map)
            learned_model, learning_report = learner.learn_action_model(allowed_observations)
            self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model, learning_report)
            learned_domain_path = self.validate_learned_domain(allowed_observations, learned_model, test_set_dir_path)

        self.numeric_performance_calc.calculate_performance(learned_domain_path, len(allowed_observations))
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(self, allowed_observations: List[Observation], learned_model: LearnerDomain,
                                test_set_dir_path: Path) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        self.logger.debug("Checking that the test set problems can solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations)

        return domain_file_path

    def run_cross_validation(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        self._init_numeric_performance_calculator()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold(max_items=MAX_FOLD_SIZE)):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()
        if self._learning_algorithm in NUMERIC_ALGORITHMS:
            self.numeric_performance_calc.export_numeric_learning_performance()
            self.learning_statistics_manager.write_complete_joint_statistics()


def main():
    args = sys.argv
    working_directory_path = Path(args[1])
    domain_file_name = args[2]
    learning_algorithm = LearningAlgorithmType.numeric_sam
    if len(args) > 3:
        fluents_map_path = Path(args[3])
    else:
        fluents_map_path = None

    offline_learner = POL(working_directory_path=working_directory_path,
                          domain_file_name=domain_file_name,
                          learning_algorithm=learning_algorithm,
                          fluents_map_path=fluents_map_path,
                          use_metric_ff=False)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

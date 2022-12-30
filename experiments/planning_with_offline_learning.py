"""The POL main framework - Compile, Learn and Plan."""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation

from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from experiments.numeric_performance_calculator import NumericPerformanceCalculator
from experiments.utils import init_numeric_performance_calculator
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner, PolynomialSAMLearning, ConditionalSAM
from utilities import LearningAlgorithmType, SolverType
from validators import DomainValidator

DEFAULT_SPLIT = 5

NUMERIC_ALGORITHMS = [LearningAlgorithmType.numeric_sam, LearningAlgorithmType.plan_miner,
                      LearningAlgorithmType.polynomial_sam, LearningAlgorithmType.raw_numeric_sam,
                      LearningAlgorithmType.conditional_sam]

LEARNING_ALGORITHMS = {
    LearningAlgorithmType.sam_learning: SAMLearner,
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    # difference is that the learner is not given any fluents to assist in learning
    LearningAlgorithmType.raw_numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.polynomial_sam: PolynomialSAMLearning,
    LearningAlgorithmType.conditional_sam: ConditionalSAM,
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
    max_num_antecedents: int

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 learning_algorithm: LearningAlgorithmType, fluents_map_path: Optional[Path],
                 solver_type: SolverType, max_num_antecedents: int):
        self.logger = logging.getLogger(__name__)
        self.max_num_antecedents = max_num_antecedents
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
            solver_type=solver_type)

    def _init_numeric_performance_calculator(self) -> None:
        """Initializes the algorithm of the numeric precision / recall calculator."""
        if self._learning_algorithm not in NUMERIC_ALGORITHMS:
            return

        self.numeric_performance_calc = init_numeric_performance_calculator(self.working_directory_path,
                                                                            self.domain_file_name,
                                                                            self._learning_algorithm)

    def export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path,
                              file_name: Optional[str] = None) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        :param file_name: the name of the file to export the domain to.
        """
        domain_file_name = file_name or self.domain_file_name
        domain_path = test_set_path / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
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
            if (index + 1) % 10 != 0:
                continue

            self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
            learner = LEARNING_ALGORITHMS[self._learning_algorithm](partial_domain=partial_domain,
                                                                    preconditions_fluent_map=self.fluents_map,
                                                                    max_antecedents_size=self.max_num_antecedents)
            learned_model, learning_report = learner.learn_action_model(allowed_observations)
            self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model, learning_report)
            learned_domain_path = self.validate_learned_domain(allowed_observations, learned_model, test_set_dir_path)

        if self._learning_algorithm in NUMERIC_ALGORITHMS:
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
        self.export_learned_domain(learned_model, self.working_directory_path / "results_directory",
                                   f"{learned_model.name}_{len(allowed_observations)}_trajectories.pddl")
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations)

        return domain_file_path

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        self._init_numeric_performance_calculator()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()
        if self._learning_algorithm in NUMERIC_ALGORITHMS:
            self.numeric_performance_calc.export_numeric_learning_performance()
            self.learning_statistics_manager.write_complete_joint_statistics()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the POL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithm", required=True, type=int, choices=[1, 2, 3, 4, 6, 9],
                        help="The type of learning algorithm. "
                             "\n 1: sam_learning\n2: esam_learning\n3: numeric_sam\n4: raw_numeric_sam\n"
                             "6: polynomial_sam\n 9: conditional_sam")
    parser.add_argument("--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' "
                                                                   "fluents", default=None)
    parser.add_argument("--solver_type", required=False, type=int, choices=[1, 2, 3],
                        help="The solver that should be used for the sake of validation", default=3)
    parser.add_argument("--max_antecedent_size", required=False, type=int, help="The maximum antecedent size", default=1)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    offline_learner = POL(working_directory_path=Path(args.working_directory_path),
                          domain_file_name=args.domain_file_name,
                          learning_algorithm=LearningAlgorithmType(args.learning_algorithm),
                          fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
                          solver_type=SolverType(args.solver_type),
                          max_num_antecedents=args.max_antecedent_size or 0)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

"""Runs experiments for the numeric model learning algorithms."""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pddl_plus_parser.models import Observation, Domain

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner, PolynomialSAMLearning
from utilities import LearningAlgorithmType, SolverType
from validators import DomainValidator

LEARNING_ALGORITHMS = {
    LearningAlgorithmType.numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.raw_numeric_sam: NumericSAMLearner,
    LearningAlgorithmType.polynomial_sam: PolynomialSAMLearning,
    LearningAlgorithmType.raw_polynomial_nam: PolynomialSAMLearning,
}

NO_INSIGHT_NUMERIC_ALGORITHMS = [
    LearningAlgorithmType.raw_numeric_sam.value,
    LearningAlgorithmType.raw_polynomial_nam.value,
]


class OfflineNumericExperimentRunner(OfflineBasicExperimentRunner):
    """Class to conduct offline numeric action model learning experiments."""

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 learning_algorithm: LearningAlgorithmType, fluents_map_path: Optional[Path],
                 solver_type: SolverType, problem_prefix: str = "pfile"):
        super().__init__(working_directory_path=working_directory_path, domain_file_name=domain_file_name,
                         learning_algorithm=learning_algorithm, solver_type=solver_type, problem_prefix=problem_prefix)
        if learning_algorithm.value in NO_INSIGHT_NUMERIC_ALGORITHMS:
            self.fluents_map = None

        else:
            with open(fluents_map_path, "rt") as json_file:
                self.fluents_map = json.load(json_file)

        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name,
            solver_type=solver_type, preoblem_prefix=problem_prefix)

    def _apply_learning_algorithm(
            self, partial_domain: Domain, allowed_observations: List[Observation],
            test_set_dir_path: Path) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning statistics.
        """

        learner = LEARNING_ALGORITHMS[self._learning_algorithm](
            partial_domain=partial_domain, preconditions_fluent_map=self.fluents_map)
        return learner.learn_action_model(allowed_observations)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs the numeric action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True,
                        help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--learning_algorithm", required=True, type=int, choices=[3, 4, 6, 14],
                        help="The type of learning algorithm. "
                             "\n3: numeric_sam\n4: raw_numeric_sam\n 6: polynomial_sam\n ")
    parser.add_argument("--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' "
                                                                   "fluents", default=None)
    parser.add_argument("--solver_type", required=False, type=int, choices=[2, 3],
                        help="The solver that should be used for the sake of validation.\nMetric-FF - 2, ENHSP - 3.",
                        default=3)
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names",
                        type=str, default="pfile")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    offline_learner = OfflineNumericExperimentRunner(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        learning_algorithm=LearningAlgorithmType(args.learning_algorithm),
        fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
        solver_type=SolverType(args.solver_type),
        problem_prefix=args.problems_prefix)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

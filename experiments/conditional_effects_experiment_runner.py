"""Runs experiments for the numeric model learning algorithms."""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pddl_plus_parser.models import Observation, Domain

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner, configure_logger
from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner, PolynomialSAMLearning, UniversallyConditionalSAM
from utilities import LearningAlgorithmType, SolverType
from validators import DomainValidator


class OfflineConditionalEffectsExperimentRunner(OfflineBasicExperimentRunner):
    """Class to conduct offline numeric action model learning experiments."""

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        universals_map_path: Optional[Path],
        max_num_antecedents: int,
        problem_prefix: str = "pfile",
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=LearningAlgorithmType.universal_sam,
            solver_type=SolverType.metric_ff,
            problem_prefix=problem_prefix,
        )
        self.max_num_antecedents = max_num_antecedents
        if universals_map_path is not None:
            with open(universals_map_path, "rt") as json_file:
                self.action_to_universals_map = json.load(json_file)

        else:
            self.action_to_universals_map = None

        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path,
            LearningAlgorithmType.universal_sam,
            self.working_directory_path / domain_file_name,
            problem_prefix=problem_prefix,
        )

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning statistics.
        """

        learner = UniversallyConditionalSAM(
            partial_domain=partial_domain, max_antecedents_size=self.max_num_antecedents, universals_map=self.action_to_universals_map
        )
        return learner.learn_action_model(allowed_observations)

    def run_fold(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Runs the numeric action model learning algorithms on the input fold.

        :param fold_num: the number of the fold to run.
        :param train_set_dir_path: the path to the directory containing the training set problems.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        """
        self._init_semantic_performance_calculator(test_set_path=test_set_dir_path)
        self.learn_model_offline(fold_num, train_set_dir_path, test_set_dir_path)
        self.domain_validator.clear_statistics()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the numeric action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--universals_map", required=False, help="The path to the file mapping indicating whether there are universals or not", default=None
    )
    parser.add_argument(
        "--learning_algorithm",
        required=True,
        type=int,
        help="The type of learning algorithm. " "\n3: numeric_sam\n4: raw_numeric_sam\n 6: polynomial_sam\n ",
    )
    parser.add_argument("--max_antecedent_size", required=False, type=int, help="The maximum antecedent size", default=1)
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    configure_logger(args)
    working_directory_path = Path(args.working_directory_path)
    offline_learner = OfflineConditionalEffectsExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        universals_map_path=Path(args.universals_map) if args.universals_map else None,
        max_num_antecedents=args.max_antecedent_size or 0,
        problem_prefix=args.problems_prefix,
    )
    offline_learner.run_fold(
        fold_num=args.fold_number,
        train_set_dir_path=(working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}",
        test_set_dir_path=(working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}",
    )


if __name__ == "__main__":
    main()

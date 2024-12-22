"""Runs experiments for the numeric model learning algorithms."""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pddl_plus_parser.models import Observation, Domain

from experiments.concurrent_execution.parallel_basic_experiment_runner import configure_iteration_logger
from experiments.concurrent_execution.parallel_numeric_experiment_runner import SingleIterationNSAMExperimentRunner
from experiments.experiments_consts import NUMERIC_SAM_ALGORITHM_VERSIONS
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType, NegativePreconditionPolicy
from validators import DomainValidator


class SingleIterationTripletsNSAMExperimentRunner(SingleIterationNSAMExperimentRunner):
    """Class to conduct offline numeric action model learning experiments."""

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        polynom_degree: int,
        learning_algorithm: LearningAlgorithmType,
        fluents_map_path: Optional[Path],
        problem_prefix: str = "pfile",
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=learning_algorithm,
            problem_prefix=problem_prefix,
            polynom_degree=polynom_degree,
            fluents_map_path=fluents_map_path,
            running_triplets_experiment=True,
        )
        self.fluents_map = None
        if fluents_map_path is not None:
            with open(fluents_map_path, "rt") as json_file:
                self.fluents_map = json.load(json_file)

        self.polynom_degree = polynom_degree
        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name, problem_prefix=problem_prefix,
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

        learner = NUMERIC_SAM_ALGORITHM_VERSIONS[self._learning_algorithm](
            partial_domain=partial_domain, polynomial_degree=self.polynom_degree, relevant_fluents=self.fluents_map
        )
        return learner.learn_action_model(allowed_observations)

    def _learn_model_offline(
        self, allowed_observations: List[Observation], partial_domain: Domain, test_set_dir_path: Path, fold_num: int,
    ):
        """

        :param allowed_observations:
        :param partial_domain:
        :param test_set_dir_path:
        :param fold_num:
        :return:
        """
        # For now, we are only interested in the no-remove policy.
        policy = NegativePreconditionPolicy.no_remove
        learned_domain, learning_report = self._apply_learning_algorithm(partial_domain, allowed_observations, test_set_dir_path)
        self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_domain, learning_report, policy=policy)
        learned_domain_path = self.validate_learned_domain(
            allowed_observations, learned_domain, test_set_dir_path, fold_num, float(learning_report["learning_time"]), policy
        )
        self.semantic_performance_calc.calculate_performance(learned_domain_path, sum([len(observation) for observation in allowed_observations]))
        self.logger.info(f"Finished the learning phase for the fold - {fold_num} and {len(allowed_observations)} observations!")

    def run_action_triplets_experiment(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Runs the experiment while iterating on the action triplets instead of on the trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: This assumes that the folder contains all the train set.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Executing the experiments on the action triplets instead of the trajectories for the fold - {fold_num}!")
        self._init_semantic_performance_calculator(fold_num)
        partial_domain = self.read_domain_file(train_set_dir_path)
        complete_train_set = self.collect_observations(train_set_dir_path, partial_domain)
        transitions_based_training_set = self.create_transitions_based_training_set(complete_train_set)
        execution_scheme = [index + 1 for index in range(10)] + [index for index in range(20, len(transitions_based_training_set), 10)]
        i = 0
        for index in execution_scheme:
            self._learn_model_offline([*transitions_based_training_set[0:index]], partial_domain, test_set_dir_path, fold_num)
            i += 1
            if i == 2:
                break

        self.semantic_performance_calc.export_semantic_performance(fold_num)
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the numeric action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--learning_algorithm", required=True, type=int, choices=[3, 15, 19, 20, 21],
    )
    parser.add_argument(
        "--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' " "fluents", default=None,
    )
    parser.add_argument(
        "--solver_type",
        required=False,
        type=int,
        choices=[2, 3],
        help="The solver that should be used for the sake of validation.\nMetric-FF - 2, ENHSP - 3.",
        default=3,
    )
    parser.add_argument(
        "--polynom_degree", required=False, help="The degree of the polynomial to set in the learning algorithm.", default=0,
    )
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument("--debug", required=False, help="Whether in debug mode.", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    configure_iteration_logger(args)
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    working_directory_path = Path(args.working_directory_path)
    offline_learner = SingleIterationTripletsNSAMExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        learning_algorithm=learning_algorithm,
        fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
        polynom_degree=int(args.polynom_degree),
        problem_prefix=args.problems_prefix,
    )
    offline_learner.run_action_triplets_experiment(
        fold_num=args.fold_number,
        train_set_dir_path=(working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
        test_set_dir_path=(working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
    )


if __name__ == "__main__":
    main()

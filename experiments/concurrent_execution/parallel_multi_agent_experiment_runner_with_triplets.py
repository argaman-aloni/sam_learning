"""Runs experiments for the numeric model learning algorithms."""
import argparse
import os
from pathlib import Path
from typing import List

from experiments.concurrent_execution.parallel_basic_experiment_runner import (
    configure_iteration_logger,
    PLANNER_EXECUTION_TIMEOUT,
)
from experiments.concurrent_execution.parallel_multi_agent_experiment_runner import SingleIterationMultiAgentExperimentRunner
from statistics import LearningStatisticsManager
from statistics.utils import init_semantic_performance_calculator_for_ma_experiments
from utilities import LearningAlgorithmType

EXPERIMENT_TIMEOUT = os.environ.get("PLANNER_EXECUTION_TIMEOUT", PLANNER_EXECUTION_TIMEOUT)


class MultiAgentTripletsBasedExperimentRunner(SingleIterationMultiAgentExperimentRunner):
    """Class to conduct offline multi-agent action model learning experiments."""

    learning_statistics_manager: LearningStatisticsManager

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learning_algorithm: LearningAlgorithmType,
        problem_prefix: str = "pfile",
        executing_agents: List[str] = None,
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=learning_algorithm,
            problem_prefix=problem_prefix,
            executing_agents=executing_agents,
            running_triplets_experiment=True,
        )

    def run_action_triplets_experiment(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Runs the experiment while iterating on the action triplets instead of on the trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: This assumes that the folder contains all the train set.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Executing the experiments on the action triplets instead of the trajectories for the fold - {fold_num}!")
        self.semantic_performance_calc = init_semantic_performance_calculator_for_ma_experiments(
            working_directory_path=self.working_directory_path,
            domain_file_name=self.domain_file_name,
            learning_algorithm=self._learning_algorithm,
            executing_agents=self.executing_agents,
            test_set_dir_path=test_set_dir_path,
        )
        partial_domain = self.read_domain_file(train_set_dir_path)
        complete_train_set = self.collect_observations(train_set_dir_path, partial_domain)
        transitions_based_training_set = self.create_transitions_based_training_set(complete_train_set)
        execution_scheme = [index + 1 for index in range(10)] + [index for index in range(20, min(len(transitions_based_training_set), 1000), 10)]
        for index in execution_scheme:
            self._learn_model_offline([*transitions_based_training_set[0:index]], partial_domain, test_set_dir_path, fold_num)

        self.semantic_performance_calc.export_semantic_performance(fold_num)
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the multi agent action model learning with triplets instead of trajectories.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--learning_algorithm", required=True, type=int, choices=[1, 7, 25],
    )
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument("--debug", required=False, help="Whether in debug mode.", type=bool, default=False)
    parser.add_argument("--executing_agents", required=True, help="The names of the agents that are executing the actions")
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    configure_iteration_logger(args)
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    executing_agents = args.executing_agents.replace("[", "").replace("]", "").split(",")
    working_directory_path = Path(args.working_directory_path)
    offline_learner = MultiAgentTripletsBasedExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        learning_algorithm=learning_algorithm,
        executing_agents=executing_agents,
        problem_prefix=args.problems_prefix,
    )
    offline_learner.run_action_triplets_experiment(
        fold_num=args.fold_number,
        train_set_dir_path=(working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
        test_set_dir_path=(working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
    )


if __name__ == "__main__":
    main()

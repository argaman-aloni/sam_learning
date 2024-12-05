"""Runs experiments for the numeric model learning algorithms."""
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, Domain, MultiAgentObservation

from experiments.concurrent_execution.parallel_basic_experiment_runner import (
    ParallelExperimentRunner,
    configure_iteration_logger,
    PLANNER_EXECUTION_TIMEOUT,
)
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, MultiAgentSAM, MASAMPlus
from statistics.utils import init_semantic_performance_calculator_for_ma_experiments
from utilities import LearningAlgorithmType, NegativePreconditionPolicy, MappingElement, SolverType
from validators import DomainValidator

EXPERIMENT_TIMEOUT = os.environ.get("PLANNER_EXECUTION_TIMEOUT", PLANNER_EXECUTION_TIMEOUT)


class SingleIterationMultiAgentExperimentRunner(ParallelExperimentRunner):
    """Class to conduct offline multi-agent action model learning experiments."""

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
        )
        self.executing_agents = executing_agents
        self.ma_domain_path = None
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name, problem_prefix=problem_prefix,
        )

    def _filter_baseline_single_agent_trajectory(self, complete_observation: MultiAgentObservation) -> Observation:
        """Create a single agent observation from a multi-agent observation.

        :param complete_observation: the multi-agent observation to filter.
        :return: the filtered single agent observation.
        """
        filtered_observation = Observation()
        filtered_observation.add_problem_objects(complete_observation.grounded_objects)
        for component in complete_observation.components:
            if component.grounded_joint_action.action_count > 1:
                self.logger.debug(
                    f"Skipping the joint action - {component.grounded_joint_action} " f"since it contains multiple agents executing at once.!"
                )
                continue

            filtered_observation.add_component(component.previous_state, component.grounded_joint_action.operational_actions[0], component.next_state)

        return filtered_observation

    def _validate_learned_domain_with_macro_actions(
        self,
        allowed_observations: List[Observation],
        learned_model: LearnerDomain,
        test_set_dir_path: Path,
        fold_number: int,
        learning_time: float,
        policy: NegativePreconditionPolicy,
        mapping: Dict[str, MappingElement],
    ) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param fold_number: the number of the fold that is currently running.
        :param learning_time: the time it took to learn the domain (in seconds).
        :param policy: the policy to use for the negative preconditions.
        :param mapping: the mapping of the learner+, between macro name and binding.
        :return: the path for the learned domain.
        """
        domain_file_path = self._export_domain_and_backup(allowed_observations, fold_number, learned_model, policy, test_set_dir_path)
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        portfolio = [SolverType.fast_downward]
        self.domain_validator.validate_domain(
            tested_domain_file_path=domain_file_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=allowed_observations,
            timeout=EXPERIMENT_TIMEOUT,
            learning_time=learning_time,
            solvers_portfolio=portfolio,
            preconditions_removal_policy=policy,
            mapping=mapping,
        )

        return domain_file_path

    @staticmethod
    def _apply_ma_sam_plus_algorithm(
        allowed_observations: List[MultiAgentObservation], partial_domain: Domain, policy: NegativePreconditionPolicy
    ) -> Tuple[LearnerDomain, Dict[str, str], Dict[str, MappingElement]]:
        """Learns the action model using the extended version of the multi-agent algorithm Multi-Agent SAM+.

        :param allowed_observations: the allowed observations.
        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :return: the learned action model, the learned action model's learning statistics and the mapping used to create single agent plans from multi-agent plans.
        """
        learner = MASAMPlus(partial_domain=partial_domain, negative_precondition_policy=policy)
        return learner.learn_combined_action_model_with_macro_actions(allowed_observations)

    def _apply_multi_agent_learning_algorithms(
        self, partial_domain: Domain, allowed_observations: Union[List[Observation], List[MultiAgentObservation]], policy: NegativePreconditionPolicy
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :return: the learned action model and the learned action model's learning statistics.
        """
        if self._learning_algorithm == LearningAlgorithmType.sam_learning:
            learner = SAMLearner(partial_domain=partial_domain, negative_preconditions_policy=policy)
            return learner.learn_action_model(allowed_observations)

        elif self._learning_algorithm == LearningAlgorithmType.ma_sam:
            learner = MultiAgentSAM(partial_domain=partial_domain, negative_precondition_policy=policy)
            return learner.learn_combined_action_model(allowed_observations)

        raise ValueError(f"Unknown learning algorithm - {self._learning_algorithm}")

    def _learn_model_offline(
        self,
        allowed_observations: Union[List[Observation], List[MultiAgentObservation]],
        partial_domain: Domain,
        test_set_dir_path: Path,
        fold_num: int,
        iteration_number: int,
    ):
        """

        :param allowed_observations:
        :param partial_domain:
        :param test_set_dir_path:
        :param fold_num:
        :param iteration_number:
        :return:
        """
        if self._learning_algorithm in [LearningAlgorithmType.ma_sam, LearningAlgorithmType.sam_learning]:
            for policy in NegativePreconditionPolicy:
                learned_domain, learning_report = self._apply_multi_agent_learning_algorithms(partial_domain, allowed_observations, policy)
                self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_domain, learning_report, policy=policy)
                learned_domain_path = self.validate_learned_domain(
                    allowed_observations, learned_domain, test_set_dir_path, fold_num, float(learning_report["learning_time"]), policy
                )
                self.semantic_performance_calc.calculate_performance_for_ma_sam_experiments(learned_domain_path, len(allowed_observations), policy)

            self.domain_validator.write_statistics(fold_num, iteration_number)
            self.semantic_performance_calc.export_semantic_performance(fold_num, iteration_number)
            self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num, iteration_num=iteration_number)

        else:
            for policy in NegativePreconditionPolicy:
                learned_domain, learning_report, mapping = self._apply_ma_sam_plus_algorithm(
                    allowed_observations=allowed_observations, partial_domain=partial_domain, policy=policy
                )
                self._validate_learned_domain_with_macro_actions(
                    allowed_observations, learned_domain, test_set_dir_path, fold_num, float(learning_report["learning_time"]), policy, mapping
                )

            self.domain_validator.write_statistics(fold_num, iteration_number)

        self.logger.info(f"Finished the learning phase for the fold - {fold_num} and {len(allowed_observations)} observations!")

    def run_experiment(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path, iteration_number: int = 0) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        :param iteration_number: the number of the iteration that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        self.semantic_performance_calc = init_semantic_performance_calculator_for_ma_experiments(
            working_directory_path=self.working_directory_path,
            domain_file_name=self.domain_file_name,
            learning_algorithm=self._learning_algorithm,
            executing_agents=self.executing_agents,
            test_set_dir_path=test_set_dir_path,
        )
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_observations = []
        sorted_trajectory_paths = sorted(train_set_dir_path.glob("*.trajectory"))  # for consistency
        for index, trajectory_file_path in enumerate(sorted_trajectory_paths):
            # assuming that the folders were created so that each folder contains only the correct number of trajectories, i.e., iteration_number
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            complete_observation: MultiAgentObservation = TrajectoryParser(partial_domain, problem).parse_trajectory(
                trajectory_file_path, self.executing_agents
            )

            if self._learning_algorithm == LearningAlgorithmType.sam_learning:
                filtered_observation = self._filter_baseline_single_agent_trajectory(complete_observation)
                allowed_observations.append(filtered_observation)

            else:
                allowed_observations.append(complete_observation)

        # Execute the actual experiments
        self._learn_model_offline(allowed_observations, partial_domain, test_set_dir_path, fold_num, iteration_number)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the multi agent action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--learning_algorithm", required=True, type=int, choices=[1, 7, 25],
    )
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument("--iteration_number", required=True, help="The current iteration to execute", type=int)
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
    iteration_number = int(args.iteration_number)
    offline_learner = SingleIterationMultiAgentExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        learning_algorithm=learning_algorithm,
        executing_agents=executing_agents,
        problem_prefix=args.problems_prefix,
    )
    offline_learner.run_experiment(
        fold_num=args.fold_number,
        train_set_dir_path=(working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}_{iteration_number}",
        test_set_dir_path=(working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}_{iteration_number}",
        iteration_number=int(args.iteration_number),
    )


if __name__ == "__main__":
    main()

"""The POL main framework - Compile, Learn and Plan."""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, MultiAgentObservation, Domain, Problem

from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from experiments.numeric_performance_calculator import NumericPerformanceCalculator
from experiments.utils import init_numeric_performance_calculator
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner, PolynomialSAMLearning, MultiAgentSAM
from utilities import LearningAlgorithmType, SolverType
from validators import DomainValidator

DEFAULT_SPLIT = 5


class MAPlanningWithOfflineLearning:
    """Class that represents the POL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learning_statistics_manager: LearningStatisticsManager
    domain_validator: DomainValidator
    executing_agents: List[str]

    def __init__(self, working_directory_path: Path, domain_file_name: str, executing_agents: List[str] = None):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path,
                                 domain_file_name=domain_file_name,
                                 n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=LearningAlgorithmType.ma_sam)
        self.domain_validator = DomainValidator(
            self.working_directory_path, LearningAlgorithmType.ma_sam, self.working_directory_path / domain_file_name,
            solver_type=SolverType.fast_downward)
        self.executing_agents = executing_agents

    def _filter_baseline_multi_agent_trajectory(
            self, complete_observation: MultiAgentObservation) -> MultiAgentObservation:
        """

        :param complete_observation:
        :return:
        """
        filtered_observation = MultiAgentObservation(executing_agents=self.executing_agents)
        filtered_observation.add_problem_objects(complete_observation.grounded_objects)
        for component in complete_observation.components:
            if component.grounded_joint_action.action_count > 1:
                continue

            filtered_observation.add_component(component.previous_state,
                                               component.grounded_joint_action.actions,
                                               component.next_state)
        return filtered_observation

    def export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        """
        domain_path = test_set_path / self.domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def learn_ma_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_complete_observations = []
        allowed_filtered_observations = []
        observed_objects = {}
        for index, trajectory_file_path in enumerate(train_set_dir_path.glob("*.trajectory")):
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            observed_objects.update(problem.objects)
            complete_observation: MultiAgentObservation = TrajectoryParser(partial_domain, problem).parse_trajectory(
                trajectory_file_path, self.executing_agents)
            filtered_observation = self._filter_baseline_multi_agent_trajectory(complete_observation)
            allowed_complete_observations.append(complete_observation)
            allowed_filtered_observations.append(filtered_observation)
            if index % 5 != 0:
                self.logger.info(f"Skipping the iteration {index} to save the total amount of time!")
                continue

            self.logger.info(f"Learning the action model using {len(allowed_complete_observations)} trajectories!")
            self.learn_non_modified_trajectories(allowed_complete_observations, partial_domain, test_set_dir_path)
            self.learn_baseline_action_model(allowed_filtered_observations, partial_domain, test_set_dir_path)

        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)

    def learn_baseline_action_model(self, allowed_filtered_observations, partial_domain, test_set_dir_path):
        learner = MultiAgentSAM(partial_domain=partial_domain)
        self.domain_validator.learning_algorithm = LearningAlgorithmType.ma_sam.ma_sam_baseline
        self.learning_statistics_manager.learning_algorithm = LearningAlgorithmType.ma_sam_baseline
        learned_model, learning_report = learner.learn_combined_action_model(allowed_filtered_observations)
        self.learning_statistics_manager.add_to_action_stats(allowed_filtered_observations, learned_model,
                                                             learning_report)
        self.validate_learned_domain(allowed_filtered_observations, learned_model, test_set_dir_path)

    def learn_non_modified_trajectories(self, allowed_complete_observations, partial_domain, test_set_dir_path):
        learner = MultiAgentSAM(partial_domain=partial_domain)
        self.learning_statistics_manager.learning_algorithm = LearningAlgorithmType.ma_sam
        self.domain_validator.learning_algorithm = LearningAlgorithmType.ma_sam
        learned_model, learning_report = learner.learn_combined_action_model(allowed_complete_observations)
        self.learning_statistics_manager.add_to_action_stats(allowed_complete_observations, learned_model,
                                                             learning_report)
        self.validate_learned_domain(allowed_complete_observations, learned_model, test_set_dir_path)

    def validate_learned_domain(self, allowed_observations: List[MultiAgentObservation],
                                learned_model: LearnerDomain,
                                test_set_dir_path: Path) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              used_observations=allowed_observations)

        return domain_file_path

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_ma_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the POL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--executing_agents", required=False, default=None,
                        help="In case of a multi-agent action model learning, the names of the agents that "
                             "are executing the actions")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    executing_agents = args.executing_agents.replace("[", "").replace("]", "").split(",") \
        if args.executing_agents is not None else None

    offline_learner = MAPlanningWithOfflineLearning(working_directory_path=Path(args.working_directory_path),
                                                    domain_file_name=args.domain_file_name,
                                                    executing_agents=executing_agents)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, ActionCall, State, MultiAgentObservation, \
    JointActionCall, MultiAgentComponent

from experiments.performance_calculation_utils import _calculate_single_action_applicability_rate
from utilities import LearningAlgorithmType

SEMANTIC_PRECISION_STATS = ["action_name", "num_trajectories", "precondition_precision", "precondition_recall", ]


class SemanticPerformanceCalculator:
    """Class responsible for calculating the semantic precision and recall of a model."""

    model_domain: Domain
    dataset_observations: List[Observation]
    learning_algorithm: LearningAlgorithmType
    combined_stats: List[Dict[str, Any]]
    logger: logging.Logger
    results_dir_path: Path

    def __init__(self, model_domain: Domain, observations: List[Union[Observation, MultiAgentObservation]],
                 working_directory_path: Path, learning_algorithm: LearningAlgorithmType):
        self.logger = logging.getLogger(__name__)
        self.model_domain = model_domain
        self.dataset_observations = observations
        self.learning_algorithm = learning_algorithm
        self.combined_stats = []
        self.results_dir_path = working_directory_path / "results_directory"

    def _calculate_action_applicability_rate(
            self, action_call: Union[ActionCall, JointActionCall], learned_domain: Domain,
            num_false_negatives: Dict[str, int],
            num_false_positives: Dict[str, int], num_true_positives: Dict[str, int], observed_state: State) -> None:
        """Test whether an action is applicable in both the model domain and the generated domain.

        :param action_call: the action call that is tested for applicability.
        :param learned_domain: the domain that was learned using the action model learning algorithm.
        :param num_false_negatives: the dictionary mapping between the action name and the number of false negative
            executions.
        :param num_false_positives: the dictionary mapping between the action name and the number of false positive
            executions.
        :param num_true_positives: the dictionary mapping between the action name and the number of true positive
            executions.
        :param observed_state: the state that is currently being tested.
        """
        if isinstance(action_call, JointActionCall):
            for action in action_call.actions:
                _calculate_single_action_applicability_rate(
                    action, learned_domain, self.model_domain, num_false_negatives, num_false_positives,
                    num_true_positives, observed_state)
        else:
            _calculate_single_action_applicability_rate(
                action_call, learned_domain, self.model_domain, num_false_negatives, num_false_positives,
                num_true_positives, observed_state)

    @staticmethod
    def _extract_states_and_actions(observation: Union[Observation, MultiAgentObservation]) -> Tuple[
        List[State], List[Union[ActionCall, JointActionCall]]]:
        """Extract the states and the actions of the trajectory to use for the performance calculations.

        :param observation: the observation data of the input trajectory.
        :return: the observed states in a list and the list of the plan actions.
        """
        observed_states: List[State] = []
        executed_actions = []
        for observation_component in observation.components:
            if isinstance(observation_component, MultiAgentComponent):
                executed_actions.append(observation_component.grounded_joint_action)
            else:
                executed_actions.append(observation_component.grounded_action_call)

            if observation_component.previous_state.is_init:
                observed_states.append(observation_component.previous_state)

            observed_states.append(observation_component.next_state)

        return observed_states, executed_actions

    def calculate_preconditions_semantic_performance(
            self, learned_domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned preconditions.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :return: the precision and recall dictionaries.
        """
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        num_false_positives = defaultdict(int)
        for observation in self.dataset_observations:
            observed_states, executed_actions = self._extract_states_and_actions(observation)
            for action_call in executed_actions:
                actions_to_check = action_call.actions if isinstance(action_call, JointActionCall) else [action_call]
                for action in actions_to_check:
                    for observed_state in observed_states:
                        if action.name not in learned_domain.actions:
                            break

                        self._calculate_action_applicability_rate(
                            action, learned_domain, num_false_negatives, num_false_positives, num_true_positives,
                            observed_state)

        precision_dict = {}
        recall_dict = {}
        for action_name, tp_rate in num_true_positives.items():
            precision_dict[action_name] = tp_rate / (tp_rate + num_false_positives[action_name])
            recall_dict[action_name] = tp_rate / (tp_rate + num_false_negatives[action_name])

        return precision_dict, recall_dict

    def calculate_semantic_performance(self, learned_domain_path: Path, num_used_observations: int):
        """Calculate the semantic precision and recall of the learned domain.

        :param learned_domain_path:
        :param num_used_observations:
        :return:
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        precision, recall = self.calculate_preconditions_semantic_performance(learned_domain)
        for action_name in learned_domain.actions:
            action_stats = {
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "precondition_precision": precision[action_name],
                "precondition_recall": recall[action_name]
            }
            self.combined_stats.append(action_stats)

    def export_semantic_performance(self, fold_num: int) -> None:
        """Exports the precision values of the learned preconditions to a CSV file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  f"{fold_num}_preconditions_semantic_performance.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=SEMANTIC_PRECISION_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

    def export_combined_semantic_performance(self) -> None:
        """Export the numeric learning statistics to a CSV report file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  "combined_preconditions_semantic.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=SEMANTIC_PRECISION_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, NoReturn, Tuple, Any

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, Operator, ActionCall, State

from utilities import LearningAlgorithmType

NUMERIC_PERFORMANCE_STATS = ["action_name", "num_trajectories", "ratio_actions_learned",
                             "precondition_precision", "precondition_recall", "effects_mse"]


class NumericPerformanceCalculator:
    """Class responsible for calculating the numeric precision and recall of a model."""

    model_domain: Domain
    dataset_observations: List[Observation]
    learning_algorithm: LearningAlgorithmType
    combined_stats: List[Dict[str, Any]]
    logger: logging.Logger
    results_dir_path: Path

    def __init__(self, model_domain: Domain, observations: List[Observation],
                 working_directory_path: Path, learning_algorithm: LearningAlgorithmType):
        self.logger = logging.getLogger(__name__)
        self.model_domain = model_domain
        self.dataset_observations = observations
        self.learning_algorithm = learning_algorithm
        self.combined_stats = []
        self.results_dir_path = working_directory_path / "results_directory"

    @staticmethod
    def _ground_tested_operator(action_call: ActionCall, learned_domain: Domain) -> Operator:
        """Ground the tested action based on the trajectory data.

        :param action_call: the grounded action call in the observation component.
        :param learned_domain: the domain that was learned using the action model learning algorithm.
        :return: the grounded operator.
        """
        grounded_operator = Operator(
            action=learned_domain.actions[action_call.name],
            domain=learned_domain,
            grounded_action_call=action_call.parameters)
        grounded_operator.ground()
        return grounded_operator

    def _calculate_action_applicability_rate(
            self, action_call: ActionCall, learned_domain: Domain, num_false_negatives: Dict[str, int],
            num_false_positives: Dict[str, int], num_true_positives: Dict[str, int], observed_state: State) -> NoReturn:
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
        tested_grounded_operator = self._ground_tested_operator(action_call, learned_domain)
        model_grounded_operator = self._ground_tested_operator(action_call, self.model_domain)
        is_applicable_in_test = tested_grounded_operator.is_applicable(observed_state)
        is_applicable_in_model = model_grounded_operator.is_applicable(observed_state)
        num_true_positives[action_call.name] += int(is_applicable_in_test == is_applicable_in_model)
        num_false_positives[action_call.name] += int(is_applicable_in_test and not is_applicable_in_model)
        num_false_negatives[action_call.name] += int(not is_applicable_in_test and is_applicable_in_model)

    @staticmethod
    def _extract_states_and_actions(observation: Observation) -> Tuple[List[State], List[ActionCall]]:
        """Extract the states and the actions of the trajectory to use for the performance calculations.

        :param observation: the observation data of the input trajectory.
        :return: the observed states in a list and the list of the plan actions.
        """
        observed_states: List[State] = []
        executed_actions: List[ActionCall] = []
        for observation_component in observation.components:
            action_call = observation_component.grounded_action_call
            executed_actions.append(action_call)
            if observation_component.previous_state.is_init:
                observed_states.append(observation_component.previous_state)

            observed_states.append(observation_component.next_state)

        return observed_states, executed_actions

    def calculate_precondition_performance(self, learned_domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
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
                for observed_state in observed_states:
                    if action_call.name not in learned_domain.actions:
                        break

                    self._calculate_action_applicability_rate(
                        action_call, learned_domain, num_false_negatives, num_false_positives, num_true_positives,
                        observed_state)

        precision_dict = {}
        recall_dict = {}
        for action_name, tp_rate in num_true_positives.items():
            precision_dict[action_name] = tp_rate / (tp_rate + num_false_positives[action_name])
            recall_dict[action_name] = tp_rate / (tp_rate + num_false_negatives[action_name])

        return precision_dict, recall_dict

    def calculate_effects_performance(self, learned_domain: Domain) -> Dict[str, float]:
        """Calculates the effects MSE value using the actual state fluents and the ones generated using the learned
            action.

        Note:
            MSE is calculated as follows - 1/n * Sum((x-x')^2)

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :return: a mapping between the action name and its MSE value.
        """
        squared_errors = defaultdict(list)
        for observation in self.dataset_observations:
            for observation_component in observation.components:
                action_call = observation_component.grounded_action_call
                if action_call.name not in learned_domain.actions:
                    continue

                grounded_operator = self._ground_tested_operator(action_call, learned_domain)
                previous_state = observation_component.previous_state
                learned_next_state_fluents = grounded_operator.update_state_functions(previous_state)
                actual_next_state = observation_component.next_state
                for fluent_name, fluent_data in actual_next_state.state_fluents.items():
                    learned_value = learned_next_state_fluents[fluent_name].value
                    squared_errors[action_call.name].append(math.pow(fluent_data.value - learned_value, 2))

        return {
            action_name: sum(square_errors) / len(square_errors)
            for action_name, square_errors in squared_errors.items()
        }

    def calculate_performance(self, learned_domain_path: Path, num_used_observations: int) -> NoReturn:
        """Calculates the model's performance with both the precision and the recall values calculated.

        :param learned_domain_path: the path to the learned action model.
        :param num_used_observations: the number of observations used to learn the action model.
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        precision, recall = self.calculate_precondition_performance(learned_domain)
        effects_mse = self.calculate_effects_performance(learned_domain)
        for action_name in learned_domain.actions:
            action_stats = {
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "ratio_actions_learned": len(learned_domain.actions) / len(self.model_domain.actions),
                "precondition_precision": precision[action_name],
                "precondition_recall": recall[action_name],
                "effects_mse": effects_mse[action_name]}
            self.combined_stats.append(action_stats)

    def export_numeric_learning_performance(self) -> NoReturn:
        """Export the numeric learning statistics to a CSV report file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  f"_numeric_learning_performance_stats.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=NUMERIC_PERFORMANCE_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

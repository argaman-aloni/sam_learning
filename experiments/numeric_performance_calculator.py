"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, NoReturn, Tuple, Any

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, Operator, ActionCall

from experiments.util_types import LearningAlgorithmType

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

    def calculate_precondition_performance(self, learned_domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculates the precision recall values of the learned preconditions.

        :param learned_domain: the action model that was learned using the action model learning algorithm
        :return: the precision and recall dictionaries.
        """
        num_true_positives = defaultdict(int)
        num_false_negatives = defaultdict(int)
        for observation in self.dataset_observations:
            for observation_component in observation.components:
                action_call = observation_component.grounded_action_call
                if action_call.name not in learned_domain.actions:
                    continue

                grounded_operator = self._ground_tested_operator(action_call, learned_domain)
                is_applicable = grounded_operator.is_applicable(observation_component.previous_state)
                num_true_positives[action_call.name] += int(is_applicable == True)
                num_false_negatives[action_call.name] += int(is_applicable == False)

        precision_dict = {}
        recall_dict = {}
        for action_name, tp_rate in num_true_positives.items():
            precision_dict[action_name] = tp_rate / (tp_rate + 0)
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

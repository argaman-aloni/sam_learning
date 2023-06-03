"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

import math
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, MultiAgentObservation

from experiments.performance_calculation_utils import _ground_tested_operator
from experiments.semantic_performance_calculator import SemanticPerformanceCalculator
from utilities import LearningAlgorithmType

NUMERIC_PERFORMANCE_STATS = ["action_name", "num_trajectories", "precondition_precision", "precondition_recall",
                             "effects_precision", "effects_recall", "effects_mse"]


class NumericPerformanceCalculator(SemanticPerformanceCalculator):
    """Class responsible for calculating the numeric precision and recall of a model."""

    def __init__(self, model_domain: Domain, observations: List[Union[Observation, MultiAgentObservation]],
                 working_directory_path: Path, learning_algorithm: LearningAlgorithmType):
        super().__init__(model_domain, observations, working_directory_path, learning_algorithm)

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
                previous_state = observation_component.previous_state
                if action_call.name not in learned_domain.actions:
                    continue

                grounded_operator = _ground_tested_operator(action_call, learned_domain, observation.grounded_objects)
                try:
                    next_state = grounded_operator.apply(previous_state)
                    learned_next_state_fluents = next_state.state_fluents
                    actual_next_state = observation_component.next_state
                    for fluent_name, fluent_data in actual_next_state.state_fluents.items():
                        learned_value = learned_next_state_fluents[fluent_name].value
                        squared_errors[action_call.name].append(math.pow(fluent_data.value - learned_value, 2))

                except ValueError:
                    self.logger.debug(f"Could not apply action {action_call.name} on the state.")
                    continue

        return {
            action_name: sum(square_errors) / len(square_errors)
            for action_name, square_errors in squared_errors.items()
        }

    def calculate_performance(self, learned_domain_path: Path, num_used_observations: int) -> None:
        """Calculates the model's performance with both the precision and the recall values calculated.

        :param learned_domain_path: the path to the learned action model.
        :param num_used_observations: the number of observations used to learn the action model.
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        self.logger.info("Starting to calculate the semantic preconditions performance of the learned domain.")
        preconditions_precision, preconditions_recall = self.calculate_preconditions_semantic_performance(
            learned_domain)
        self.logger.info("Starting to calculate the semantic effects performance of the learned domain.")
        effects_precision, effects_recall = self.calculate_effects_semantic_performance(learned_domain)
        effects_mse = self.calculate_effects_performance(learned_domain)
        for action_name in self.model_domain.actions:
            action_stats = {
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "precondition_precision": preconditions_precision.get(action_name, 0),
                "precondition_recall": preconditions_recall.get(action_name, 0),
                "effects_precision": effects_precision.get(action_name, 0),
                "effects_recall": effects_recall.get(action_name, 0),
                "effects_mse": effects_mse.get(action_name, 0)
            }
            self.combined_stats.append(action_stats)

    def export_semantic_performance(self, fold_num: int) -> None:
        """Export the numeric learning statistics to a CSV report file.

        :param fold_num: the fold number of the current experiment.
        """
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  f"_numeric_learning_performance_stats_fold_{fold_num}.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=NUMERIC_PERFORMANCE_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

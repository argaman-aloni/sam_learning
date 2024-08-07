"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

import math
import sklearn
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, MultiAgentObservation

from statistics.performance_calculation_utils import _ground_executed_action
from statistics.semantic_performance_calculator import SemanticPerformanceCalculator
from utilities import LearningAlgorithmType

NUMERIC_PERFORMANCE_STATS = [
    "learning_algorithm",
    "action_name",
    "num_trajectories",
    "precondition_precision",
    "precondition_recall",
    "effects_precision",
    "effects_recall",
    "effects_mse",
]


class NumericPerformanceCalculator(SemanticPerformanceCalculator):
    """Class responsible for calculating the numeric precision and recall of a model."""

    def __init__(
        self,
        model_domain: Domain,
        model_domain_path: Path,
        observations: List[Union[Observation, MultiAgentObservation]],
        working_directory_path: Path,
        learning_algorithm: LearningAlgorithmType,
    ):
        super().__init__(model_domain, model_domain_path, observations, working_directory_path, learning_algorithm)

    def calculate_effects_mse(self, learned_domain: Domain) -> Dict[str, float]:
        """Calculates the effects MSE value using the actual state fluents and the ones generated using the learned
            action.

        Note:
            The final value is averaged over the number of states per action.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :return: a mapping between the action name and its MSE value.
        """
        squared_errors = defaultdict(list)
        mse_values = {action_name: 1 for action_name in learned_domain.actions.keys()}
        for observation in self.dataset_observations:
            for observation_component in observation.components:
                action_call = observation_component.grounded_action_call
                previous_state = observation_component.previous_state
                model_next_state = observation_component.next_state
                if action_call.name not in learned_domain.actions:
                    continue

                grounded_operator = _ground_executed_action(action_call, learned_domain, observation.grounded_objects)
                try:
                    next_state = grounded_operator.apply(previous_state, allow_inapplicable_actions=False)

                except ValueError:
                    self.logger.debug("The action is not applicable in the state.")
                    next_state = previous_state.copy()
                    # since the learned action is not applicable in the state there is no point to compare
                    # with an action that is not applicable in the model domain.
                    model_next_state = previous_state.copy()

                values = [
                    (next_state.state_fluents[fluent].value, model_next_state.state_fluents[fluent].value)
                    for fluent in next_state.state_fluents.keys()
                ]
                actual_values, expected_values = zip(*values)
                state_mse = sklearn.metrics.mean_squared_error(expected_values, actual_values)
                squared_errors[action_call.name].append(state_mse)

        mse_values.update({action_name: sum(square_errors) / len(square_errors) for action_name, square_errors in squared_errors.items()})
        return mse_values

    def calculate_performance(self, learned_domain_path: Path, num_used_observations: int) -> None:
        """Calculates the model's performance with both the precision and the recall values calculated.

        :param learned_domain_path: the path to the learned action model.
        :param num_used_observations: the number of observations used to learn the action model.
        """
        learned_domain = DomainParser(domain_path=learned_domain_path, partial_parsing=False).parse_domain()
        self.logger.info("Starting to calculate the semantic preconditions performance of the learned domain.")
        preconditions_precision, preconditions_recall = self.calculate_preconditions_semantic_performance(learned_domain, learned_domain_path)
        self.logger.info("Starting to calculate the semantic effects performance of the learned domain.")
        effects_precision, effects_recall = self.calculate_effects_semantic_performance(learned_domain)
        effects_mse = self.calculate_effects_mse(learned_domain)
        for action_name in self.model_domain.actions:
            action_stats = {
                "learning_algorithm": self.learning_algorithm.name,
                "action_name": action_name,
                "num_trajectories": num_used_observations,
                "precondition_precision": preconditions_precision.get(action_name, 1),
                "precondition_recall": preconditions_recall.get(action_name, 0),
                "effects_precision": effects_precision.get(action_name, 1),
                "effects_recall": effects_recall.get(action_name, 1),
                "effects_mse": effects_mse.get(action_name, 0),
            }
            self.combined_stats.append(action_stats)

        self.logger.info(f"Finished calculating the numeric learning performance for {self.learning_algorithm.name}.")

    def export_semantic_performance(self, fold_num: int, iteration_num: int = 0) -> None:
        """Export the numeric learning statistics to a CSV report file.

        :param fold_num: the fold number of the current experiment.
        :param iteration_num: the iteration number of the current experiment.
        """
        statistics_path = (
            self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}"
            f"_numeric_learning_performance_stats_fold_{fold_num}_{iteration_num}.csv"
        )
        with open(statistics_path, "wt", newline="") as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=NUMERIC_PERFORMANCE_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

    def export_combined_semantic_performance(self) -> None:
        """Export the numeric learning statistics to a CSV report file."""
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" "combined_numeric_performance.csv"
        with open(statistics_path, "wt", newline="") as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=NUMERIC_PERFORMANCE_STATS)
            stats_writer.writeheader()
            for data_line in self.combined_stats:
                stats_writer.writerow(data_line)

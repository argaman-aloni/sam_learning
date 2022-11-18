"""Module responsible for calculating our approach for numeric precision and recall."""
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, NoReturn, Union

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, MultiAgentObservation

from experiments.performance_calculation_utils import _ground_tested_operator
from experiments.semantic_performance_calculator import SemanticPerformanceCalculator
from utilities import LearningAlgorithmType

NUMERIC_PERFORMANCE_STATS = ["action_name", "num_trajectories", "ratio_actions_learned",
                             "precondition_precision", "precondition_recall", "effects_mse"]


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
                if action_call.name not in learned_domain.actions:
                    continue

                grounded_operator = _ground_tested_operator(action_call, learned_domain)
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
        precision, recall = super().calculate_preconditions_semantic_performance(learned_domain)
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

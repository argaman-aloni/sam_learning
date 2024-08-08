"""Module to manage the action model learning statistics."""
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation, MultiAgentObservation, MultiAgentComponent, Predicate

from statistics.discrete_precision_recall_calculator import PrecisionRecallCalculator
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType

LEARNED_ACTIONS_STATS_COLUMNS = [
    "learning_algorithm",
    "domain_name",
    "num_trajectories",
    "learning_time",
    "num_trajectory_triplets",
    "total_number_of_actions",
    "num_safe_actions",
    "num_unsafe_actions",
    "learned_action_name",
    "num_triplets_action_appeared",
    "learned_discrete_preconditions",
    "num_discrete_preconditions",
    "learned_discrete_effects",
    "num_discrete_effects",
    "ground_truth_preconditions",
    "ground_truth_effects",
    "preconditions_precision",
    "effects_precision",
    "preconditions_recall",
    "effects_recall",
    "action_precision",
    "action_recall",
    "f1_score"
]

NUMERIC_LEARNING_STAT_COLUMNS = [
    "learning_algorithm",
    "domain_name",
    "num_trajectories",
    "num_trajectory_triplets",
    "total_number_of_actions",
    "#numeric_actions_no_solution",
    "#numeric_actions_no_convex_hull",
    "#numeric_actions_infinite_number_solutions",
    "model_precision",
    "model_recall",
    "model_f1_score"
]

SOLVING_STATISTICS = [
    "learning_algorithm",
    "domain_name",
    "num_trajectories",
    "num_trajectory_triplets",
    "#problems_solved",
    "model_precision",
    "model_recall",
    "model_f1_score"
]


class LearningStatisticsManager:
    """Class that manages the statistics about the action's learning properties gathered from the learning process."""
    model_domain: Domain
    learning_algorithm: LearningAlgorithmType
    working_directory_path: Path
    results_dir_path: Path
    action_learning_stats: List[Dict[str, Any]]
    numeric_learning_stats: List[Dict[str, Any]]
    merged_numeric_stats: List[Dict[str, Any]]
    merged_action_stats: List[Dict[str, Any]]

    def __init__(self, working_directory_path: Path, domain_path: Path, learning_algorithm: LearningAlgorithmType):
        self.working_directory_path = working_directory_path
        self.model_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
        self.learning_algorithm = learning_algorithm
        self.action_learning_stats = []
        self.numeric_learning_stats = []
        self.merged_action_stats = []
        self.merged_numeric_stats = []
        self.results_dir_path = self.working_directory_path / "results_directory"

    @staticmethod
    def _update_action_appearances(used_observations: List[Union[Observation, MultiAgentObservation]]) -> Counter:
        """Updates the number of times each action appeared in the input trajectories.

        :param used_observations: the observations used to learn the actions.
        :return: Counter object mapping the action name to the number of appearances.
        """
        action_appearance_counter = Counter()
        for observation in used_observations:
            for component in observation.components:
                if isinstance(component, MultiAgentComponent):
                    for action in component.grounded_joint_action.actions:
                        action_appearance_counter[action.name] += 1
                else:
                    action_appearance_counter[component.grounded_action_call.name] += 1
        return action_appearance_counter

    def _extract_all_preconditions(self, action_name, learned_domain):
        learned_preconditions = [p.untyped_representation for _, p in
                                 learned_domain.actions[action_name].preconditions if isinstance(p, Predicate)]
        ground_truth_preconditions = [p.untyped_representation for _, p in
                                      self.model_domain.actions[action_name].preconditions if isinstance(p, Predicate)]
        return ground_truth_preconditions, learned_preconditions

    def create_results_directory(self) -> None:
        """Creates the results' directory that contains the learning results."""
        self.results_dir_path.mkdir(exist_ok=True)

    def add_to_action_stats(
            self, used_observations: List[Union[Observation, MultiAgentObservation]],
            learned_domain: LearnerDomain, learning_report: Optional[Dict[str, str]] = None) -> None:
        """Add the action data to the statistics.

        :param used_observations: the observations that were used to learn the action.
        :param learned_domain: the domain that was learned from the action model learning algorithm.
        :param learning_report: the report on the status of the learned actions, whether they were safe to learn or not.
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        action_appearance_counter = self._update_action_appearances(used_observations)

        precision_recall_calculator = PrecisionRecallCalculator()
        for action_name, action_data in learned_domain.actions.items():
            precision_recall_calculator.add_action_data(action_data, self.model_domain.actions[action_name])
            ground_truth_preconditions, learned_preconditions = self._extract_all_preconditions(action_name,
                                                                                                learned_domain)

            learned_discrete_effects = [
                p.untyped_representation for p in learned_domain.actions[action_name].discrete_effects]
            num_safe_actions = len([learning_report[action_name] for action_name in learning_report if
                                    learning_report[action_name] == "OK"])
            num_unsafe_actions = len([learning_report[action_name] for action_name in learning_report
                                      if learning_report[action_name] == "NOT SAFE"])
            learning_time = learning_report["learning_time"]
            action_stats = {
                "learning_algorithm": self.learning_algorithm.name,
                "domain_name": self.model_domain.name,
                "num_trajectories": len(used_observations),
                "learning_time": learning_time,
                "num_trajectory_triplets": num_triplets,
                "total_number_of_actions": len(self.model_domain.actions),
                "num_safe_actions": num_safe_actions,
                "num_unsafe_actions": num_unsafe_actions,
                "learned_action_name": action_name,
                "num_triplets_action_appeared": action_appearance_counter[action_name],
                "learned_discrete_preconditions": learned_preconditions,
                "num_discrete_preconditions": len(learned_preconditions),
                "learned_discrete_effects": learned_discrete_effects,
                "num_discrete_effects": len(learned_discrete_effects),
                "ground_truth_preconditions": ground_truth_preconditions,
                "ground_truth_effects": [p.untyped_representation for p in
                                         self.model_domain.actions[action_name].discrete_effects],
                **precision_recall_calculator.export_action_syntactic_statistics(action_name)
            }
            self.action_learning_stats.append(action_stats)

        if self.learning_algorithm == LearningAlgorithmType.numeric_sam:
            self._collect_numeric_learning_statistics(
                used_observations, learning_report, precision_recall_calculator)

    def _export_statistics_data(self, fold_number: int, columns: List[str],
                                action_data_to_export: List[Dict[str, Any]], stats_data_file_name: str) -> None:
        """Exports statistics to a report CSV file.

        :param fold_number: the number of the currently running fold.
        :param columns: the column names in the CSV report file.
        :param action_data_to_export: the data to export to the report.
        :param stats_data_file_name: the name of the statistics to publish in the report file.
        """
        statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}_{self.model_domain.name}" \
                                                  f"_{stats_data_file_name}_{fold_number}.csv"
        with open(statistics_path, "wt", newline='') as statistics_file:
            stats_writer = csv.DictWriter(statistics_file, fieldnames=columns)
            stats_writer.writeheader()
            for data_line in action_data_to_export:
                stats_writer.writerow(data_line)

    def export_action_learning_statistics(self, fold_number: int) -> None:
        """Export the statistics collected about the actions.

        :param fold_number: the number of the currently running fold.
        """
        self._export_statistics_data(fold_number, LEARNED_ACTIONS_STATS_COLUMNS, self.action_learning_stats,
                                     "action_stats_fold")

    def export_all_folds_action_stats(self) -> None:
        """Export the statistics collected about the actions."""
        output_path = self.results_dir_path / f"all_folds_action_learning_stats.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LEARNED_ACTIONS_STATS_COLUMNS)
            writer.writeheader()
            writer.writerows(self.merged_action_stats)

    def export_numeric_learning_statistics(self, fold_number: int) -> None:
        """Export the numeric learning statistics that were collected from the learning report.

        :param fold_number: the number of the currently running fold.
        """
        self._export_statistics_data(fold_number, NUMERIC_LEARNING_STAT_COLUMNS, self.numeric_learning_stats,
                                     "numeric_learning_fold")

    def clear_statistics(self) -> None:
        """Clears the statistics so that each fold will have no relation to its predecessors."""
        self.merged_numeric_stats.extend(self.numeric_learning_stats)
        self.merged_action_stats.extend(self.action_learning_stats)
        self.numeric_learning_stats.clear()
        self.action_learning_stats.clear()

    def _collect_numeric_learning_statistics(
            self, used_observations: List[Union[Observation, MultiAgentObservation]], learning_report: Dict[str, str],
            precision_recall_calc: PrecisionRecallCalculator) -> None:
        """Collects the numeric learning statistics from the learning report.

        :param used_observations: the observations that were used to learn the action model.
        :param learning_report: the report that the learning algorithm submitted.
        :param precision_recall_calc: the object that calculates the precision and recall of the learned action model.
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        actions_stats_counter = Counter(learning_report.values())
        model_precision = precision_recall_calc.calculate_syntactic_precision()
        model_recall = precision_recall_calc.calculate_syntactic_recall()
        model_f1_score = 0 if model_precision + model_recall == 0 else \
            2 * (model_precision * model_recall) / (model_precision + model_recall)

        model_stats = {
            "learning_algorithm": self.learning_algorithm.name,
            "domain_name": self.model_domain.name,
            "num_trajectories": len(used_observations),
            "num_trajectory_triplets": num_triplets,
            "total_number_of_actions": len(self.model_domain.actions),
            "#numeric_actions_no_solution": actions_stats_counter["no_solution_found"],
            "#numeric_actions_no_convex_hull": actions_stats_counter["convex_hull_not_found"],
            "#numeric_actions_infinite_number_solutions": actions_stats_counter["not_enough_data"],
            "model_precision": model_precision,
            "model_recall": model_recall,
            "model_f1_score": model_f1_score
        }
        self.numeric_learning_stats.append(model_stats)

    def write_complete_joint_statistics(self) -> None:
        """Writes a statistics file containing all the folds combined data."""
        output_path = self.results_dir_path / f"{self.learning_algorithm.name}_all_folds_numeric_learning_stats.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=NUMERIC_LEARNING_STAT_COLUMNS)
            writer.writeheader()
            writer.writerows(self.merged_numeric_stats)

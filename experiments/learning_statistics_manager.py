"""Module to manage the action model learning statistics."""
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, NoReturn, List, Optional

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from experiments.action_precision_recall_calculator import PrecisionRecallCalculator
from experiments.util_types import LearningAlgorithmType
from sam_learning.core import LearnerDomain

LEARNED_ACTIONS_STATS_COLUMNS = [
    "learning_algorithm",
    "domain_name",
    "num_trajectories",
    "num_trajectory_triplets",
    "total_number_of_actions",
    "learned_action_name",
    "num_triplets_action_appeared",
    "learned_discrete_preconditions",
    "learned_discrete_add_effects",
    "learned_discrete_delete_effects",
    "ground_truth_preconditions",
    "ground_truth_add_effects",
    "ground_truth_delete_effects",
    "preconditions_precision",
    "add_effects_precision",
    "delete_effects_precision",
    "preconditions_recall",
    "add_effects_recall",
    "delete_effects_recall",
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
    "#numeric_actions_learned_ok",
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

    def __init__(self, working_directory_path: Path, domain_path: Path, learning_algorithm: LearningAlgorithmType):
        self.working_directory_path = working_directory_path
        self.model_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
        self.learning_algorithm = learning_algorithm
        self.action_learning_stats = []
        self.numeric_learning_stats = []
        self.merged_numeric_stats = []
        self.results_dir_path = self.working_directory_path / "results_directory"

    @staticmethod
    def _update_action_appearances(used_observations: List[Observation]) -> Counter:
        """Updates the number of times each action appeared in the input trajectories.

        :param used_observations: the observations used to learn the actions.
        :return: Counter object mapping the action name to the number of appearances.
        """
        action_appearance_counter = Counter()
        for observation in used_observations:
            for component in observation.components:
                action_appearance_counter[component.grounded_action_call.name] += 1
        return action_appearance_counter

    def create_results_directory(self) -> NoReturn:
        """Creates the results' directory that contains the learning results."""
        self.results_dir_path.mkdir(exist_ok=True)

    def add_to_action_stats(self, used_observations: List[Observation], learned_domain: LearnerDomain,
                            learning_report: Optional[Dict[str, str]] = None) -> NoReturn:
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
            action_stats = {
                "learning_algorithm": self.learning_algorithm.name,
                "domain_name": self.model_domain.name,
                "num_trajectories": len(used_observations),
                "num_trajectory_triplets": num_triplets,
                "total_number_of_actions": len(self.model_domain.actions),
                "learned_action_name": action_name,
                "num_triplets_action_appeared": action_appearance_counter[action_name],
                "learned_discrete_preconditions": [p.untyped_representation for p in
                                                   learned_domain.actions[action_name].positive_preconditions],
                "learned_discrete_add_effects": [p.untyped_representation for p in
                                                 learned_domain.actions[action_name].add_effects],
                "learned_discrete_delete_effects": [p.untyped_representation for p in
                                                    learned_domain.actions[action_name].delete_effects],
                "ground_truth_preconditions": [p.untyped_representation for p in
                                               self.model_domain.actions[action_name].positive_preconditions],
                "ground_truth_add_effects": [p.untyped_representation for p in
                                             self.model_domain.actions[action_name].add_effects],
                "ground_truth_delete_effects": [p.untyped_representation for p in
                                                self.model_domain.actions[action_name].delete_effects],
                **precision_recall_calculator.export_action_statistics(action_name)
            }
            self.action_learning_stats.append(action_stats)

        if self.learning_algorithm == LearningAlgorithmType.numeric_sam:
            self._collect_numeric_learning_statistics(
                used_observations, learning_report, precision_recall_calculator)

    def _export_statistics_data(self, fold_number: int, columns: List[str],
                                action_data_to_export: List[Dict[str, Any]], stats_data_file_name: str) -> NoReturn:
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

    def export_action_learning_statistics(self, fold_number: int) -> NoReturn:
        """Export the statistics collected about the actions.

        :param fold_number: the number of the currently running fold.
        """
        self._export_statistics_data(fold_number, LEARNED_ACTIONS_STATS_COLUMNS, self.action_learning_stats,
                                     "action_stats_fold")

    def export_numeric_learning_statistics(self, fold_number: int) -> NoReturn:
        """Export the numeric learning statistics that were collected from the learning report.

        :param fold_number: the number of the currently running fold.
        """
        self._export_statistics_data(fold_number, NUMERIC_LEARNING_STAT_COLUMNS, self.numeric_learning_stats,
                                     "numeric_learning_fold")

    def clear_statistics(self) -> NoReturn:
        """Clears the statistics so that each fold will have no relation to its predecessors."""
        self.merged_numeric_stats.extend(self.numeric_learning_stats)
        self.numeric_learning_stats.clear()
        self.action_learning_stats.clear()

    def _collect_numeric_learning_statistics(
            self, used_observations: List[Observation], learning_report: Dict[str, str],
            precision_recall_calc: PrecisionRecallCalculator) -> NoReturn:
        """

        :param used_observations:
        :param learning_report:
        :param precision_recall_calc:
        :return:
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        actions_stats_counter = Counter(learning_report.values())
        model_precision = precision_recall_calc.calculate_model_precision()
        model_recall = precision_recall_calc.calculate_model_recall()
        model_f1_score = 0 if model_precision + model_recall == 0 else \
            2 * (model_precision * model_recall) / (model_precision + model_recall)

        model_stats = {
            "learning_algorithm": self.learning_algorithm.name,
            "domain_name": self.model_domain.name,
            "num_trajectories": len(used_observations),
            "num_trajectory_triplets": num_triplets,
            "total_number_of_actions": len(self.model_domain.actions),
            "#numeric_actions_learned_ok": actions_stats_counter["OK"],
            "#numeric_actions_no_solution": actions_stats_counter["no_solution_found"],
            "#numeric_actions_no_convex_hull": actions_stats_counter["convex_hull_not_found"],
            "#numeric_actions_infinite_number_solutions": actions_stats_counter["not_enough_data"],
            "model_precision": model_precision,
            "model_recall": model_recall,
            "model_f1_score": model_f1_score
        }
        self.numeric_learning_stats.append(model_stats)

    def write_complete_joint_statistics(self) -> NoReturn:
        """Writes a statistics file containing all the folds combined data."""
        output_path = self.results_dir_path / f"{self.learning_algorithm.name}_all_folds_numeric_learning_stats.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=NUMERIC_LEARNING_STAT_COLUMNS)
            writer.writeheader()
            writer.writerows(self.merged_numeric_stats)

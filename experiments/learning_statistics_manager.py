"""Module to manage the action model learning statistics."""
from collections import Counter
from pathlib import Path
from typing import Any, Dict, NoReturn, List

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from experiments.action_precision_recall_calculator import PrecisionRecallCalculator
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
    "#numeric_actions_infinite_number_solutions"
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
    model_domain: Domain
    learning_algorithm: str
    working_directory_path: Path
    results_dir_path: Path
    action_learning_stats: List[Dict[str, Any]]
    solving_stats: List[Dict[str, Any]]
    numeric_learning_stats: List[Dict[str, Any]]

    def __init__(self, working_directory_path: Path, domain_path: Path, learning_algorithm: str):
        self.working_directory_path = working_directory_path
        self.model_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
        self.learning_algorithm = learning_algorithm
        self.action_learning_stats = []
        self.solving_stats = {}
        self.numeric_learning_stats = {}
        self.results_dir_path = self.working_directory_path / "results_directory"

    def create_results_directory(self) -> NoReturn:
        """Creates the results directory that contains the learning results."""
        self.results_dir_path.mkdir(exist_ok=True)

    def add_to_action_stats(self, used_observations: List[Observation], learned_domain: LearnerDomain) -> NoReturn:
        """

        :param used_observations:
        :return:
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        action_appearance_counter = Counter()
        for observation in used_observations:
            for component in observation.components:
                action_appearance_counter[component.grounded_action_call.name] += 1
        precision_recall_calculator = PrecisionRecallCalculator()
        for action_name, action_data in learned_domain.actions.items():
            precision_recall_calculator.add_action_data(action_data, self.model_domain.actions[action_name])
            action_stats = {
                "learning_algorithm": self.learning_algorithm,
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

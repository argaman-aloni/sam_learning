"""Module for computing statistics on the experiment trajectories."""

import csv
from pathlib import Path
from typing import List, Dict

from pddl_plus_parser.models import Observation, Problem, Domain

TRAJECTORY_STATS_COLUMNS = [
    "min_trajectory_length",
    "max_trajectory_length",
    "average_trajectory_length",
    "average_number_of_problem_objects",
    "min_number_of_problem_objects",
    "max_number_of_problem_objects",
    "average_number_of_functions_in_problem",
    "min_number_of_functions_in_problem",
    "max_number_of_functions_in_problem",
    "average_number_of_predicates_in_problem",
    "min_number_of_predicates_in_problem",
    "max_number_of_predicates_in_problem",
    "action_distribution",
]


def compute_trajectory_statistics(trajectories: List[Observation], problems: List[Problem], domain: Domain) -> Dict[str, float]:
    """Compute statistics for the given trajectories."""
    action_distribution: Dict[str, int] = {action_name: 0 for action_name in domain.actions.keys()}
    number_of_problem_objects: List[int] = []
    number_of_functions_in_problem: List[int] = []
    number_of_predicates_in_problem: List[int] = []
    trajectory_lengths: List[int] = []

    for trajectory, problem in zip(trajectories, problems):
        trajectory_lengths.append(len(trajectory))
        number_of_problem_objects.append(len(problem.objects))
        number_of_functions_in_problem.append(len(problem.initial_state_fluents))
        number_of_predicates_in_problem.append(sum(len(preds) for preds in problem.initial_state_predicates.values()))
        for component in trajectory.components:
            action_distribution[component.grounded_action_call.name] += 1

    statistics = {
        "min_trajectory_length": min(trajectory_lengths),
        "max_trajectory_length": max(trajectory_lengths),
        "average_trajectory_length": sum(trajectory_lengths) / len(trajectory_lengths) if trajectory_lengths else 0,
        "average_number_of_problem_objects": (
            sum(number_of_problem_objects) / len(number_of_problem_objects) if number_of_problem_objects else 0
        ),
        "min_number_of_problem_objects": min(number_of_problem_objects),
        "max_number_of_problem_objects": max(number_of_problem_objects),
        "average_number_of_functions_in_problem": (
            sum(number_of_functions_in_problem) / len(number_of_functions_in_problem) if number_of_functions_in_problem else 0
        ),
        "min_number_of_functions_in_problem": min(number_of_functions_in_problem),
        "max_number_of_functions_in_problem": max(number_of_functions_in_problem),
        "average_number_of_predicates_in_problem": (
            sum(number_of_predicates_in_problem) / len(number_of_predicates_in_problem) if number_of_predicates_in_problem else 0
        ),
        "min_number_of_predicates_in_problem": min(number_of_predicates_in_problem),
        "max_number_of_predicates_in_problem": max(number_of_predicates_in_problem),
        "action_distribution": sum(action_distribution.values()) / len(action_distribution),
    }
    return statistics


def export_trajectory_statistics(statistics: Dict[str, float], filepath: Path) -> None:
    """Export the computed statistics to a file."""
    with open(filepath, "w") as csv_file:
        stats_writer = csv.DictWriter(csv_file, fieldnames=statistics.keys())
        stats_writer.writeheader()
        stats_writer.writerow(statistics)

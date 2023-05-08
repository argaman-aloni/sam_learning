"""Module to validate the correctness of the learned action models that were generated."""
import csv
import logging
import re
from pathlib import Path
from typing import NoReturn, Dict, List, Any, Optional, Union

from pddl_plus_parser.models import Observation, MultiAgentObservation

from solvers import FastDownwardSolver, MetricFFSolver, ENHSPSolver
from utilities import LearningAlgorithmType, SolverType, SolutionOutputTypes
from validators.validator_script_data import VALID_PLAN, INAPPLICABLE_PLAN, \
    GOAL_NOT_REACHED, run_validate_script

SOLVER_TYPES = {
    SolverType.fast_downward: FastDownwardSolver,
    SolverType.metric_ff: MetricFFSolver,
    SolverType.enhsp: ENHSPSolver
}

SOLVING_STATISTICS = [
    "learning_algorithm",
    "num_trajectories",
    "num_trajectory_triplets",
    "ok",
    "no_solution",
    "timeout",
    "solver_error",
    "not_applicable",
    "goal_not_achieved",
    "problems_ok",
    "problems_no_solution",
    "problems_timeout",
    "problems_not_applicable",
    "problems_goal_not_achieved"
]

DEBUG_STATISTICS = [
    "problems_ok",
    "problems_no_solution",
    "problems_timeout",
    "problems_solver_error",
    "problems_not_applicable",
    "problems_goal_not_achieved"
]

MAX_RUNNING_TIME = 60

ACTION_LINE_REGEX = r"(\[\d+, \d+\]): \(.*\)"


class DomainValidator:
    """Validates that the learned domain can create plans.

    Note:
        There is no validation on the plan content since the learning process is safe.
    """

    logger: logging.Logger
    solver: Union[ENHSPSolver, MetricFFSolver, FastDownwardSolver]
    solving_stats: List[Dict[str, Any]]
    aggregated_solving_stats: List[Dict[str, Any]]
    learning_algorithm: LearningAlgorithmType
    reference_domain_path: Path
    results_dir_path: Path

    def __init__(self, working_directory_path: Path,
                 learning_algorithm: LearningAlgorithmType, reference_domain_path: Path, solver_type: SolverType):
        self.logger = logging.getLogger(__name__)
        self.solver = SOLVER_TYPES[solver_type]()
        self.solving_stats = []
        self.aggregated_solving_stats = []
        self.learning_algorithm = learning_algorithm
        self.results_dir_path = working_directory_path / "results_directory"
        self.reference_domain_path = reference_domain_path

    @staticmethod
    def _clear_plans(test_set_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param test_set_directory: the path to the directory containing the plans.
        """
        for solver_output_path in test_set_directory.glob("*.solution"):
            solver_output_path.unlink(missing_ok=True)

    def _validate_solution_content(self, solution_file_path: Path, problem_file_path: Path,
                                   iteration_statistics: Dict[str, Union[int, List[str]]]) -> NoReturn:
        """Validates that the solution file contains a valid plan.

        :param solution_file_path: the path to the solution file.
        """
        validation_file_path = run_validate_script(domain_file_path=self.reference_domain_path,
                                                   problem_file_path=problem_file_path,
                                                   solution_file_path=solution_file_path)

        with open(validation_file_path, "r") as validation_file:
            validation_file_content = validation_file.read()
            if VALID_PLAN in validation_file_content:
                self.logger.info("The plan is valid.")
                iteration_statistics["ok"] += 1
                iteration_statistics["problems_ok"].append(problem_file_path.name)

            elif INAPPLICABLE_PLAN in validation_file_content:
                self.logger.info("The plan is not applicable.")
                iteration_statistics["not_applicable"] += 1
                iteration_statistics["problems_not_applicable"].append(problem_file_path.name)

            elif GOAL_NOT_REACHED in validation_file_content:
                self.logger.info("The plan did not reach the required goal.")
                iteration_statistics["goal_not_achieved"] += 1
                iteration_statistics["problems_goal_not_achieved"].append(problem_file_path.name)

    @staticmethod
    def _extract_num_triplets(used_observations: Union[List[Observation],
                                                       List[MultiAgentObservation], List[Path]] = None) -> int:
        """Extracts the number of trajectory triplets from the observations.

        :param used_observations: the observations used to generate the plans.
        :return: the number of trajectory triplets in the used observations.
        """
        for observation in used_observations:
            if isinstance(observation, Path):
                with open(observation, "r") as observation_file:
                    num_operators = 0
                    for line in observation_file.readlines():
                        match = re.match(ACTION_LINE_REGEX, line)
                        num_operators = num_operators + 1 if match else num_operators

                    return num_operators

        num_triplets = sum([len(observation.components) for observation in used_observations])
        return num_triplets

    def validate_domain(
            self, tested_domain_file_path: Path, test_set_directory_path: Optional[Path] = None,
            used_observations: Union[List[Union[Observation, MultiAgentObservation]], List[Path]] = None) -> NoReturn:
        """Validates that using the input domain problems can be solved.

        :param tested_domain_file_path: the path of the domain that was learned using POL.
        :param test_set_directory_path: the path to the directory containing the test set problems.
        :param used_observations: the observations that were used to learn the domain.
        """
        num_triplets = self._extract_num_triplets(used_observations)
        self.logger.info("Solving the test set problems using the learned domain!")
        solving_report = self.solver.execute_solver(
            problems_directory_path=test_set_directory_path,
            domain_file_path=tested_domain_file_path
        )
        solving_stats = {solution_type.name: 0 for solution_type in SolutionOutputTypes}
        for debug_statistic in DEBUG_STATISTICS:
            solving_stats[debug_statistic] = []

        for problem_file_name, entry in solving_report.items():
            if entry == SolutionOutputTypes.ok.name:
                solution_file_path = test_set_directory_path / f"{problem_file_name}.solution"
                problem_file_path = test_set_directory_path / f"{problem_file_name}.pddl"
                self._validate_solution_content(solution_file_path=solution_file_path,
                                                problem_file_path=problem_file_path,
                                                iteration_statistics=solving_stats)
                continue

            solving_stats[entry] += 1
            solving_stats[f"problems_{entry}"].append(problem_file_name)

        self.solving_stats.append({
            "learning_algorithm": self.learning_algorithm.name,
            "num_trajectories": len(used_observations),
            "num_trajectory_triplets": num_triplets,
            **solving_stats
        })
        self._clear_plans(test_set_directory_path)

    def write_statistics(self, fold_num: int) -> NoReturn:
        """Writes the statistics of the learned model into a CSV file.

        :param fold_num: the index of the fold that is currently being tested.
        """
        output_statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}" \
                                                         f"_problem_solving_stats_{fold_num}.csv"
        with open(output_statistics_path, 'wt', newline='') as csv_file:
            test_set_writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS)
            test_set_writer.writeheader()
            test_set_writer.writerows(self.solving_stats)

    def write_complete_joint_statistics(self) -> NoReturn:
        """Writes a statistics file containing all the folds combined data."""
        output_path = self.results_dir_path / f"{self.learning_algorithm.name}_all_folds_solving_stats.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS)
            writer.writeheader()
            writer.writerows(self.aggregated_solving_stats)

    def clear_statistics(self) -> NoReturn:
        """Clears the statistics so that each fold will have no relation to its predecessors."""
        self.aggregated_solving_stats.extend(self.solving_stats)
        self.solving_stats.clear()

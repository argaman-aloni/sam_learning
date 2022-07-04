"""Module to validate the correctness of the learned action models that were generated."""
import csv
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import NoReturn, Dict, List, Any, Optional, Tuple

from pddl_plus_parser.models import Observation

from solvers import FastDownwardSolver, MetricFFSolver, ENHSPSolver
from utilities import LearningAlgorithmType, SolverType, SolutionOutputTypes
from validators.validator_script_data import EXECUTION_SCRIPT, VALIDATOR_DIRECTORY, VALID_PLAN, INAPPLICABLE_PLAN, \
    GOAL_NOT_REACHED, BATCH_JOB_SUBMISSION_REGEX, write_batch_and_validate_plan

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
    "not_applicable",
    "goal_not_achieved",
]

MAX_RUNNING_TIME = 60


class DomainValidator:
    """Validates that the learned domain can create plans.

    Note:
        There is no validation on the plan content since the learning process is safe.
    """

    logger: logging.Logger
    solver: ENHSPSolver
    solving_stats: List[Dict[str, Any]]
    aggregated_solving_stats: List[Dict[str, Any]]
    learning_algorithm: LearningAlgorithmType
    reference_domain_path: Path
    results_dir_path: Path

    def __init__(self, working_directory_path: Path,
                 learning_algorithm: LearningAlgorithmType, reference_domain_path: Path):
        self.logger = logging.getLogger(__name__)
        self.solver = ENHSPSolver()
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
                                   iteration_statistics: Dict[str, int]) -> NoReturn:
        """Validates that the solution file contains a valid plan.

        :param solution_file_path: the path to the solution file.
        """
        script_file_path, validation_file_path = \
            write_batch_and_validate_plan(logger=self.logger, domain_path=self.reference_domain_path,
                                          problem_file_path=problem_file_path, solution_file_path=solution_file_path)

        self.logger.debug("Cleaning the sbatch and output file from the problems directory.")
        script_file_path.unlink(missing_ok=True)
        for job_file_path in Path(VALIDATOR_DIRECTORY).glob("job-*.out"):
            job_file_path.unlink(missing_ok=True)

        with open(validation_file_path, "r") as validation_file:
            validation_file_content = validation_file.read()
            if VALID_PLAN in validation_file_content:
                self.logger.info("The plan is valid.")
                iteration_statistics["ok"] += 1
            elif INAPPLICABLE_PLAN in validation_file_content:
                self.logger.info("The plan is not applicable.")
                iteration_statistics["not_applicable"] += 1
            elif GOAL_NOT_REACHED in validation_file_content:
                self.logger.info("The plan did not reach the required goal.")
                iteration_statistics["goal_not_achieved"] += 1

    def validate_domain(self, tested_domain_file_path: Path, test_set_directory_path: Optional[Path] = None,
                        used_observations: List[Observation] = None) -> NoReturn:
        """Validates that using the input domain problems can be solved.

        :param tested_domain_file_path: the path of the domain that was learned using POL.
        :param test_set_directory_path: the path to the directory containing the test set problems.
        :param used_observations: the observations that were used to learn the domain.
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        self.logger.info("Solving the test set problems using the learned domain!")
        solving_report = self.solver.execute_solver(
            problems_directory_path=test_set_directory_path,
            domain_file_path=tested_domain_file_path
        )
        solving_stats = {solution_type.name: 0 for solution_type in SolutionOutputTypes}
        for problem_file_name, entry in solving_report.items():
            if entry == SolutionOutputTypes.ok.name:
                solution_file_path = test_set_directory_path / f"{problem_file_name}.solution"
                problem_file_path = test_set_directory_path / f"{problem_file_name}.pddl"
                self._validate_solution_content(solution_file_path=solution_file_path,
                                                problem_file_path=problem_file_path,
                                                iteration_statistics=solving_stats)
                continue

            solving_stats[entry] += 1

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
"""Module to validate the correctness of the learned action models that were generated."""
import csv
import logging
import os
import shutil
from pathlib import Path
from typing import NoReturn, Dict, List, Any, Union, Optional

from pddl_plus_parser.exporters import MetricFFParser
from pddl_plus_parser.models import Observation

from experiments.util_types import LearningAlgorithmType, SolverType
from solvers import FastDownwardSolver, MetricFFSolver

SOLVER_TYPES = {
    SolverType.fast_downward: FastDownwardSolver,
    SolverType.metric_ff: MetricFFSolver
}

SOLVING_STATISTICS = [
    "learning_algorithm",
    "num_trajectories",
    "num_trajectory_triplets",
    "#problems_solved",
]


class SafeDomainValidator:
    """"""

    logger: logging.Logger
    expected_domain_path: str
    solver: Union[FastDownwardSolver, MetricFFSolver]
    solving_stats: List[Dict[str, Any]]
    validation_set_stats: List[Dict[str, Any]]
    learning_algorithm: LearningAlgorithmType
    results_dir_path: Path
    validation_directory_path: Path

    def __init__(self, working_directory_path: Path, solver_type: SolverType,
                 learning_algorithm: LearningAlgorithmType):
        self.logger = logging.getLogger(__name__)
        self.solver = SOLVER_TYPES[solver_type]()
        self.solving_stats = []
        self.validation_set_stats = []
        self.learning_algorithm = learning_algorithm
        self.validation_directory_path = working_directory_path / "validation_set"
        self.results_dir_path = working_directory_path / "results_directory"

    def copy_validation_problems(self, tested_domain_file_path: Path, validation_set_problems: List[Path]) -> NoReturn:
        """

        :param tested_domain_file_path:
        :param validation_set_problems:
        :return:
        """
        self.logger.info("Copying the tested domain and the validation set problems.")
        shutil.copy(tested_domain_file_path, self.validation_directory_path / tested_domain_file_path.name)
        for problem_path in validation_set_problems:
            shutil.copy(problem_path, self.validation_directory_path / problem_path.name)

    def validate_domain(self, tested_domain_file_path: Path, test_set_directory_path: Optional[Path] = None,
                        used_observations: List[Observation] = None, is_validation: bool = False) -> NoReturn:
        """

        :param tested_domain_file_path:
        :param test_set_directory_path:
        :param used_observations:
        :param is_validation:
        :return:
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        self.logger.info("Solving the test set problems using the learned domain!")
        script_file_name = "solver_execution_script.sh"
        execution_script_path = test_set_directory_path / script_file_name if not is_validation \
            else self.validation_directory_path / script_file_name
        problems_dir_path = test_set_directory_path if is_validation else self.validation_directory_path
        self.solver.write_batch_and_execute_solver(
            script_file_path=execution_script_path,
            problems_directory_path=problems_dir_path,
            domain_file_path=tested_domain_file_path
        )
        num_valid_solutions = 0
        for solution_file_path in test_set_directory_path.glob("*.solution"):
            num_valid_solutions += self._validate_solution_content(solution_file_path)

        solver_stats = self.solving_stats if not is_validation else self.validation_set_stats
        solver_stats.append({
            "learning_algorithm": self.learning_algorithm.name,
            "num_trajectories": len(used_observations),
            "num_trajectory_triplets": num_triplets,
            "#problems_solved": num_valid_solutions,
        })
        self._clear_plans(problems_dir_path)

    @staticmethod
    def _clear_plans(test_set_directory: Path) -> NoReturn:
        """

        :param test_set_directory:
        """
        for solver_output_path in test_set_directory.glob("*.solution"):
            os.remove(solver_output_path)

    def clear_validation_problems(self) -> NoReturn:
        """"""
        for problem_file_path in self.validation_set_stats.glob("pfile*.pddl"):
            os.remove(problem_file_path)

    def write_statistics(self, fold_num: int) -> NoReturn:
        """Writes the statistics of the learned model into a CSV file.

        :param fold_num:
        """
        output_statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}" \
                                                         f"_problem_solving_stats_{fold_num}.csv"
        validation_statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}" \
                                                             f"validation_testing_stats_{fold_num}.csv"
        with open(output_statistics_path, 'wt', newline='') as csv_file, \
                open(validation_statistics_path, 'wt', newline='') as validations_stats_file:
            test_set_writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS)
            validation_set_writer = csv.DictWriter(validations_stats_file, fieldnames=SOLVING_STATISTICS)
            test_set_writer.writeheader()
            validation_set_writer.writeheader()
            validation_set_writer.writerows(self.validation_set_stats)
            test_set_writer.writerows(self.solving_stats)
            # for data in self.solving_stats:
            #     test_set_writer.writerow(data)

    def _validate_solution_content(self, solution_file_path: Path) -> int:
        """

        :param solution_file_path:
        :return:
        """
        if self.learning_algorithm == LearningAlgorithmType.numeric_sam:
            return 1 if MetricFFParser().is_valid_plan_file(solution_file_path) else 0

        return 0  # TODO: complete for fast downward.

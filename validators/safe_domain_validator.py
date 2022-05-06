"""Module to validate the correctness of the learned action models that were generated."""
import csv
import logging
import os
import shutil
from pathlib import Path
from typing import NoReturn, Dict, List, Any, Union, Optional

from pddl_plus_parser.exporters import MetricFFParser, TrajectoryExporter
from pddl_plus_parser.models import Observation, Domain
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser

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
    "ok",
    "no-solution",
    "timeout",
    "not-applicable"
]

OK = "ok"
NOT_APPLICABLE = "not-applicable"


class DomainValidator:
    """Validates that the learned domain can create plans.

    Note:
        There is no validation on the plan content since the learning process is safe.
    """

    logger: logging.Logger
    reference_domain: Domain
    solver: Union[FastDownwardSolver, MetricFFSolver]
    solving_stats: List[Dict[str, Any]]
    aggregated_solving_path: List[Dict[str, Any]]
    validation_set_stats: List[Dict[str, Any]]
    learning_algorithm: LearningAlgorithmType
    results_dir_path: Path
    validation_directory_path: Path

    def __init__(self, working_directory_path: Path, solver_type: SolverType,
                 learning_algorithm: LearningAlgorithmType, reference_domain_path: Path):
        self.logger = logging.getLogger(__name__)
        self.solver = SOLVER_TYPES[solver_type]()
        self.solving_stats = []
        self.validation_set_stats = []
        self.aggregated_solving_path = []
        self.learning_algorithm = learning_algorithm
        self.validation_directory_path = working_directory_path / "validation_set"
        self.results_dir_path = working_directory_path / "results_directory"
        self.reference_domain = DomainParser(domain_path=reference_domain_path, partial_parsing=False).parse_domain()

    def copy_validation_problems(self, tested_domain_file_path: Path, validation_set_problems: List[Path]) -> NoReturn:
        """Copies the problems that were used in the learning process to the validation set directory.

        :param tested_domain_file_path: the domain that was learned in the current iteration.
        :param validation_set_problems: the problems to copy.
        """
        self.logger.info("Copying the tested domain and the validation set problems.")
        shutil.copy(tested_domain_file_path, self.validation_directory_path / tested_domain_file_path.name)
        for problem_path in validation_set_problems:
            shutil.copy(problem_path, self.validation_directory_path / problem_path.name)

    def validate_domain(self, tested_domain_file_path: Path, test_set_directory_path: Optional[Path] = None,
                        used_observations: List[Observation] = None, is_validation: bool = False) -> NoReturn:
        """Validates that using the input domain problems can be solved.

        :param tested_domain_file_path: the path of the domain that was learned using POL.
        :param test_set_directory_path: the path to the directory containing the test set problems.
        :param used_observations: the observations that were used to learn the domain.
        :param is_validation: whether the problems belong to the validation set or to the test set.
        """
        num_triplets = sum([len(observation.components) for observation in used_observations])
        self.logger.info("Solving the test set problems using the learned domain!")
        script_file_name = "solver_execution_script.sh"
        execution_script_path = test_set_directory_path / script_file_name if not is_validation \
            else self.validation_directory_path / script_file_name
        problems_dir_path = test_set_directory_path if not is_validation else self.validation_directory_path
        self.solver.write_batch_and_execute_solver(
            script_file_path=execution_script_path,
            problems_directory_path=problems_dir_path,
            domain_file_path=tested_domain_file_path
        )
        solving_stats = {"ok": 0, "no-solution": 0, "timeout": 0, "not-applicable": 0}
        for solution_file_path in test_set_directory_path.glob("*.solution"):
            problem_file_path = problems_dir_path / f"{solution_file_path.stem}.pddl"
            self._validate_solution_content(solution_file_path, problem_file_path, solving_stats)

        validation_stats = self.solving_stats if not is_validation else self.validation_set_stats
        validation_stats.append({
            "learning_algorithm": self.learning_algorithm.name,
            "num_trajectories": len(used_observations),
            "num_trajectory_triplets": num_triplets,
            **solving_stats
        })
        self._clear_plans(problems_dir_path)

    @staticmethod
    def _clear_plans(test_set_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param test_set_directory: the path to the directory containg the plans.
        """
        for solver_output_path in test_set_directory.glob("*.solution"):
            os.remove(solver_output_path)

    def clear_validation_problems(self) -> NoReturn:
        """Clears the copied validation set problems."""
        for problem_file_path in self.validation_directory_path.glob("pfile*.pddl"):
            os.remove(problem_file_path)

    def write_statistics(self, fold_num: int) -> NoReturn:
        """Writes the statistics of the learned model into a CSV file.

        :param fold_num: the index of the fold that is currently being tested.
        """
        output_statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}" \
                                                         f"_problem_solving_stats_{fold_num}.csv"
        validation_statistics_path = self.results_dir_path / f"{self.learning_algorithm.name}" \
                                                             f"_validation_testing_stats_{fold_num}.csv"
        with open(output_statistics_path, 'wt', newline='') as csv_file, \
                open(validation_statistics_path, 'wt', newline='') as validations_stats_file:
            test_set_writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS)
            validation_set_writer = csv.DictWriter(validations_stats_file, fieldnames=SOLVING_STATISTICS)
            test_set_writer.writeheader()
            validation_set_writer.writeheader()
            validation_set_writer.writerows(self.validation_set_stats)
            test_set_writer.writerows(self.solving_stats)

    def write_complete_joint_statistics(self) -> NoReturn:
        """Writes a statistics file containing all the folds combined data."""
        output_path = self.results_dir_path / f"{self.learning_algorithm.name}_all_folds_solving_stats.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS)
            writer.writeheader()
            writer.writerows(self.validation_set_stats)

    def clear_statistics(self) -> NoReturn:
        """Clears the statistics so that each fold will have no relation to its predecessors."""
        self.validation_set_stats.clear()
        self.aggregated_solving_path.extend(self.solving_stats)
        self.solving_stats.clear()

    def _validate_solution_content(self, solution_file_path: Path, problem_file_path: Path,
                                   iteration_statistics: Dict[str, int]) -> NoReturn:
        """Validates that the solution file contains a valid plan.

        :param solution_file_path: the path to the solution file.
        :return: 1 if there is a plan in the solution file, zero otherwise.
        """
        if self.learning_algorithm == LearningAlgorithmType.numeric_sam:
            solution_status, sequence = MetricFFParser().get_solving_status(solution_file_path)
            if solution_status == OK:
                self.logger.debug("The planner created a solution! "
                                  "Will now validate that the action sequence is applicable!")
                problem = ProblemParser(problem_file_path, self.reference_domain).parse_problem()
                try:
                    TrajectoryExporter(self.reference_domain).parse_plan(problem=problem, action_sequence=sequence)
                    iteration_statistics[OK] += 1
                    return

                except ValueError:
                    self.logger.warning("Found a solution that was not applicable!")
                    iteration_statistics[NOT_APPLICABLE] += 1

            iteration_statistics[solution_status] += 1
            return

        return  # TODO: complete for fast downward.

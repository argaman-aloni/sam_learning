"""Module to validate the correctness of the learned action models that were generated."""
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from pddl_plus_parser.models import Observation, MultiAgentObservation

from solvers import FastDownwardSolver, MetricFFSolver, ENHSPSolver, FFADLSolver
from utilities import SolverType, SolutionOutputTypes

from validators import DomainValidator
from utilities import NegativePreconditionPolicy
from utilities import MacroActionParser, MappingElement
from validators.safe_domain_validator import (
    SOLVING_STATISTICS,
    SOLVER_TYPES,
    DEBUG_STATISTICS,
    VALIDATED_AGAINST_EXPERT_PLAN,
    NOT_VALIDATED_AGAINST_EXPERT_PLAN,
    NUMERIC_STATISTICS_LABELS,
)

SOLVING_STATISTICS_MACRO = ["fold", "policy", *SOLVING_STATISTICS]


class MacroDomainValidator(DomainValidator):
    """Validates that the learned domain can create plans.

    Note:
        There is no validation on the plan content since the learning process is safe.
    """

    def validate_domain_macro(
        self,
        fold: int,
        policy: NegativePreconditionPolicy,
        tested_domain_file_path: Path,
        test_set_directory_path: Optional[Path] = None,
        used_observations: Union[List[Union[Observation, MultiAgentObservation]], List[Path]] = None,
        tolerance: float = 0.01,
        timeout: int = 5,
        learning_time: float = 0,
        solvers_portfolio: List[SolverType] = None,
        mapping: Dict[str, MappingElement] = None,
    ) -> None:
        """Validates that using the input domain problems can be solved.

        :param tested_domain_file_path: the path of the domain that was learned using POL.
        :param test_set_directory_path: the path to the directory containing the test set problems.
        :param used_observations: the observations that were used to learn the domain.
        :param timeout: the timeout for the solver.
        :param tolerance: the numeric tolerance to use.
        :param learning_time: the time it took to learn the domain (in seconds).
        :param solvers_portfolio: the solvers to use for the validation, can be one or more and each will try to solve each planning problem at most once..
        :param mapping: the learned model mapper from macro action to mapping element.

        """
        num_triplets = self._extract_num_triplets(used_observations)
        solving_stats: Dict[str, Any] = {label: 0 for label in NUMERIC_STATISTICS_LABELS}
        for debug_statistic in DEBUG_STATISTICS:
            solving_stats[debug_statistic] = []

        self.logger.info("Solving the test set problems using the learned domain!")
        for problem_path in test_set_directory_path.glob(f"{self.problem_prefix}*.pddl"):
            problem_solving_report = {}
            problem_solved = False
            problem_file_name = problem_path.stem
            for solver_type in solvers_portfolio:
                start_time = time.time()
                solver = SOLVER_TYPES[solver_type]()
                solver.solve_problem(
                    problems_directory_path=test_set_directory_path,
                    domain_file_path=tested_domain_file_path,
                    problem_file_path=problem_path,
                    solving_timeout=timeout,
                    solving_stats=problem_solving_report,
                )
                end_time = time.time()
                solving_stats["solving_time"] = end_time - start_time  # time in seconds
                if problem_solving_report[problem_file_name] == SolutionOutputTypes.ok.name:
                    problem_solved = True
                    solution_file_path = test_set_directory_path / f"{problem_file_name}.solution"

                    if mapping:
                        self.adapt_solution_file(solution_path=solution_file_path, mapping=mapping)

                    self._validate_solution_content(
                        solution_file_path=solution_file_path, problem_file_path=problem_path, iteration_statistics=solving_stats
                    )
                    break

            if problem_solved:
                continue

            expert_plan_path = self.working_directory_path / f"{problem_file_name}.solution"
            solving_outcome = problem_solving_report[problem_file_name]
            solving_stats[solving_outcome] += 1
            solving_stats[f"problems_{solving_outcome}"].append(problem_file_name)
            self._validate_against_expert_plan(
                solution_file_path=expert_plan_path,
                problem_file_path=problem_path,
                iteration_statistics=solving_stats,
                tested_domain_path=tested_domain_file_path,
            )

        self._calculate_solving_percentages(solving_stats)
        self._calculate_expert_validation_statistics(solving_stats)
        num_trajectories = len(used_observations)
        self.solving_stats.append(
            {
                "fold": fold,
                "learning_algorithm": self.learning_algorithm.name,
                "policy": policy,
                "num_trajectories": num_trajectories,
                "num_trajectory_triplets": num_triplets,
                "learning_time": learning_time,
                "solver": [solver.name for solver in solvers_portfolio],
                **solving_stats,
            }
        )
        self._clear_plans(test_set_directory_path)

    def write_statistics(self, fold_num: int, iteration: Optional[int] = None) -> None:
        """Writes the statistics of the learned model into a CSV file.

        :param fold_num: the index of the fold that is currently being tested.
        :param iteration: the index of the iteration that is currently being tested.
        """
        output_statistics_path = self.results_dir_path / (
            f"{self.learning_algorithm.name}"
            f"_problem_solving_stats_fold_{fold_num}"
            f"{f'_{iteration}_trajectories' if iteration is not None else ''}.csv"
        )
        with open(output_statistics_path, "wt", newline="") as csv_file:
            test_set_writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS_MACRO)
            test_set_writer.writeheader()
            test_set_writer.writerows(self.solving_stats)

    def write_complete_joint_statistics(self, fold: Optional[int] = None) -> None:
        """Writes a statistics file containing all the folds combined data."""

        output_path = (
            self.results_dir_path / f"{self.learning_algorithm.name}_all_folds_solving_stats.csv"
            if fold is None
            else self.results_dir_path / f"{self.learning_algorithm.name}_problem_solving_stats_{fold}.csv"
        )
        with open(output_path, "wt", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SOLVING_STATISTICS_MACRO)
            writer.writeheader()
            writer.writerows(self.aggregated_solving_stats)

    @staticmethod
    def adapt_solution_file(solution_path: Path, mapping: Dict[str, MappingElement]):
        """
        Post Processing solution files with macro actions and adapt them to something validators can work with.
        That essentially means replacing macro action lines with its consisting solo actions lines.

        :param parsed_domain: the learned domain as a parsed full domain.
        :param solution_path: the path to the solution file output from the learned domain.
        :param mapping: the macro actions mapping of the learned domain.
        """
        with open(solution_path, "r") as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            extracted_lines = MacroActionParser.extract_actions_from_macro_action(action_line=line, mapper=mapping)
            new_lines.extend([extracted_line if extracted_line.endswith("\n") else f"{extracted_line}\n" for extracted_line in extracted_lines])

        with open(solution_path, "w") as file:
            file.writelines(new_lines)

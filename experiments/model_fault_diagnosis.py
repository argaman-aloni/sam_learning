"""Conducts the experiments to test how efficiently the learning algorithm can diagnose faults."""
import csv
import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import NoReturn, List, Dict, Any

from pddl_plus_parser.exporters import ENHSPParser, TrajectoryExporter
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, Domain

from experiments.experiments_trajectories_creator import ExperimentTrajectoriesCreator
from experiments.k_fold_split import KFoldSplit
from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner
from solvers import ENHSPSolver
from utilities import SolverType, SolutionOutputTypes

random.seed(42)
DIAGNOSIS_COLUMNS = ["domain_type", "problems_type", "ok", "no_solution", "timeout", "not_applicable"]


class ModelFaultDiagnosis:
    """Class that contains the logic to conduct fault diagnosis experimentation."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    model_domain_file_name: str
    solver: ENHSPSolver
    fluents_map: Dict[str, List[str]]

    def __init__(self, work_dir_path: Path, faulty_domain_path: Path, original_domain_file_name: str,
                 fluents_map_path: Path):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = work_dir_path
        self.k_fold = KFoldSplit(working_directory_path=work_dir_path, domain_file_name=original_domain_file_name,
                                 only_train_test=True)
        self.model_domain_file_name = original_domain_file_name
        self.solver = ENHSPSolver()
        with open(fluents_map_path, "rt") as json_file:
            self.fluents_map = json.load(json_file)

        self.results_dir_path = work_dir_path / "results_directory"
        self.faulty_domain_path = faulty_domain_path
        self.model_domain_file_path = self.working_directory_path / self.model_domain_file_name
        self.model_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()

    def export_domain(self, learned_domain: LearnerDomain, domain_directory_path: Path) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param domain_directory_path: the path to the test set directory where the domain would be exported to.
        """
        domain_path = domain_directory_path / self.model_domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def _clear_plans(self, working_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param working_directory: the path to the directory containing the plans.
        """
        for solver_output_path in working_directory.glob("*.solution"):
            os.remove(solver_output_path)
        self._delete_trajectories(working_directory)

    @staticmethod
    def _delete_trajectories(train_set_dir_path: Path) -> NoReturn:
        """Removes the trajectories created by the original domain prior to the added defect.

        :param train_set_dir_path: the path to the train set directory containing the problems to solve.
        """
        for trajectory_file_path in train_set_dir_path.glob("*.trajectory"):
            os.remove(trajectory_file_path)

    def _generate_faulty_domain(self, train_set_dir_path: Path) -> Domain:
        """

        :param train_set_dir_path:
        :return:
        """
        faulty_domain = DomainParser(domain_path=self.faulty_domain_path, partial_parsing=False).parse_domain()
        os.remove(train_set_dir_path / self.model_domain_file_name)
        shutil.copy(self.faulty_domain_path, train_set_dir_path / self.faulty_domain_path.name)
        self._delete_trajectories(train_set_dir_path)
        return faulty_domain

    def _extract_applicable_trajectories(self, plans_dir_path: Path, domain: Domain) -> List[Observation]:
        """

        :param plans_dir_path:
        :param domain:
        :return:
        """
        copied_file_path = shutil.copy(self.model_domain_file_path, plans_dir_path)
        trajectory_creator = ExperimentTrajectoriesCreator(self.model_domain_file_name, plans_dir_path)
        trajectory_creator.fix_solution_files(SolverType.enhsp)
        trajectory_creator.create_domain_trajectories()

        trajectories = []
        for trajectory_path in plans_dir_path.glob("*.trajectory"):
            problem_path = plans_dir_path / f"{trajectory_path.stem}.pddl"
            problem = ProblemParser(problem_path, domain).parse_problem()
            trajectories.append(TrajectoryParser(domain, problem).parse_trajectory(trajectory_path))

        os.remove(copied_file_path)
        return trajectories

    def _learn_safe_action_model(self, faulty_domain: Domain, test_set_dir_path: Path,
                                 train_set_dir_path: Path) -> NoReturn:
        """

        :param faulty_domain:
        :param test_set_dir_path:
        :param train_set_dir_path:
        :return:
        """
        train_trajectories = self._extract_applicable_trajectories(train_set_dir_path, self.model_domain)
        learner = NumericSAMLearner(partial_domain=faulty_domain, preconditions_fluent_map=self.fluents_map)
        learned_model, _ = learner.learn_action_model(train_trajectories)
        self.export_domain(learned_model, test_set_dir_path)

    def _write_diagnosis(self, all_diagnosis_stats: List[Dict[str, Any]]) -> NoReturn:
        """

        :param all_diagnosis_stats:
        :return:
        """
        output_path = self.results_dir_path / f"diagnosis_statistics.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DIAGNOSIS_COLUMNS)
            writer.writeheader()
            writer.writerows(all_diagnosis_stats)

    def _solve_and_validate(self, problems_dir_path: Path,
                            domain_file_path: Path, domain_type: str, problems_type: str) -> Dict[str, Any]:
        """

        :param problems_dir_path:
        :param domain_file_path:
        :param domain_type:
        :param problems_type:
        :return:
        """
        self.logger.info(f"Starting to work on solving the problems in the directory - {problems_dir_path.absolute()}")
        solving_report = self.solver.execute_solver(
            problems_directory_path=problems_dir_path,
            domain_file_path=domain_file_path
        )
        statistics = {solution_type.name: 0 for solution_type in SolutionOutputTypes}
        counted_stats = Counter(list(solving_report.values()))
        statistics.update(counted_stats)
        statistics["domain_type"] = domain_type
        statistics["problems_type"] = problems_type

        for problem_name, solving_status in solving_report.items():
            if solving_status == SolutionOutputTypes.ok:
                problem_file_path = problems_dir_path / f"{problem_name}.pddl"
                solution_file_path = problems_dir_path / f"{problem_name}.solution"
                sequence = ENHSPParser().parse_plan_content(solution_file_path)
                problem = ProblemParser(problem_file_path, self.model_domain).parse_problem()
                self.logger.debug("Validating the applicability of the plan.")
                try:
                    TrajectoryExporter(self.model_domain).parse_plan(problem=problem, action_sequence=sequence)

                except ValueError:
                    self.logger.warning("The parsed plan is inapplicable according to the model domain!")
                    statistics[SolutionOutputTypes.not_applicable.name] += 1

        return statistics

    def evaluate_fault_diagnosis(self, train_set_dir_path: Path, test_set_dir_path: Path) -> NoReturn:
        """

        :param train_set_dir_path:
        :param test_set_dir_path:
        :return:
        """
        all_diagnosis_stats = []
        self.logger.info("Starting the fault diagnosis simulation!")
        faulty_domain = self._generate_faulty_domain(train_set_dir_path)
        self.logger.debug("Solving the train set problems using the faulty domain.")
        faulty_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=self.faulty_domain_path, domain_type="faulty",
            problems_type="train")
        all_diagnosis_stats.append(faulty_train_stats)

        self.logger.debug("Learning the action model based on the train set!")
        self._learn_safe_action_model(faulty_domain, test_set_dir_path, train_set_dir_path)
        learned_domain_file_path = test_set_dir_path / self.model_domain_file_name

        self.logger.debug("Solving the test set problems using the learned SAFE domain.")
        safe_test_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=learned_domain_file_path, domain_type="safe",
            problems_type="test")
        all_diagnosis_stats.append(safe_test_stats)
        self._clear_plans(test_set_dir_path)

        self.logger.debug("solving the test set problems using the FAULTY domain.")
        faulty_test_set_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=self.faulty_domain_path, domain_type="faulty",
            problems_type="test")
        all_diagnosis_stats.append(faulty_test_set_stats)
        self._clear_plans(train_set_dir_path)

        self.logger.debug("solving the train set problems (again for validation) using the SAFE learned domain.")
        safe_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=learned_domain_file_path, domain_type="safe",
            problems_type="train")
        all_diagnosis_stats.append(safe_train_stats)

        self._write_diagnosis(all_diagnosis_stats)
        print(all_diagnosis_stats)

    def run_fault_diagnosis(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        for train_dir_path, test_dir_path in self.k_fold.create_k_fold():
            self.evaluate_fault_diagnosis(train_dir_path, test_dir_path)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    diagnoser = ModelFaultDiagnosis(
        work_dir_path=Path("/sise/home/mordocha/numeric_planning/domains/depot_diagnosis/"),
        faulty_domain_path=Path("/sise/home/mordocha/numeric_planning/domains/depot_diagnosis/faulty_domain.pddl"),
        fluents_map_path=Path("/sise/home/mordocha/numeric_planning/domains/depot_diagnosis/depot_fluents_map.json"),
        original_domain_file_name="depot_numeric.pddl"
    )
    diagnoser.run_fault_diagnosis()

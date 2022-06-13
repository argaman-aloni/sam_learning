"""Conducts the experiments to test how efficiently the learning algorithm can diagnose faults."""
import csv
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import NoReturn, List, Dict, Any, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Observation, Domain

from experiments.k_fold_split import KFoldSplit
from fault_detection import FaultGenerator, FaultRepair
from sam_learning.core import LearnerDomain
from solvers import ENHSPSolver
from utilities import SolutionOutputTypes

FAULTY_DOMAIN_PDDL = "faulty_domain.pddl"

random.seed(42)
DIAGNOSIS_COLUMNS = ["domain_type", "problems_type", "ok", "no_solution", "timeout", "not_applicable",
                     "state_difference"]
ACTION_FAULT_DETECTION_COLUMNS = ["action_name", "is_precondition_faulty", "is_effect_faulty"]


class ModelFaultDiagnosis:
    """Class that contains the logic to conduct fault diagnosis experimentation."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    model_domain_file_name: str
    solver: ENHSPSolver
    fluents_map: Dict[str, List[str]]
    results_dir_path: Path
    model_domain_file_path: Path
    model_domain: Domain

    def __init__(self, work_dir_path: Path, original_domain_file_name: str, fluents_map_path: Path):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = work_dir_path
        self.k_fold = KFoldSplit(working_directory_path=work_dir_path, domain_file_name=original_domain_file_name,
                                 only_train_test=True)
        self.model_domain_file_name = original_domain_file_name
        self.solver = ENHSPSolver()
        self.fault_repair = FaultRepair(work_dir_path, original_domain_file_name, fluents_map_path)
        self.fault_generator = FaultGenerator(work_dir_path, original_domain_file_name)

        self.results_dir_path = work_dir_path / "results_directory"
        self.model_domain_file_path = self.working_directory_path / self.model_domain_file_name
        self.model_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()

    def _export_domain(self, domain: LearnerDomain, domain_directory_path: Path,
                       domain_file_name: Optional[str] = None) -> NoReturn:
        """Exports a domain into a file so that it will be used to solve the test set problems.

        :param domain: the domain to export
        :param domain_directory_path: the path to the directory in which the domain will be copied to.
        :param domain_file_name: the name of the domain file in case it differs from the original name.
        """
        domain_path = domain_directory_path / domain_file_name if domain_file_name is not None else \
            domain_directory_path / self.model_domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(domain.to_pddl())

    def _clear_plans(self, working_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param working_directory: the path to the directory containing the plans.
        """
        for solver_output_path in working_directory.glob("*.solution"):
            os.remove(solver_output_path)

    def _generate_faulty_domain(self, directory_path: Path) -> LearnerDomain:
        """Generate d domain with a defect of sort that makes it unsafe.

        Note:
            currently support removing preconditions as defect options.

        :param directory_path: the path to the directory where the domain file will be written to.
        :return: the domain with the randomized defect.
        """
        faulty_domain = self.fault_generator.generate_faulty_domain()
        os.remove(directory_path / self.model_domain_file_name)
        self._export_domain(faulty_domain, directory_path, FAULTY_DOMAIN_PDDL)
        return faulty_domain

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

    def _solve_and_validate(
            self, problems_dir_path: Path, domain_file_path: Path,
            domain_type: str, problems_type: str) -> Tuple[List[Observation], List[Observation], Dict[str, Any]]:
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
        self.logger.info(f"Finished solving the problems in the directory - {problems_dir_path.absolute()}")
        statistics = {solution_type.name: 0 for solution_type in SolutionOutputTypes}
        counted_stats = Counter(list(solving_report.values()))
        statistics.update(counted_stats)
        statistics["domain_type"] = domain_type
        statistics["problems_type"] = problems_type

        valid_observations, faulty_observations, faults_detected = self.fault_repair.execute_plans_on_agent(
            problems_dir_path, domain_file_path)

        counted_fault_detection = Counter(list(faults_detected.values()))
        statistics["ok"] = counted_fault_detection["ok"]
        statistics["not_applicable"] = counted_fault_detection["not_applicable"]
        statistics["state_difference"] = counted_fault_detection["state_difference"]

        return valid_observations, faulty_observations, statistics

    def evaluate_fault_diagnosis(self, train_set_dir_path: Path, test_set_dir_path: Path) -> NoReturn:
        """Conducts the experiments to evaluate the efficiency of the 'repair' property.

        :param train_set_dir_path: the path to the train set folder.
        :param test_set_dir_path: the path to the test set folder.
        """
        all_diagnosis_stats = []
        self.logger.info("Starting the fault diagnosis simulation!")
        faulty_domain = self._generate_faulty_domain(train_set_dir_path)
        self.logger.debug("Solving the train set problems using the faulty domain.")
        faulty_domain_path = train_set_dir_path / FAULTY_DOMAIN_PDDL
        valid_observations, faulty_observations, faulty_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=faulty_domain_path, domain_type="faulty",
            problems_type="train")
        all_diagnosis_stats.append(faulty_train_stats)

        if len(valid_observations) == 0:
            self.logger.warning("No valid observations found in the train set.")
            raise ValueError("No valid observations found in the train set.")

        faulty_action_name = valid_observations[0].components[0].grounded_action_call.name
        self.logger.debug(f"Found a defected action! action - {faulty_action_name}")
        repaired_domain = self.fault_repair.repair_model(faulty_domain, valid_observations, faulty_action_name)
        self._export_domain(repaired_domain, test_set_dir_path)
        learned_domain_file_path = test_set_dir_path / self.model_domain_file_name

        self.logger.debug("Solving the test set problems using the learned SAFE domain.")
        safe_test_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=learned_domain_file_path, domain_type="safe",
            problems_type="test")
        all_diagnosis_stats.append(safe_test_stats)
        self._clear_plans(test_set_dir_path)

        self.logger.debug("solving the test set problems using the FAULTY domain.")
        faulty_test_set_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=faulty_domain_path, domain_type="faulty",
            problems_type="test")
        all_diagnosis_stats.append(faulty_test_set_stats)
        self._clear_plans(train_set_dir_path)

        self.logger.debug("solving the train set problems (again for validation) using the SAFE learned domain.")
        safe_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=learned_domain_file_path, domain_type="safe",
            problems_type="train")
        all_diagnosis_stats.append(safe_train_stats)

        self._write_diagnosis(all_diagnosis_stats)

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
        fluents_map_path=Path("/sise/home/mordocha/numeric_planning/domains/depot_diagnosis/depot_fluents_map.json"),
        original_domain_file_name="depot_numeric.pddl"
    )
    diagnoser.run_fault_diagnosis()

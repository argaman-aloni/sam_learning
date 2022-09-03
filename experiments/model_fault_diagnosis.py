"""Conducts the experiments to test how efficiently the learning algorithm can diagnose faults."""
import argparse
import csv
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import NoReturn, List, Dict, Any, Optional, Tuple, Iterator

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Observation, Domain

from experiments import LearningStatisticsManager
from experiments.k_fold_split import KFoldSplit
from fault_detection import FaultGenerator, FaultRepair, DefectType, RepairAlgorithmType
from sam_learning.core import LearnerDomain
from solvers import ENHSPSolver
from utilities import SolutionOutputTypes, LearningAlgorithmType

FAULTY_DOMAIN_PDDL = "faulty_domain.pddl"

random.seed(42)
DIAGNOSIS_COLUMNS = ["repair_method", "domain_type", "problems_type", "ok", "no_solution", "timeout", "not_applicable",
                     "goal_not_achieved", "state_difference", "action_name", "defect_type"]

REPAIR_TO_LEARNING_ALGORITHM = {
    RepairAlgorithmType.oblique_tree: LearningAlgorithmType.oblique_tree,
    RepairAlgorithmType.extended_svc: LearningAlgorithmType.extended_svc,
    RepairAlgorithmType.numeric_sam: LearningAlgorithmType.numeric_sam,
    RepairAlgorithmType.raw_numeric_sam: LearningAlgorithmType.raw_numeric_sam
}


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
        self._action_index = 0
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=work_dir_path,
            domain_path=self.model_domain_file_path,
            learning_algorithm=LearningAlgorithmType.numeric_sam)

    def _export_domain(self, domain: LearnerDomain, domain_directory_path: Path,
                       domain_file_name: Optional[str] = None,
                       is_faulty: bool = False,
                       defect_type: DefectType = DefectType.numeric_precondition_sign,
                       action_name: str = "", repair_type: str = "") -> NoReturn:
        """Exports a domain into a file so that it will be used to solve the test set problems.

        :param domain: the domain to export
        :param domain_directory_path: the path to the directory in which the domain will be copied to.
        :param domain_file_name: the name of the domain file in case it differs from the original name.
        """
        domain_path = domain_directory_path / domain_file_name if domain_file_name is not None else \
            domain_directory_path / self.model_domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(domain.to_pddl())

        self.logger.debug("Exporting the domain to the results directory!")
        faulty = "faulty" if is_faulty else "repaired"
        with open(self.results_dir_path / f"{faulty}_domain_{action_name}_{repair_type}_{defect_type.name}.pddl",
                  "wt") as domain_file:
            domain_file.write(domain.to_pddl())

    @staticmethod
    def _clear_plans(working_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param working_directory: the path to the directory containing the plans.
        """
        for solver_output_path in working_directory.glob("*.solution"):
            solver_output_path.unlink(missing_ok=True)

    def _generate_faulty_domain_based_on_defect_type(
            self, action_name: str, directory_path: Path, defect_type: DefectType) -> LearnerDomain:
        """Generates a domain with a defect based on the defect type and the action name.

        :param action_name: the name of the action that should be altered.
        :param directory_path: the path to the directory in which the domain will be copied to.
        :param defect_type: the type of defect to be introduced.
        :return: the generated domain.
        """
        faulty_domain = self.fault_generator.generate_faulty_domain(
            defect_type=defect_type, action_to_alter=action_name)
        (directory_path / self.model_domain_file_name).unlink(missing_ok=True)
        self._export_domain(faulty_domain, directory_path, FAULTY_DOMAIN_PDDL,
                            is_faulty=True, defect_type=defect_type, action_name=action_name)
        return faulty_domain

    def _generate_faulty_domain(self, directory_path: Path) -> Iterator[Tuple[str, LearnerDomain, DefectType]]:
        """Generate d domain with a defect of sort that makes it unsafe.

        Note:
            currently support removing preconditions as defect options.

        :param directory_path: the path to the directory where the domain file will be written to.
        :return: the domain with the randomized defect, the name of the defected action and the type of injected defect.
        """
        for action_name, action_data in self.model_domain.actions.items():
            if len(action_data.numeric_preconditions) > 0:
                self.logger.info(f"Generating faulty domain for action: {action_name} "
                                 f"by altering its numeric preconditions!")
                faulty_domain = self._generate_faulty_domain_based_on_defect_type(
                    action_name, directory_path, DefectType.numeric_precondition_numeric_change)
                yield action_name, faulty_domain, DefectType.numeric_precondition_numeric_change

                self.logger.info(f"Generating faulty domain for action: {action_name} "
                                 f"by removing one numeric precondition!")
                faulty_domain = self._generate_faulty_domain_based_on_defect_type(
                    action_name, directory_path, DefectType.removed_numeric_precondition)
                yield action_name, faulty_domain, DefectType.removed_numeric_precondition

            if len(action_data.numeric_effects) > 0:
                self.logger.info(f"Generating faulty domain for action: {action_name} "
                                 f"by altering its numeric effect!")
                faulty_domain = self._generate_faulty_domain_based_on_defect_type(
                    action_name, directory_path, DefectType.numeric_effect)
                yield action_name, faulty_domain, DefectType.numeric_effect

    def _write_diagnosis(self, all_diagnosis_stats: List[Dict[str, Any]]) -> NoReturn:
        """Write the diagnosis statistics to a file.

        :param all_diagnosis_stats: the collected diagnosis statistics.
        """
        output_path = self.results_dir_path / f"diagnosis_statistics.csv"
        with open(output_path, 'wt', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DIAGNOSIS_COLUMNS)
            writer.writeheader()
            writer.writerows(all_diagnosis_stats)

    def _solve_and_validate(
            self, problems_dir_path: Path, domain_file_path: Path,
            domain_type: str, problems_type: str) -> Tuple[List[Observation], List[Observation], Dict[str, Any]]:
        """Solves a set of problems and validates the solutions.

        :param problems_dir_path: the path to the directory containing the problems.
        :param domain_file_path: the path to the domain file.
        :param domain_type: the type of domain (repaired or faulty).
        :param problems_type: the type of problems (test or train).
        :return: the observations of the test valid plan execution and the observations of the
            faulty execution and the solving statistics.
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
            problems_dir_path, domain_file_path, solving_report, is_repaired_model=domain_type == "repaired")

        counted_fault_detection = Counter(list(faults_detected.values()))
        statistics["ok"] = counted_fault_detection["ok"]
        statistics["not_applicable"] = counted_fault_detection["not_applicable"]
        statistics["state_difference"] = counted_fault_detection["state_difference"]

        return valid_observations, faulty_observations, statistics

    def _clear_domains(self, train_set_dir_path: Path, test_set_dir_path: Path) -> NoReturn:
        """Removes the domains files from the directories.

        :param train_set_dir_path: the train set directory containing the faulty domain file.
        :param test_set_dir_path: the test set directory containing the  domain file.
        :return:
        """
        (test_set_dir_path / self.model_domain_file_name).unlink(missing_ok=True)
        (train_set_dir_path / FAULTY_DOMAIN_PDDL).unlink(missing_ok=True)

    def _repair_action_model(self, defect_type: DefectType, faulty_action_name: str, faulty_domain: LearnerDomain,
                             test_set_dir_path: Path, valid_observations: List[Observation],
                             faulty_observations: List[Observation],
                             repair_algorithm_type: RepairAlgorithmType = RepairAlgorithmType.numeric_sam) -> Path:
        """Repairs the action model of the faulty action by learning the preconditions and effects from valid
            observations.

        :param defect_type: the type of defect that was injected to the faulty action.
        :param faulty_action_name: the name of the faulty action.
        :param faulty_domain: the faulty domain (containing the action with the injected fault).
        :param test_set_dir_path: the path to the test set directory.
        :param valid_observations: the valid observations that were created by executing the plans on an agent.
        :return: the path to the repaired domain file.
        """
        self.logger.debug(f"Found a defected action! action - {faulty_action_name}")
        repaired_domain, report = self.fault_repair.repair_model(faulty_domain, valid_observations, faulty_observations,
                                                         faulty_action_name, repair_algorithm_type)
        used_observations = [*valid_observations, *faulty_observations]
        self.learning_statistics_manager.add_to_action_stats(used_observations, repaired_domain, report)
        self._export_domain(domain=repaired_domain, domain_directory_path=test_set_dir_path,
                            domain_file_name=None, is_faulty=False, defect_type=defect_type,
                            action_name=faulty_action_name, repair_type=repair_algorithm_type.name)
        learned_domain_file_path = test_set_dir_path / self.model_domain_file_name
        return learned_domain_file_path

    def _run_repaired_model_on_test(self, all_diagnosis_stats: List[Dict[str, Any]], faulty_action_name: str,
                                    learned_domain_file_path: Path, test_set_dir_path: Path,
                                    repair_method: RepairAlgorithmType, defect_type: DefectType) -> NoReturn:
        """Tries to solve the test set problems using the repaired domain.

        :param all_diagnosis_stats: the collected diagnosis statistics.
        :param faulty_action_name: the name of the faulty action.
        :param learned_domain_file_path: the path to the learned domain file.
        :param test_set_dir_path: the path to the test set directory.
        :param repair_method: the repair method used to repair the faulty action.
        :param defect_type: the type of defect that was injected to the faulty action.
        """
        self.logger.debug("Solving the test set problems using the learned REPAIRED domain.")
        _, _, safe_test_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=learned_domain_file_path, domain_type="repaired",
            problems_type="test")
        safe_test_stats["action_name"] = faulty_action_name
        safe_test_stats["repair_method"] = repair_method.name
        safe_test_stats["defect_type"] = defect_type.name
        all_diagnosis_stats.append(safe_test_stats)
        self._clear_plans(test_set_dir_path)

    def run_faulty_model_on_test(self, all_diagnosis_stats: List[Dict[str, Any]], faulty_action_name: str,
                                 faulty_domain_path: Path, test_set_dir_path: Path,
                                 repair_method: RepairAlgorithmType, defect_type: DefectType) -> NoReturn:
        """Tries to solve the test set problems using the faulty domain.

        :param all_diagnosis_stats: the collected diagnosis statistics.
        :param faulty_action_name: the name of the faulty action.
        :param faulty_domain_path: the path to the faulty domain file.
        :param test_set_dir_path: the path to the test set directory.
        :param repair_method: the repair method used to repair the faulty action.
        :param defect_type: the type of defect that was injected to the faulty action.
        """
        self.logger.debug("solving the test set problems using the FAULTY domain.")
        _, _, faulty_test_set_stats = self._solve_and_validate(
            problems_dir_path=test_set_dir_path, domain_file_path=faulty_domain_path, domain_type="faulty",
            problems_type="test")
        faulty_test_set_stats["action_name"] = faulty_action_name
        faulty_test_set_stats["repair_method"] = repair_method.name
        faulty_test_set_stats["defect_type"] = defect_type.name
        all_diagnosis_stats.append(faulty_test_set_stats)
        self._clear_plans(test_set_dir_path)

    def run_repair_model_on_train(
            self, all_diagnosis_stats: List[Dict[str, Any]], faulty_action_name: str, repaired_domain_file_path: Path,
            train_set_dir_path: Path, repair_method: RepairAlgorithmType, defect_type: DefectType) -> NoReturn:
        """Tries to solve the train set problems using the repaired domain.

        :param all_diagnosis_stats: the collected diagnosis statistics.
        :param faulty_action_name: the name of the faulty action.
        :param repaired_domain_file_path: the path to the repaired domain file.
        :param train_set_dir_path: the path to the train set directory.
        :param repair_method: the repair method used to repair the faulty action.
        :param defect_type: the type of defect that was injected to the faulty action.
        """
        self.logger.debug("solving the train set problems (again for validation) using the REPAIRED learned domain.")
        _, _, safe_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=repaired_domain_file_path, domain_type="repaired",
            problems_type="train")
        safe_train_stats["action_name"] = faulty_action_name
        safe_train_stats["repair_method"] = repair_method.name
        safe_train_stats["defect_type"] = defect_type.name
        all_diagnosis_stats.append(safe_train_stats)
        self._clear_plans(train_set_dir_path)

    def _write_action_diagnosis_stats(self, all_diagnosis_stats: List[Dict[str, Any]], faulty_action: str) -> NoReturn:
        """Writes the diagnosis statistics for a single action to a file.

        :param all_diagnosis_stats: the list of all diagnosis statistics.
        :param faulty_action: the name of the action that is faulty.
        """
        action_diagnosis_stats = [stats for stats in all_diagnosis_stats if stats["action_name"] == faulty_action]
        action_diagnosis_stats_file_path = self.results_dir_path / f"{faulty_action}_diagnosis_stats.csv"
        with action_diagnosis_stats_file_path.open("w") as action_diagnosis_stats_file:
            writer = csv.DictWriter(action_diagnosis_stats_file, fieldnames=DIAGNOSIS_COLUMNS)
            writer.writeheader()
            writer.writerows(action_diagnosis_stats)

        self.learning_statistics_manager.export_action_learning_statistics(fold_number=self._action_index)

    def _run_single_fault_detection(
            self, all_diagnosis_stats: List[Dict[str, Any]], faulty_action: str, faulty_domain: LearnerDomain,
            test_set_dir_path: Path, train_set_dir_path: Path, defect_type: DefectType,
            repair_algorithm_type: RepairAlgorithmType = RepairAlgorithmType.numeric_sam) -> NoReturn:
        """Runs a single fault detection experiment, i.e. tries to detect a single fault in an action.

        :param all_diagnosis_stats: the list of all diagnosis statistics.
        :param faulty_action: the name of the action that is faulty.
        :param faulty_domain: the domain that contains the faulty action.
        :param test_set_dir_path: the path to the test set directory.
        :param train_set_dir_path: the path to the train set directory.
        :param defect_type: the type of defect that is injected.
        """
        self.logger.debug("Solving the train set problems using the faulty domain.")
        faulty_domain_path = train_set_dir_path / FAULTY_DOMAIN_PDDL
        valid_observations, faulty_observations, faulty_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=faulty_domain_path, domain_type="faulty",
            problems_type="train")

        faulty_train_stats["action_name"] = faulty_action
        faulty_train_stats["repair_method"] = repair_algorithm_type.name
        faulty_train_stats["defect_type"] = defect_type.name
        all_diagnosis_stats.append(faulty_train_stats)
        self._clear_plans(train_set_dir_path)

        if len(valid_observations) == 0:
            self.logger.warning("No valid observations found in the train set.")
            return

        faulty_action_name = valid_observations[0].components[0].grounded_action_call.name
        learned_domain_file_path = self._repair_action_model(
            defect_type, faulty_action_name, faulty_domain,
            test_set_dir_path, valid_observations, faulty_observations, repair_algorithm_type)

        self._run_repaired_model_on_test(all_diagnosis_stats, faulty_action_name, learned_domain_file_path,
                                         test_set_dir_path, repair_algorithm_type, defect_type)
        self.run_faulty_model_on_test(all_diagnosis_stats, faulty_action, faulty_domain_path, test_set_dir_path,
                                      repair_algorithm_type, defect_type)
        self.run_repair_model_on_train(all_diagnosis_stats, faulty_action_name, learned_domain_file_path,
                                       train_set_dir_path, repair_algorithm_type, defect_type)

    def evaluate_fault_diagnosis(self, train_set_dir_path: Path, test_set_dir_path: Path) -> NoReturn:
        """Conducts the experiments to evaluate the efficiency of the 'repair' property.

        :param train_set_dir_path: the path to the train set folder.
        :param test_set_dir_path: the path to the test set folder.
        """
        all_diagnosis_stats = []
        self.logger.info("Starting the fault diagnosis simulation!")
        for faulty_action, faulty_domain, defect_type in self._generate_faulty_domain(train_set_dir_path):
            for repair_method in RepairAlgorithmType:
                self.logger.info(f"Trying to repair a faulty model using the algorithm '{repair_method.name}'.")
                try:
                    self.learning_statistics_manager.learning_algorithm = REPAIR_TO_LEARNING_ALGORITHM[repair_method]
                    self._run_single_fault_detection(
                        all_diagnosis_stats, faulty_action, faulty_domain, test_set_dir_path, train_set_dir_path,
                        defect_type, repair_method)
                    self._write_action_diagnosis_stats(all_diagnosis_stats, faulty_action)

                except ValueError:
                    self.logger.debug("The injected fault did not cause any problems in the train set.")
                    continue

            self.logger.info("Finished running fault diagnosis on all the available approaches.")
            self._clear_domains(train_set_dir_path, test_set_dir_path)
            self._action_index += 1
            self.learning_statistics_manager.clear_statistics()

        self.logger.debug("Finished the fault diagnosis simulation!")
        self._write_diagnosis(all_diagnosis_stats)
        self.learning_statistics_manager.write_complete_joint_statistics()

    def run_fault_diagnosis(self) -> NoReturn:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for train_dir_path, test_dir_path in self.k_fold.create_k_fold():
            self.evaluate_fault_diagnosis(train_dir_path, test_dir_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the fault diagnosis simulation on a selected domain.")
    parser.add_argument("--work_dir_path", required=True,
                        help="The path to the directory that containing the defected domain.")
    parser.add_argument("--fluents_map_path", required=True,
                        help="The path to the fluents map file (mapping of the numeric fluents preconditions).")
    parser.add_argument("--original_domain_file_name", required=True, help="the name of the original domain file.")
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    args = parse_arguments()
    diagnoser = ModelFaultDiagnosis(
        work_dir_path=Path(args.work_dir_path),
        fluents_map_path=Path(args.fluents_map_path),
        original_domain_file_name=args.original_domain_file_name)
    diagnoser.run_fault_diagnosis()

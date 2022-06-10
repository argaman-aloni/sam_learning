"""Conducts the experiments to test how efficiently the learning algorithm can diagnose faults."""
import csv
import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import NoReturn, List, Dict, Any, Optional

from anytree import AnyNode
from pddl_plus_parser.exporters import ENHSPParser, TrajectoryExporter
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, Domain, Action, NumericalExpressionTree, Operator

from experiments.experiments_trajectories_creator import ExperimentTrajectoriesCreator
from experiments.k_fold_split import KFoldSplit
from sam_learning.core import LearnerDomain, ConditionType
from sam_learning.learners import NumericSAMLearner
from solvers import ENHSPSolver
from utilities import SolverType, SolutionOutputTypes

FAULTY_DOMAIN_PDDL = "faulty_domain.pddl"

random.seed(42)
DIAGNOSIS_COLUMNS = ["domain_type", "problems_type", "ok", "no_solution", "timeout", "not_applicable"]
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
        with open(fluents_map_path, "rt") as json_file:
            self.fluents_map = json.load(json_file)

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

    @staticmethod
    def _delete_trajectories(train_set_dir_path: Path) -> NoReturn:
        """Removes the trajectories created by the original domain prior to the added defect.

        :param train_set_dir_path: the path to the train set directory containing the problems to solve.
        """
        for trajectory_file_path in train_set_dir_path.glob("*.trajectory"):
            os.remove(trajectory_file_path)

    def _clear_plans(self, working_directory: Path) -> NoReturn:
        """Clears the plan filed from the directory.

        :param working_directory: the path to the directory containing the plans.
        """
        for solver_output_path in working_directory.glob("*.solution"):
            os.remove(solver_output_path)

        self._delete_trajectories(working_directory)

    def _alter_action_preconditions(self, faulty_action: Action) -> NoReturn:
        """Alters the action's preconditions so that it will contain a defect.

        :param faulty_action: the action to alter.
        """
        self.logger.info(f"Altering the action - {faulty_action.name} preconditions!")
        precondition_to_alter: NumericalExpressionTree = random.choice(list(faulty_action.numeric_preconditions))
        self.logger.debug(f"Precondition to alter: {precondition_to_alter.to_pddl()}")
        if precondition_to_alter.root.value == ">=" or precondition_to_alter.root.value == ">":
            precondition_to_alter.root.value = "<="
            self.logger.debug(f"Altered precondition: {precondition_to_alter.to_pddl()}")
            return

        if precondition_to_alter.root.value == "<=" or precondition_to_alter.root.value == "<":
            precondition_to_alter.root.value = ">="
            self.logger.debug(f"Altered precondition: {precondition_to_alter.to_pddl()}")
            return

    def _alter_action_effects(self, faulty_action: Action) -> NoReturn:
        """Alter the action's effects so that it will contain a defect.

        :param faulty_action: the action to alter.
        """
        self.logger.info(f"Altering the action - {faulty_action.name} effects!")
        effect_to_alter: NumericalExpressionTree = random.choice(list(faulty_action.numeric_effects))
        node = effect_to_alter.root
        while not node.is_leaf:
            node = node.children[1]

        if isinstance(node.value, float):
            new_value = node.value + 5
            node.value = new_value
            node.id = str(new_value)
        else:
            function_name = node.id
            function_value = node.value
            new_add_node = AnyNode(id="+", value="+", children=[
                AnyNode(id=function_name, value=function_value),
                AnyNode(id=str(5), value=5)
            ])
            node.parent.children[1] = new_add_node

        self.logger.debug(f"Altered effect: {effect_to_alter.to_pddl()}")

    @staticmethod
    def _select_action_to_alter(altered_domain: Domain) -> Action:
        """Selects an action to alter using random selection on the domain's actions.

        :param altered_domain: the domain to select the action from.
        :return: the selected action.
        """
        actions_to_choose = list(altered_domain.actions.values())
        random.shuffle(actions_to_choose)
        faulty_action = random.choice(actions_to_choose)

        while len(faulty_action.numeric_preconditions) == 0 and len(faulty_action.numeric_effects) == 0:
            faulty_action = random.choice(actions_to_choose)

        return faulty_action

    def _set_faulty_domain_and_defected_action(self) -> LearnerDomain:
        """Sets the domain fields and alters the domain's actions so that it will contain a defect.

        :return: a serializable domain object.
        """
        altered_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()
        faulty_action = self._select_action_to_alter(altered_domain)
        self.logger.debug(f"Altering the action - {faulty_action.name}!")
        if len(faulty_action.numeric_preconditions) > 0:
            self._alter_action_preconditions(faulty_action)

        else:  # len(faulty_action.numeric_effects) > 0:
            self._alter_action_effects(faulty_action)

        faulty_domain = LearnerDomain(altered_domain)
        for original_action, faulty_domain_action in zip(altered_domain.actions.values(),
                                                         faulty_domain.actions.values()):
            faulty_domain_action.positive_preconditions = original_action.positive_preconditions
            faulty_domain_action.inequality_preconditions = original_action.inequality_preconditions
            numeric_preconditions = [precond.to_pddl() for precond in original_action.numeric_preconditions]
            faulty_domain_action.numeric_preconditions = (numeric_preconditions, ConditionType.injunctive)
            faulty_domain_action.add_effects = original_action.add_effects
            faulty_domain_action.delete_effects = original_action.delete_effects
            numeric_effects = [effect.to_pddl() for effect in original_action.numeric_effects]
            faulty_domain_action.numeric_effects = numeric_effects

        return faulty_domain

    def _generate_faulty_domain(self, directory_path: Path) -> Domain:
        """Generate d domain with a defect of sort that makes it unsafe.

        Note:
            currently support removing preconditions as defect options.

        :param directory_path: the path to the directory where the domain file will be written to.
        :return: the domain with the randomized defect.
        """
        faulty_domain = self._set_faulty_domain_and_defected_action()

        self.logger.debug(faulty_domain.to_pddl())
        os.remove(directory_path / self.model_domain_file_name)
        faulty_domain_path = directory_path / FAULTY_DOMAIN_PDDL
        self._export_domain(faulty_domain, directory_path, FAULTY_DOMAIN_PDDL)
        self._delete_trajectories(directory_path)

        return DomainParser(domain_path=faulty_domain_path, partial_parsing=False).parse_domain()

    def _extract_applicable_trajectories(self, plans_dir_path: Path) -> List[Observation]:
        """Extracts only the trajectories which are created from applicable plans according to the original domain.

        :param plans_dir_path: the directory containing the plans where part of them might be inapplicable.
        :return: the observations created from the applicable plans.
        """
        copied_file_path = shutil.copy(self.model_domain_file_path, plans_dir_path)
        trajectory_creator = ExperimentTrajectoriesCreator(self.model_domain_file_name, plans_dir_path)
        trajectory_creator.fix_solution_files(SolverType.enhsp)
        trajectory_creator.create_domain_trajectories()

        trajectories = []
        for trajectory_path in plans_dir_path.glob("*.trajectory"):
            problem_path = plans_dir_path / f"{trajectory_path.stem}.pddl"
            problem = ProblemParser(problem_path, self.model_domain).parse_problem()
            trajectories.append(TrajectoryParser(self.model_domain, problem).parse_trajectory(trajectory_path))

        os.remove(copied_file_path)
        return trajectories

    def _learn_safe_action_model(self, faulty_domain: Domain, test_set_dir_path: Path,
                                 train_set_dir_path: Path) -> NoReturn:
        """Runs the safe action model learning algorithm on the applicable trajectories.

        :param faulty_domain: the domain that contains a defect.
        :param test_set_dir_path: the path to the test set directory.
        :param train_set_dir_path: the path to the train set directory.
        """
        train_trajectories = self._extract_applicable_trajectories(train_set_dir_path)
        learner = NumericSAMLearner(partial_domain=faulty_domain, preconditions_fluent_map=self.fluents_map)
        learned_model, _ = learner.learn_action_model(train_trajectories)
        self._export_domain(learned_model, test_set_dir_path)

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
            if solving_status != SolutionOutputTypes.ok.name:
                continue

            problem_file_path = problems_dir_path / f"{problem_name}.pddl"
            solution_file_path = problems_dir_path / f"{problem_name}.solution"
            sequence = ENHSPParser().parse_plan_content(solution_file_path)
            problem = ProblemParser(problem_file_path, self.model_domain).parse_problem()
            self.logger.debug("Validating the applicability of the plan according to the original domain.")
            try:
                TrajectoryExporter(self.model_domain).parse_plan(problem=problem, action_sequence=sequence)

            except ValueError:
                self.logger.debug("The plan is inapplicable!")
                statistics[SolutionOutputTypes.not_applicable.name] += 1
                statistics[SolutionOutputTypes.ok.name] -= 1
                os.remove(solution_file_path)

        return statistics

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
        faulty_train_stats = self._solve_and_validate(
            problems_dir_path=train_set_dir_path, domain_file_path=faulty_domain_path, domain_type="faulty",
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

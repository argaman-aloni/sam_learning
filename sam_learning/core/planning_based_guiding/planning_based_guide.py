"""A guide to the online learning process that is based on planning."""
import random
from pathlib import Path

from pddl_plus_parser.models import Domain, PDDLObject, ActionCall, State, Operator, Precondition
from typing import Dict, List

from sam_learning.core import LearnerDomain
from sam_learning.core.planning_based_guiding.guiding_problem_generator import generate_instance
from solvers import MetricFFSolver

TMP_PROBLEM_PDDL = "tmp_problem.pddl"

TEMP_DOMAIN_FILE_PATH = "tmp_domain.pddl"


class PlanningBasedGuide:

    def __init__(self, working_directory_path: Path):
        self._workdir_path = working_directory_path
        self._domain = Domain()
        self._planner = MetricFFSolver()

    def _export_domain(self) -> None:
        """Exports the domain to a file."""
        exportable_domain = LearnerDomain(self._domain)
        for action in self._domain.actions.values():
            exportable_domain.actions[action.name].preconditions = action.preconditions
            exportable_domain.actions[action.name].discrete_effects = action.discrete_effects
            exportable_domain.actions[action.name].numeric_effects = action.numeric_effects

        with open(self._workdir_path / TEMP_DOMAIN_FILE_PATH, "w") as domain_file:
            domain_file.write(exportable_domain.to_pddl())

    def _export_problem(
            self, problem_objects: Dict[str, PDDLObject], current_state: State, goal_predicates: List[str]) -> None:
        """Exports the problem to a file."""
        problem_instance = generate_instance(domain_name=self._domain.name,
                                             problem_object=problem_objects,
                                             current_state=current_state,
                                             goal_predicates=goal_predicates)
        with open(self._workdir_path / TMP_PROBLEM_PDDL, "w") as problem_file:
            problem_file.write(problem_instance)

    def set_domain(self, domain: Domain) -> None:
        """Sets the domain of the guide.

        :param domain: the domain of the guide.
        """
        self._domain = domain

    def guide_learning(
            self, problem_objects: Dict[str, PDDLObject], action_to_learn: str,
            grounded_actions: List[ActionCall]) -> None:
        """Guides the learning process."""
        pass



"""Module that repairs faulty domains by fixing the action that contains the defect."""
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, NoReturn

from pddl_plus_parser.exporters import ENHSPParser
from pddl_plus_parser.exporters.numeric_trajectory_exporter import parse_action_call
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import State, Observation, Operator, ActionCall, Domain

from fault_detection.defect_types import RepairAlgorithmType
from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner
from sam_learning.learners.oblique_tree_model_learner import ObliqueTreeModelLearner
from validators import VALID_PLAN, GOAL_NOT_REACHED, INAPPLICABLE_PLAN
from validators.validator_script_data import run_validate_script

FAULTY_ACTION_LOCATOR_REGEX = re.compile(r"Plan failed because of unsatisfied precondition in:\n\((\w+) [\w+ ]*\)",
                                         flags=re.MULTILINE)


class FaultRepair:
    """Class that detects and repairs faults in planning domains."""

    working_directory_path: Path
    model_domain_file_name: str
    model_domain_file_path: Path
    model_domain: Domain
    fluents_map: Dict[str, List[str]]
    diagnosis_statistics: List[Dict[str, str]]
    logger: logging.Logger

    def __init__(self, working_directory_path: Path, model_domain_file_name: str, fluents_map_path: Path):
        self.working_directory_path = working_directory_path
        self.model_domain_file_name = model_domain_file_name
        self.model_domain_file_path = self.working_directory_path / self.model_domain_file_name
        self.model_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()
        with open(fluents_map_path, "rt") as json_file:
            self.fluents_map = json.load(json_file)

        self.diagnosis_statistics = []
        self.logger = logging.getLogger(__name__)

    def _validate_applied_action(self, faulty_action_name: str, valid_next_state: State,
                                 faulty_next_state: State) -> bool:
        """Validates whether the applied action resulted in the same state as when applying it on the agent.

        :param faulty_action_name: the name of the action that contains a defect.
        :param valid_next_state: the state received after applying the action on the agent.
        :param faulty_next_state: the state received after applying the faulty action.
        :return: whether the states are the same.
        """
        for predicate_name, grounded_predicates in valid_next_state.state_predicates.items():
            if predicate_name not in faulty_next_state.state_predicates:
                self.logger.debug(f"The predicate {predicate_name} is missing in the faulty observation!")
                return False

            if set(predicate.untyped_representation for predicate in grounded_predicates) != \
                    set(predicate.untyped_representation for predicate in faulty_next_state.state_predicates[
                        predicate_name]):
                self.logger.debug(
                    f"The action {faulty_action_name} resulted in different state in the executing agent!"
                    f"\nValid next state: {valid_next_state.serialize()}"
                    f"\nPossibly faulty next state: {faulty_next_state.serialize()}")
                return False

        for fluent_name, fluent in valid_next_state.state_fluents.items():
            if fluent.state_representation != faulty_next_state.state_fluents[fluent_name].state_representation:
                self.logger.debug(
                    f"The action {faulty_action_name} resulted in different state in the executing agent!"
                    f"\nExpected - {fluent.state_representation} and received - "
                    f"{faulty_next_state.state_fluents[fluent_name].state_representation}")
                return False

        return True

    def _generate_grounded_operators(self, action_name: str,
                                     faulty_domain: Domain, parameters: List[str]) -> Tuple[Operator, Operator]:
        """Generates the valid and faulty grounded operators from the action descriptor.

        :param action_name: the name of the action that contains a defect.
        :param faulty_domain: the domain the contains a defect.
        :param parameters: the parameters with which the action was executed.
        :return: the faulty and valid grounded operators.
        """
        valid_operator = Operator(action=self.model_domain.actions[action_name],
                                  domain=self.model_domain, grounded_action_call=parameters)
        possibly_faulty_operator = Operator(action=faulty_domain.actions[action_name],
                                            domain=faulty_domain, grounded_action_call=parameters)
        return possibly_faulty_operator, valid_operator

    def _is_plan_applicable(self, problem_file_path: Path, solution_file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validates that the solution file contains a valid plan.

        :param problem_file_path: the path to the problem file.
        :param solution_file_path: the path to the solution file.

        :return: whether the plan is valid and the name of the faulty action if it is not.
        """
        validation_file_path = \
            run_validate_script(domain_file_path=self.model_domain_file_path,
                                problem_file_path=problem_file_path, solution_file_path=solution_file_path)

        with open(validation_file_path, "r") as validation_file:
            validation_file_content = validation_file.read()
            if VALID_PLAN in validation_file_content or GOAL_NOT_REACHED in validation_file_content:
                return True, None

            if INAPPLICABLE_PLAN in validation_file_content:
                match = FAULTY_ACTION_LOCATOR_REGEX.search(validation_file_content)
                faulty_action_name = match.group(1)
                return False, faulty_action_name

    def _observe_single_plan(
            self, faulty_domain: Domain, problem_file_path: Path,
            solution_file_path: Path) -> Tuple[Optional[Observation], Optional[Observation], Optional[str]]:
        """Observes a single plan and determines whether there are faults in it and where they might be.

        :param plan_sequence: The plan sequence to observe.
        :param faulty_domain: the domain with the faulty action.
        :param problem: the problem that was solved using the plan sequence.
        :return: the faulty and the valid observation and the name of the faulty action.
        """

        plan_applicable, inapplicable_action = self._is_plan_applicable(problem_file_path, solution_file_path)
        if not plan_applicable:
            return None, None, inapplicable_action

        plan_sequence = ENHSPParser().parse_plan_content(solution_file_path)
        problem = ProblemParser(problem_file_path, self.model_domain).parse_problem()
        valid_observation = Observation()
        faulty_observation = Observation()
        faulty_action_name = None
        valid_previous_state = State(predicates=problem.initial_state_predicates,
                                     fluents=problem.initial_state_fluents, is_init=True)
        faulty_previous_state = valid_previous_state

        for grounded_action in plan_sequence:
            self.logger.info(f"The executed action: {grounded_action}")
            descriptor = parse_action_call(grounded_action)
            action_name = descriptor.name
            parameters = descriptor.parameters
            faulty_operator, valid_operator = self._generate_grounded_operators(action_name, faulty_domain, parameters)

            valid_next_state = valid_operator.apply(valid_previous_state)
            faulty_next_state = faulty_operator.apply(faulty_previous_state)
            valid_observation.add_component(valid_previous_state, ActionCall(action_name, parameters), valid_next_state)
            is_state_identical = self._validate_applied_action(action_name, valid_next_state, faulty_next_state)
            if not is_state_identical:
                faulty_action_name = faulty_action_name or action_name
                faulty_observation.add_component(faulty_previous_state, ActionCall(action_name, parameters),
                                                 faulty_next_state)

            valid_previous_state = valid_next_state
            faulty_previous_state = faulty_next_state

        return valid_observation, faulty_observation, faulty_action_name

    @staticmethod
    def _filter_redundant_observations(
            faulty_action_name: str, faulty_action_observations: List[Observation],
            valid_action_observations: List[Observation]) -> NoReturn:
        """Filters out the observations that do not belong to the faulty action.

        :param faulty_action_name: the action that contains a defect.
        :param faulty_action_observations: the observations obtained from applying the faulty action.
        :param valid_action_observations: the observations obtained from executing the actions on the agent.
        """
        for valid_observation, faulty_observation in zip(valid_action_observations, faulty_action_observations):
            relevant_valid_components = [component for component in valid_observation.components
                                         if component.grounded_action_call.name == faulty_action_name]
            relevant_faulty_components = [component for component in faulty_observation.components if
                                          component.grounded_action_call.name == faulty_action_name]
            valid_observation.components = relevant_valid_components
            faulty_observation.components = relevant_faulty_components

    def execute_plans_on_agent(
            self, plans_dir_path: Path, faulty_domain_path: Path,
            is_repaired_model: bool = False) -> Tuple[List[Observation], List[Observation], Dict[str, str]]:
        """Executes the plans on the agent and returns the learned information about the possible faults and the
        execution status.

        :param faulty_domain_path: the path to the domain file that might be either faulty or the fixed file.
        :param plans_dir_path: the path to the directory containing the plans.
        :param is_repaired_model: whether the model has been repaired or not.
        :return: the statistics about the execution of the plans and the observations.
        """
        faulty_action_observations = []
        valid_action_observations = []
        observed_plans = {}
        faulty_action_name = None
        faulty_domain = DomainParser(domain_path=faulty_domain_path).parse_domain()
        for solution_file_path in plans_dir_path.glob("*.solution"):
            problem_file_path = plans_dir_path / f"{solution_file_path.stem}.pddl"
            valid_observation, faulty_observation, faulty_action = self._observe_single_plan(
                faulty_domain, problem_file_path, solution_file_path)
            if faulty_action is None:
                self.logger.debug(f"The plan {solution_file_path.stem} was validated and is applicable!")
                observed_plans[solution_file_path.stem] = "ok"

            if valid_observation is None:
                faulty_action_name = faulty_action
                self.logger.debug(f"The plan {solution_file_path.stem} is not valid!")
                observed_plans[solution_file_path.stem] = "not_applicable"

            if faulty_action is not None:
                faulty_action_name = faulty_action
                self.logger.debug(f"Detected a faulty action in plan {solution_file_path.stem}! "
                                  f"The action {faulty_action} is faulty!")
                observed_plans[solution_file_path.stem] = "state_difference"

            if valid_observation is not None:
                valid_action_observations.append(valid_observation)
                faulty_action_observations.append(faulty_observation)

        if faulty_action_name is None and not is_repaired_model:
            self.logger.warning("No fault detected!")
            raise ValueError("No fault detected!")

        self._filter_redundant_observations(faulty_action_name, faulty_action_observations, valid_action_observations)

        return valid_action_observations, faulty_action_observations, observed_plans

    def repair_model(
            self, faulty_domain: LearnerDomain, valid_observations: List[Observation],
            faulty_observations: List[Observation] = None, faulty_action_name: str = None,
            repair_algorithm_type: RepairAlgorithmType = RepairAlgorithmType.numeric_sam) -> LearnerDomain:
        """Repairs an action model that contains a defect by learning valid observations.

        :param faulty_domain: the domain that contains a defected action.
        :param valid_observations: the valid observations obtained from executing the actions on the agent.
        :param faulty_observations: the observations containing the defect in them.
        :param faulty_action_name: the name of the action that contains a defect.
        :param repair_algorithm_type: the algorithm that will be used to repair the model.
        :return: the action model with the defect repaired.
        """
        partial_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()
        repaired_action = None
        if repair_algorithm_type == RepairAlgorithmType.numeric_sam:
            learner = NumericSAMLearner(partial_domain=partial_domain, preconditions_fluent_map=self.fluents_map)
            learned_model, _ = learner.learn_action_model(valid_observations)
            repaired_action = learned_model.actions[faulty_action_name]

        elif repair_algorithm_type == RepairAlgorithmType.oblique_tree:
            learner = ObliqueTreeModelLearner(partial_domain=partial_domain,
                                              preconditions_fluent_map=self.fluents_map, polynomial_degree=0)
            learned_model, _ = learner.learn_unsafe_action_model(valid_observations, faulty_observations)
            repaired_action = learned_model.actions[faulty_action_name]

        faulty_domain.actions[faulty_action_name] = repaired_action
        return faulty_domain

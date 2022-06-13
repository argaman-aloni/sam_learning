"""Module that repairs faulty domains by fixing the action that contains the defect."""
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, NoReturn

from pddl_plus_parser.exporters import ENHSPParser
from pddl_plus_parser.exporters.numeric_trajectory_exporter import parse_action_call, ActionDescriptor
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import State, Observation, Operator, ActionCall, Domain, Problem

from sam_learning.core import LearnerDomain
from sam_learning.learners import NumericSAMLearner

DIAGNOSIS_ANALYSIS_COLUMNS = ["problem_name", "plan_applicable", "plan_differ_from_reference",
                              "faulty_action", "fault_location"]


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

    def _generate_grounded_operators(self, action_descriptor: ActionDescriptor,
                                     faulty_domain: Domain, parameters: List[str]) -> Tuple[Operator, Operator]:
        """Generates the valid and faulty grounded operators from the action descriptor.

        :param action_descriptor: the description of the currently executed action.
        :param faulty_domain: the domain the contains a defect.
        :param parameters: the parameters with which the action was executed.
        :return: the faulty and valid grounded operators.
        """
        valid_operator = Operator(action=self.model_domain.actions[action_descriptor.name],
                                  domain=self.model_domain, grounded_action_call=parameters)
        possibly_faulty_operator = Operator(action=faulty_domain.actions[action_descriptor.name],
                                            domain=faulty_domain, grounded_action_call=parameters)
        return possibly_faulty_operator, valid_operator

    def _observe_single_plan(
            self, plan_sequence: List[str], faulty_domain: Domain,
            problem: Problem) -> Tuple[Optional[Observation], Optional[Observation], Optional[str]]:
        """Observes a single plan and determines whether there are faults in it and where they might be.

        :param plan_sequence: The plan sequence to observe.
        :param faulty_domain: the domain with the faulty action.
        :param problem: the problem that was solved using the plan sequence.
        :return:
        """
        valid_observation = Observation()
        faulty_observation = Observation()
        faulty_action_name = None
        valid_previous_state = State(predicates=problem.initial_state_predicates,
                                     fluents=problem.initial_state_fluents, is_init=True)
        possibly_invalid_previous_state = valid_previous_state

        for grounded_action in plan_sequence:
            self.logger.info(f"The executed action: {grounded_action}")
            action_descriptor = parse_action_call(grounded_action)
            parameters = action_descriptor.parameters
            possibly_faulty_operator, valid_operator = self._generate_grounded_operators(
                action_descriptor, faulty_domain, parameters)

            if not valid_operator.is_applicable(valid_previous_state):
                self.logger.debug(f"The action {grounded_action} is not applicable according to the executing agent!")
                return None, None, action_descriptor.name

            valid_next_state = valid_operator.apply(valid_previous_state)
            possibly_faulty_next_state = possibly_faulty_operator.apply(possibly_invalid_previous_state)
            if not self._validate_applied_action(action_descriptor.name, valid_next_state, possibly_faulty_next_state):
                faulty_action_name = action_descriptor.name if faulty_action_name is None else faulty_action_name
                faulty_observation.add_component(
                    possibly_invalid_previous_state, ActionCall(action_descriptor.name, action_descriptor.parameters),
                    possibly_faulty_next_state)

            valid_observation.add_component(
                valid_previous_state, ActionCall(action_descriptor.name, action_descriptor.parameters),
                valid_next_state)

            valid_previous_state = valid_next_state
            possibly_invalid_previous_state = possibly_faulty_next_state

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
            self, plans_dir_path: Path,
            faulty_domain_path: Path) -> Tuple[List[Observation], List[Observation], Dict[str, str]]:
        """

        :param faulty_domain_path:
        :param plans_dir_path:
        :return:
        """
        faulty_action_observations = []
        valid_action_observations = []
        observed_plans = {}
        faulty_action_name = None
        faulty_domain = DomainParser(domain_path=faulty_domain_path).parse_domain()
        for solution_file_path in plans_dir_path.glob("*.solution"):
            problem_file_path = plans_dir_path / f"{solution_file_path.stem}.pddl"
            plan_sequence = ENHSPParser().parse_plan_content(solution_file_path)
            problem = ProblemParser(problem_file_path, self.model_domain).parse_problem()
            valid_observation, faulty_observation, possibly_faulty_action = self._observe_single_plan(plan_sequence,
                                                                                                      faulty_domain,
                                                                                                      problem)
            if possibly_faulty_action is None:
                self.logger.debug(f"The plan {solution_file_path.stem} was validated and is applicable!")
                observed_plans[solution_file_path.stem] = "ok"

            if valid_observation is None:
                faulty_action_name = possibly_faulty_action
                self.logger.debug(f"The plan {solution_file_path.stem} is not valid!")
                observed_plans[solution_file_path.stem] = "not_applicable"

            if possibly_faulty_action is not None:
                faulty_action_name = possibly_faulty_action
                self.logger.debug(f"Detected a faulty action in plan {solution_file_path.stem}! "
                                  f"The action {possibly_faulty_action} is faulty!")
                observed_plans[solution_file_path.stem] = "state_difference"

            if valid_observation is not None:
                valid_action_observations.append(valid_observation)
                faulty_action_observations.append(faulty_observation)

        if faulty_action_name is None:
            self.logger.warning("No fault detected!")
            raise ValueError("No fault detected!")

        self._filter_redundant_observations(faulty_action_name, faulty_action_observations, valid_action_observations)

        return valid_action_observations, faulty_action_observations, observed_plans

    def repair_model(self, faulty_domain: LearnerDomain, valid_observations: List[Observation],
                     faulty_action_name: str) -> LearnerDomain:
        """Repairds an action model that contains a defect by learning valid observations.

        :param faulty_domain: the domain that contains a defected action.
        :param valid_observations: the valid observations obtained from executing the actions on the agent.
        :param faulty_action_name: the name of the action that contains a defect.
        :return: the action model with the defect repaired.
        """
        partial_domain = DomainParser(domain_path=self.model_domain_file_path).parse_domain()
        learner = NumericSAMLearner(partial_domain=partial_domain, preconditions_fluent_map=self.fluents_map)
        learned_model, _ = learner.learn_action_model(valid_observations)
        repaired_action = learned_model.actions[faulty_action_name]
        faulty_domain.actions[faulty_action_name] = repaired_action

        return faulty_domain

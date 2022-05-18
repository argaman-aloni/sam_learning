"""Extension to SAM Learning that can learn numeric state variables."""

from typing import List, NoReturn, Dict, Tuple, Optional

from pddl_plus_parser.models import Observation, ActionCall, State, Domain

from sam_learning.core import LearnerDomain, NumericFluentStateStorage, NumericFunctionMatcher, NotSafeActionError
from .sam_learning import SAMLearner


class NumericSAMLearner(SAMLearner):
    """The Extension of SAM that is able to learn numeric state variables."""

    storage: Dict[str, NumericFluentStateStorage]
    function_matcher: NumericFunctionMatcher
    preconditions_fluent_map: Dict[str, List[str]]

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain)
        self.storage = {}
        self.function_matcher = NumericFunctionMatcher(partial_domain)
        self.preconditions_fluent_map = preconditions_fluent_map

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Adds a new action to the learned domain.

        :param grounded_action: the grounded action that was executed according to the observation.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        super().add_new_action(grounded_action, previous_state, next_state)
        self.logger.debug(f"Creating the new storage for the action - {grounded_action.name}.")
        previous_state_lifted_matches = self.function_matcher.match_state_functions(
            grounded_action, previous_state.state_fluents)
        next_state_lifted_matches = self.function_matcher.match_state_functions(
            grounded_action, next_state.state_fluents)
        self.storage[grounded_action.name] = NumericFluentStateStorage(grounded_action.name)
        self.storage[grounded_action.name].add_to_previous_state_storage(previous_state_lifted_matches)
        self.storage[grounded_action.name].add_to_next_state_storage(next_state_lifted_matches)
        self.logger.debug(f"Done creating the numeric state variable storage for the action - {grounded_action.name}")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Updates the action's data according to the new input observed triplet.

        :param grounded_action: the grounded action that was observed.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        super().update_action(grounded_action, previous_state, next_state)
        self.logger.debug(
            f"Adding the numeric state variables to the numeric storage of action - {action_name}.")
        previous_state_lifted_matches = self.function_matcher.match_state_functions(
            grounded_action, previous_state.state_fluents)
        next_state_lifted_matches = self.function_matcher.match_state_functions(
            grounded_action, next_state.state_fluents)
        self.storage[action_name].add_to_previous_state_storage(previous_state_lifted_matches)
        self.storage[action_name].add_to_next_state_storage(next_state_lifted_matches)
        self.logger.debug(f"Done updating the numeric state variable storage for the action - {grounded_action.name}")

    def learn_action_model(self, observations: List[Observation], is_baseline: bool = False) -> Tuple[
        LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input observations.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :param is_baseline: whether the current learning is a baseline learning process.
        :return: a domain containing the actions that were learned and the metadata about the learning.
        """
        self.logger.info("Starting to learn the action model!")
        allowed_actions = {}
        learning_metadata = {}
        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        for action_name, action in self.partial_domain.actions.items():
            if action_name not in self.storage:
                self.logger.debug(f"The action - {action_name} has not been observed in the trajectories!")
                continue

            self.storage[action_name].filter_out_inconsistent_state_variables()
            try:
                if self.preconditions_fluent_map is not None and len(
                        self.preconditions_fluent_map[action_name]) > 0 and not is_baseline:
                    relevant_fluents = self.preconditions_fluent_map[action_name] if \
                        self.preconditions_fluent_map is not None else None
                    action.numeric_preconditions = self.storage[action_name].construct_safe_linear_inequalities(
                        relevant_fluents)

                action.numeric_effects = self.storage[action_name].construct_assignment_equations()
                allowed_actions[action_name] = action
                learning_metadata[action_name] = "OK"

            except NotSafeActionError as e:
                self.logger.debug(f"The action -{e.action_name} is not safe for execution, reason - {e.reason}")
                learning_metadata[action_name] = e.solution_type.name

        self.partial_domain.actions = allowed_actions
        return self.partial_domain, learning_metadata

"""Class to manage the novelty calculations for the online learning setting."""
import logging
from typing import Tuple, Dict, List, Set

from pddl_plus_parser.models import ActionCall, State, PDDLFunction, Domain, Operator, Predicate, GroundedPredicate

from sam_learning.core.environment_snapshot import EnvironmentSnapshot
from sam_learning.core.predicates_matcher import PredicatesMatcher
from sam_learning.core.numeric_learning.numeric_function_matcher import NumericFunctionMatcher

EXECUTION_DB_COLUMNS = ["lifted_action", "state_predicates", "functions_and_values", "execution_result"]
SUCCESS_RESULT = 1
FAIL_RESULT = -1
DEFAULT_EXPLORATION_CONSTANT = 1000


def _try_applying_action_in_state(
        snapshot: EnvironmentSnapshot, action: ActionCall, state: State,
        domain: Domain) -> Tuple[Dict[str, PDDLFunction], Set[GroundedPredicate]]:
    """Tries to apply the action according to the currently learned model.

    :param action: the action that is being applied.
    :param state: the state that we are trying to apply the action in.
    :return: the state after the action was applied - the information returned is grounded to avoid conflict of
        bounded lifted representations.
    """
    operator = Operator(action=domain.actions[action.name], domain=domain,
                        grounded_action_call=action.parameters)
    assumed_next_state = operator.apply(state, allow_inapplicable_actions=True)
    state_predicates = snapshot.create_propositional_state_snapshot(state=assumed_next_state, current_action=action,
                                                                    observation_objects=state.get_state_objects())
    return assumed_next_state.state_fluents, state_predicates


class NoveltyLearner:
    """Class to manage the novelty calculations for the online learning setting."""

    _execution_db: Dict[str, List]
    logger: logging.Logger

    def __init__(self):
        self._execution_db = {col: [] for col in EXECUTION_DB_COLUMNS}
        self.logger = logging.getLogger(__name__)

    def _compute_state_l1_norm(self, assumed_functions: Dict[str, PDDLFunction],
                               assumed_predicates: List[Predicate], observation_index: int) -> float:
        """Computes the L1 norm of the state.

        Note:
            The L1 norm of the state is calculated as follows:
            - The difference between the predicates in the state and the predicates in the assumed state.
            - The sum of the absolute differences between the values of the functions in the state and the assumed
                functions.
            - The above two values are then summed together to get the L1 norm of the state.

        :param assumed_functions: the assumed functions in the state.
        :param assumed_predicates: the assumed predicates in the state.
        :return: the L1 norm of the state.
        """
        predicate_str = {predicate.untyped_representation for predicate in assumed_predicates}
        state_novelty = len(predicate_str.difference(self._execution_db["state_predicates"][observation_index]))
        for observed_numeric_function in self._execution_db["functions_and_values"][observation_index]:
            if observed_numeric_function.untyped_representation not in assumed_functions:
                continue

            state_novelty += abs(observed_numeric_function.value - assumed_functions[
                observed_numeric_function.untyped_representation].value)

        return state_novelty

    def _previous_state_exists(
            self, lifted_prev_state_functions: Dict[str, PDDLFunction],
            lifted_prev_state_predicates: List[Predicate], action: ActionCall) -> bool:
        """Checks if the state is in the execution DB.

        :param lifted_prev_state_functions: the lifted functions in the previous state.
        :param lifted_prev_state_predicates: the lifted predicates in the previous state.
        :param action: the action that was executed.
        :return: True if the state is in the execution DB, False otherwise.
        """
        state_predicate_str = {predicate.untyped_representation for predicate in lifted_prev_state_predicates}
        state_functions = {function.state_representation for function in lifted_prev_state_functions.values()}
        for i, action_name in enumerate(self._execution_db["lifted_action"]):
            if action_name != action.name:
                continue

            function_values = {function.state_representation for function in
                               self._execution_db["functions_and_values"][i]}
            if state_predicate_str == self._execution_db["state_predicates"][i] and \
                    function_values == state_functions:
                return True

        return False

    def calculate_novelty(
            self, action: ActionCall, state: State, domain: Domain, observed_actions: List[str]) -> float:
        """Calculates the novelty rate of the action.

        :param action: the action to calculate the novelty rate of.
        :param state: the state that the action is being applied in.
        :param domain: the domain that was learned during the online learning.
        :param observed_actions: the actions that were observed during the online learning.
        :return: the novelty rate of the action.
        """
        self.logger.info("Trying to calculate the novelty rate of the action.")
        if action.name not in observed_actions:
            self.logger.debug(f"The action {action.name} has not been observed. Setting default novelty rate.")
            return DEFAULT_EXPLORATION_CONSTANT

        self.logger.debug(f"The action {action.name} is partially learned. Calculating the novelty rate of the action.")
        snapshot = EnvironmentSnapshot(domain)
        numeric_matcher = NumericFunctionMatcher(domain)
        predicate_matcher = PredicatesMatcher(domain)
        assumed_functions, assumed_predicates = _try_applying_action_in_state(snapshot, action, state, domain)
        lifted_next_state_functions = numeric_matcher.match_state_functions(action, assumed_functions)
        lifted_next_predicates = predicate_matcher.get_possible_literal_matches(action, list(assumed_predicates))

        prev_state_predicates = snapshot.create_propositional_state_snapshot(state, action, state.get_state_objects())
        prev_state_functions = snapshot.create_numeric_state_snapshot(state, action, state.get_state_objects())
        lifted_prev_state_functions = numeric_matcher.match_state_functions(action, prev_state_functions)
        lifted_prev_state_predicates = predicate_matcher.get_possible_literal_matches(
            action, list(prev_state_predicates))

        num_successful_executions = sum(
            [1 for i, lifted_action in enumerate(self._execution_db["lifted_action"]) if
             self._execution_db["execution_result"][i] == SUCCESS_RESULT and lifted_action == action.name])

        if self._previous_state_exists(lifted_prev_state_functions, lifted_prev_state_predicates, action):
            self.logger.debug("The previous state exists in the execution DB. The action is not novel.")
            return 1 / num_successful_executions

        novelty_rates = []
        for i, action_name in enumerate(self._execution_db["lifted_action"]):
            if action_name != action.name:
                continue

            novelty_rates.append(
                self._compute_state_l1_norm(lifted_next_state_functions, lifted_next_predicates, i))

        return min(novelty_rates) / num_successful_executions

    def add_sample_to_execution_db(self, lifted_action: str, bounded_functions: Dict[str, PDDLFunction],
                                   bounded_predicates: List[Predicate], result: int) -> None:
        """Adds the new sample to the execution DB.

        :param lifted_action: the lifted action that was executed.
        :param bounded_functions: the lifted bounded functions that are a match to the executed action.
        :param bounded_predicates: the lifted bounded predicates that are a match to the executed action.
        :param result: the result of the execution, i.e., whether the action was successful or not.
        """
        self.logger.info("Adding a sample to the execution database.")
        self._execution_db["lifted_action"].append(lifted_action)
        state_predicates = {predicate.untyped_representation for predicate in bounded_predicates}
        self._execution_db["state_predicates"].append(state_predicates)
        self._execution_db["functions_and_values"].append(bounded_functions.values())
        self._execution_db["execution_result"].append(result)

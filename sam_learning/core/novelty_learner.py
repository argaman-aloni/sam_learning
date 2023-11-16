"""Class to manage the novelty calculations for the online learning setting."""
import logging
from typing import Tuple, Dict, List, Set

from pddl_plus_parser.models import ActionCall, State, PDDLFunction, Domain, Operator

EXECUTION_DB_COLUMNS = ["lifted_action", "state_predicates", "functions_and_values", "execution_result"]
SUCCESS_RESULT = 1
FAIL_RESULT = -1
DEFAULT_EXPLORATION_CONSTANT = 1000


def _try_applying_action_in_state(
        action: ActionCall, state: State, domain: Domain) -> Tuple[Dict[str, PDDLFunction], Set[str]]:
    """Tries to apply the action according to the currently learned model.

    :param action: the action that is being applied.
    :param state: the state that we are trying to apply the action in.
    :return: the state after the action was applied - the information returned is grounded to avoid conflict of
        bounded lifted representations.
    """
    operator = Operator(action=domain.actions[action.name], domain=domain,
                        grounded_action_call=action.parameters)
    assumed_next_state = operator.apply(state, allow_inapplicable_actions=True)
    state_predicates = {predicate.untyped_representation for predicates in
                        assumed_next_state.state_predicates.values() for predicate in predicates}
    return assumed_next_state.state_fluents, state_predicates


class NoveltyLearner:
    """Class to manage the novelty calculations for the online learning setting."""

    def __init__(self):
        self._execution_db = {col: [] for col in EXECUTION_DB_COLUMNS}
        self.logger = logging.getLogger(__name__)

    def _compute_state_l1_norm(
            self, assumed_functions: Dict[str, PDDLFunction], assumed_predicates: Set[str],
            observation_index: int) -> float:
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
        state_novelty = len(assumed_predicates.difference(self._execution_db["state_predicates"][observation_index]))
        state_novelty += sum([abs(observed_numeric_function.value - assumed_functions[
            observed_numeric_function.untyped_representation].value)
                              for observed_numeric_function in
                              self._execution_db["functions_and_values"][observation_index]])
        return state_novelty

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
        assumed_functions, assumed_predicates = _try_applying_action_in_state(action, state, domain)
        num_successful_executions = sum(
            [1 for i, lifted_action in enumerate(self._execution_db["lifted_action"]) if
             self._execution_db["execution_result"][i] == SUCCESS_RESULT and lifted_action == action.name])

        novelty_rates = []
        for i, _ in enumerate(self._execution_db["lifted_action"]):
            novelty_rates.append(self._compute_state_l1_norm(assumed_functions, assumed_predicates, i))

        return min(novelty_rates) / num_successful_executions

    def add_sample_to_execution_db(self, lifted_action: str, state: State, result: int) -> None:
        """Adds the new sample to the execution DB.

        :param lifted_action: the lifted action that was executed.
        :param state: the state that the action was executed in.
        :param result: the result of the execution, i.e., whether the action was successful or not.
        """
        self.logger.info("Adding a sample to the execution database.")
        self._execution_db["lifted_action"].append(lifted_action)
        state_predicates = {predicate.untyped_representation for predicates in
                            state.state_predicates.values() for predicate in predicates}
        self._execution_db["state_predicates"].append(state_predicates)
        self._execution_db["functions_and_values"].append(state.state_fluents.values())
        self._execution_db["execution_result"].append(result)

"""Module to calculate the information gain of the propositional part of an action."""

import logging
from typing import Set, List, Tuple

from pddl_plus_parser.models import Predicate, Precondition

DUMMY_EFFECT = Predicate("dummy_effect", signature={}, is_positive=True)


class OnlineDiscreteModelLearner:
    """Online model learning algorithm that learns discrete action models.
    Similar to the work of Sarath Sreedharan and Michael Katz (2023)."""

    logger: logging.Logger
    action_name: str
    cannot_be_preconditions: Set[Predicate]
    must_be_preconditions: List[Set[Predicate]]
    predicates_superset: Set[Predicate]
    cannot_be_effects: Set[Predicate]
    must_be_effects: Set[Predicate]

    def __init__(self, action_name: str, pb_predicates: Set[Predicate]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.predicates_superset = {predicate.copy() for predicate in pb_predicates}
        self.cannot_be_preconditions = set()
        self.cannot_be_effects = set()
        self.must_be_preconditions = []
        self.must_be_effects = set()

    def _add_positive_pre_state_observation(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a positive pre-state observation and deduces the predicates to filter from the preconditions as
        well as applies unit propagation to the predicates in the must_be_preconditions CNFs .

        :param predicates_in_state: the predicates observed in the state in which the action was executed successfully.
        """
        self.logger.info(f"Adding a new positive pre-state observation for the action {self.action_name}.")
        not_preconditions = self.predicates_superset.difference(predicates_in_state)
        self.cannot_be_preconditions = (
            not_preconditions if len(self.cannot_be_preconditions) == 0 else self.cannot_be_preconditions.union(not_preconditions)
        )
        for not_precondition in self.cannot_be_preconditions:
            self.logger.debug("Removing false positives from the must be preconditions set.")
            for must_be_preconditions_set in self.must_be_preconditions:
                must_be_preconditions_set.discard(not_precondition)

    def _add_positive_post_state_observation(self, pre_state_predicates: Set[Predicate], post_state_predicates: Set[Predicate]) -> None:
        """Adds a positive pre and post-state observation and deduces the predicates to filter from the effects.

        :param pre_state_predicates: the predicates observed in the state in which the action was executed successfully.
        :param post_state_predicates: the predicates observed in the state following the action execution.
        """
        self.logger.info(f"Adding a new positive post-state observation for the action {self.action_name}.")
        self.cannot_be_effects.update([predicate.copy(is_negated=True) for predicate in post_state_predicates])
        self.must_be_effects.update(set([predicate.copy() for predicate in post_state_predicates.difference(pre_state_predicates)]))
        # Removing the predicates that cannot be effects from the must be effects set.
        self.must_be_effects.difference_update(self.cannot_be_preconditions)

    def _apply_unit_propagation(self) -> None:
        """Applies unit propagation to the must be preconditions CNFs."""
        self.logger.debug("Applying unit propagation to the must be preconditions CNFs.")
        unit_clauses = [cnf for cnf in self.must_be_preconditions if len(cnf) == 1]
        new_must_be_preconditions = unit_clauses
        for cnf in self.must_be_preconditions:
            if len(cnf) == 1:
                # We verify externally that no duplication clauses are added.
                continue

            # If an atom from the unit clause is already in the CNF, the clause becomes true.
            if len(cnf.intersection({element for clause in unit_clauses for element in clause})) > 0:
                self.logger.debug("The CNF is already satisfied by the unit clause. No need to apply unit propagation.")
                continue

            new_must_be_preconditions.append(cnf)

        self.must_be_preconditions = new_must_be_preconditions

    def _add_negative_pre_state_observation(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a negative sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action could not be applied.
        """
        self.logger.debug(f"Adding a new negative sample for the action {self.action_name}.")
        preconditions_not_in_state = self.predicates_superset.difference(predicates_in_state)
        if any([cnf.issubset(predicates_in_state) and predicates_in_state.issubset(cnf) for cnf in self.must_be_preconditions]):
            self.logger.debug("The state is already in the must be preconditions. No need to add it again.")
            return

        self.must_be_preconditions.append(preconditions_not_in_state.difference(self.cannot_be_preconditions))
        self._apply_unit_propagation()

    def add_transition_data(
        self, pre_state_predicates: Set[Predicate], post_state_predicates: Set[Predicate] = None, is_transition_successful: bool = True
    ) -> None:
        """Collects the data from the transition and updates the model.

        :param pre_state_predicates: the predicates observed in the state in which the action was executed.
        :param post_state_predicates: the predicates observed in the state following the action execution.
        :param is_transition_successful: a boolean indicating if the action was successfully applied in the pre-state.
        """
        self.logger.debug(
            f"Adding {'positive' if is_transition_successful else 'negative'} transition data for the action {self.action_name}."
        )
        if is_transition_successful:
            self._add_positive_pre_state_observation(pre_state_predicates)
            self._add_positive_post_state_observation(pre_state_predicates, post_state_predicates)
            return

        self._add_negative_pre_state_observation(pre_state_predicates)
        self.logger.debug("Cannot learn anything on the effects from an inapplicable state transition!")

    def get_safe_model(self) -> Tuple[Precondition, Set[Predicate]]:
        """Returns the safe model of the action.

        :return: the safe model of the action.
        """
        self.logger.info(f"Getting the safe model for the action {self.action_name}.")
        safe_precondition = Precondition("and")
        for predicate in self.predicates_superset.difference(self.cannot_be_preconditions):
            safe_precondition.add_condition(predicate.copy())

        return safe_precondition, self.must_be_effects

    def get_optimistic_model(self) -> Tuple[Precondition, Set[Predicate]]:
        """Returns the optimistic model of the action.

        :return: the optimistic model of the action.
        """
        self.logger.info(f"Getting the optimistic model for the action {self.action_name}.")
        # if action was not observed yet, return the superset of the predicates
        optimistic_precondition = Precondition("and")
        for cnf in self.must_be_preconditions:
            if len(cnf) == 0:
                continue

            if len(cnf) == 1:
                predicate = cnf.pop()
                optimistic_precondition.add_condition(predicate.copy())
                cnf.add(predicate)
                continue

            or_condition = Precondition("or")
            for predicate in cnf:
                or_condition.add_condition(predicate.copy())

            optimistic_precondition.add_condition(or_condition)

        if len(self.cannot_be_effects) == 0:
            # Assuming that the action's effects are all the positive predicates
            return optimistic_precondition, {predicate for predicate in self.predicates_superset if predicate.is_positive}

        return optimistic_precondition, self.predicates_superset.difference(self.cannot_be_effects)

    def is_state_in_safe_model(self, state: Set[Predicate]) -> bool:
        """Checks if state predicates hold in the safe model.

        :param state: The state predicates to check.
        :return: True if the state is in the safe model, False otherwise.
        """
        safe_conditions = self.predicates_superset.difference(self.cannot_be_preconditions)
        return state.issuperset(safe_conditions)

    def is_state_not_applicable_in_safe_discrete_model(self, state: Set[Predicate]) -> bool:
        """Checks if state predicates only include the predicates that are not preconditions for the action.

        Note:
            Since actions can have empty discrete preconditions, this also checks if a failure occurred due to
            the discrete part of the action.

        :param state: The state predicates to check.
        :return: True if the state is not applicable in the safe model, False otherwise.
        """
        # a state is not applicable if the state does not include any of the predicates that are possibly "must be preconditions"
        state_does_not_contain_unit_clause = any([len(state.intersection(cnf)) == 0 for cnf in self.must_be_preconditions])
        return len(self.must_be_preconditions) > 0 and state_does_not_contain_unit_clause

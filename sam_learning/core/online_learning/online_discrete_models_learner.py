"""Module to calculate the information gain of the propositional part of an action."""
import logging
from typing import Set, List, Tuple

from pddl_plus_parser.models import Predicate, State
from pysat.examples.hitman import Hitman


class OnlineDiscreteModelLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    preconditions_superset: Set[Predicate]
    cannot_be_preconditions: Set[Predicate]
    must_be_preconditions: List[Set[Predicate]]

    def __init__(self, action_name: str, lifted_predicates: Set[Predicate]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self._predicates_superset = {predicate.copy() for predicate in lifted_predicates}
        self.preconditions_superset = {predicate.copy() for predicate in lifted_predicates}
        self.cannot_be_preconditions = set()
        self.cannot_be_effects = set()
        self.must_be_preconditions = []
        self.must_be_effects = set()
        self._hitting_set_solver = Hitman(solver="m22", htype="lbx")
        self._current_hitting_set = set()
        self._obsolete_hitting_sets = []

    def _reset_hitting_set(self) -> None:
        """Resets the hitting set solver as the preconditions superset and the other groups have changed."""
        self.cannot_be_preconditions.update(self._current_hitting_set)
        self._obsolete_hitting_sets.append(self._hitting_set_solver)
        self.logger.debug("Resetting the hitting set solver.")
        self._hitting_set_solver.delete()
        self._hitting_set_solver = Hitman(solver="m22", htype="lbx")
        self._hitting_set_solver.hit(self.preconditions_superset)
        for preconditions_set in self.must_be_preconditions:
            self._hitting_set_solver.hit(preconditions_set)

        for used_hitting_set in self._obsolete_hitting_sets:
            self._hitting_set_solver.block(used_hitting_set)

    def _add_positive_pre_state_observation(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action was executed successfully.
        """
        self.logger.info(f"Adding a new positive pre-state observation for the action {self.action_name}.")
        if len(self.cannot_be_preconditions) == 0:
            self.logger.debug("Since this is the first positive observation we need to create the complement of the predicates.")
            self.cannot_be_preconditions = {predicate for predicate in self._predicates_superset}
            self.cannot_be_preconditions.difference_update(predicates_in_state)

        not_preconditions = self.preconditions_superset.difference(predicates_in_state)
        self.cannot_be_preconditions.update(not_preconditions)
        for not_precondition in self.cannot_be_preconditions:
            self.logger.debug("Removing false positives from the must be preconditions set.")
            for must_be_preconditions_set in self.must_be_preconditions:
                must_be_preconditions_set.discard(not_precondition)

    def _add_positive_post_state_observation(self, predicates_in_pre_state: Set[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        Notice:
            We added the parameter expected_to_be_positive to indicate that in some cases the current "optimistic" preconditions
            might be wrong and need to be replaced. For example if we assumed that the preconditions indicate that the preconditions
            are "a" and the action was applicable when "a" was false, it means that the preconditions are not correct, and we need to
            replace them with the new ones. This is not the case for the cannot be preconditions, since they are always correct.

        :param predicates_in_pre_state: the predicates observed in the state in which the action was executed successfully.
        """
        self.logger.info(f"Adding a new positive post-state observation for the action {self.action_name}.")
        if len(self.cannot_be_effects) == 0:
            self.logger.debug("Since this is the first positive observation we need to create the complement of the predicates.")
            self.cannot_be_effects = {predicate for predicate in self.preconditions_superset}
            self.cannot_be_effects.difference_update(predicates_in_pre_state)

        self.must_be_effects
        not_preconditions = self.preconditions_superset.difference(predicates_in_pre_state)
        self.preconditions_superset.intersection_update(predicates_in_pre_state)
        self.cannot_be_preconditions.update(not_preconditions)
        for not_precondition in self.cannot_be_preconditions:
            self.logger.debug("Removing false positives from the must be preconditions set.")
            for must_be_preconditions_set in self.must_be_preconditions:
                must_be_preconditions_set.discard(not_precondition)

        self._reset_hitting_set()
        if not expected_to_be_positive:
            self._current_hitting_set = self._hitting_set_solver.get()

    def add_negative_sample(self, predicates_in_state: Set[Predicate], expected_to_be_negative: bool) -> None:
        """Adds a negative sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action could not be applied.
        :param expected_to_be_negative: whether the sample was expected to be negative or not.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        preconditions_not_in_state = self.preconditions_superset.difference(predicates_in_state)
        self.must_be_preconditions.append(preconditions_not_in_state)
        self._hitting_set_solver.hit(preconditions_not_in_state)
        if not expected_to_be_negative:
            self.logger.debug("Changing the hitting set as the preconditions were wrong in classifying the state.")
            self._obsolete_hitting_sets.append(self._current_hitting_set)
            for used_hitting_set in self._obsolete_hitting_sets:
                self._hitting_set_solver.block(used_hitting_set)

            self._current_hitting_set = self._hitting_set_solver.get()

    def get_possible_preconditions_hitting_set(self) -> Set[Predicate]:
        """Calculates the hitting that represents the action's preconditions.

        :return: the hitting set of the action's preconditions.
        """
        self.logger.info(f"Calculating the hitting set for the action {self.action_name}.")
        return self._current_hitting_set

    def add_transition_data(self, pre_state: State, post_state: State, is_transition_successful: bool) -> None:
        """

        :param pre_state_predicates:
        :param post_state:
        :param is_transition_successful:
        :return:
        """
        if is_transition_successful:
            self._add_positive_pre_state_observation(post_state)
            self._add_positive_post_state_observation(pre_state, post_state)
            return

        self.add_negative_sample(pre_state_predicates)
        self.logger.debug("Cannot learn anything on the effects from an inapplicable state transition!")

    def get_safe_model(self) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Returns the safe model of the action.

        :return: the safe model of the action.
        """
        self.logger.info(f"Getting the safe model for the action {self.action_name}.")
        pass

    def return_optimistic_model(self) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Returns the optimistic model of the action.

        :return: the optimistic model of the action.
        """
        self.logger.info(f"Getting the optimistic model for the action {self.action_name}.")
        return self.preconditions_superset, self.cannot_be_preconditions

    def return_consistent_model(self) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Returns the consistent model of the action.

        :return: the consistent model of the action.
        """
        self.logger.info(f"Getting the consistent model for the action {self.action_name}.")
        return self.preconditions_superset, self.cannot_be_preconditions

    def calculate_sample_information_gain(self, new_lifted_sample: Set[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_lifted_sample: the new sample to calculate the information gain of.
        :return: the information gain of the new sample.
        """
        # TODO: Check if we still need this information gain calculation, and if so, complete it.
        if new_lifted_sample.issubset(self.cannot_be_preconditions):
            return 0

        if new_lifted_sample.issuperset(self.preconditions_superset):
            return 0

        return 1  # TODO: implement this

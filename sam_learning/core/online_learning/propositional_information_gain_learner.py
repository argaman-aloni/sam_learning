"""Module to calculate the information gain of the propositional part of an action."""
import logging
from typing import Set, List

from pddl_plus_parser.models import Predicate
from pysat.examples.hitman import Hitman


class PropositionalInformationGainLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    preconditions_superset: Set[Predicate]
    cannot_be_preconditions: Set[Predicate]
    must_be_preconditions: List[Set[Predicate]]

    def __init__(self, action_name: str, lifted_predicates: Set[Predicate]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.preconditions_superset = {predicate.copy() for predicate in lifted_predicates}
        self.cannot_be_preconditions = set()
        self.must_be_preconditions = []
        self._hitting_set_solver = Hitman(solver="m22", htype="lbx")

    def add_positive_sample(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action was executed successfully.
        """
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        if len(self.cannot_be_preconditions) == 0:
            self.cannot_be_preconditions = {predicate for predicate in self.preconditions_superset}
            self.cannot_be_preconditions.difference_update(predicates_in_state)

        not_preconditions = self.preconditions_superset.difference(predicates_in_state)
        self.preconditions_superset.intersection_update(predicates_in_state)
        self.cannot_be_preconditions.update(not_preconditions)

        for not_precondition in self.cannot_be_preconditions:
            self.logger.debug("Removing false positives from the must be preconditions set.")
            for must_be_preconditions_set in self.must_be_preconditions:
                must_be_preconditions_set.discard(not_precondition)

    def add_negative_sample(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a negative sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action could not be applied.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        preconditions_not_in_state = self.preconditions_superset.difference(predicates_in_state)
        self.must_be_preconditions.append(preconditions_not_in_state)

    def get_possible_preconditions_hitting_set(self) -> Set[Predicate]:
        """Calculates the hitting that represents the action's preconditions.

        :return: the hitting set of the action's preconditions.
        """
        self.logger.info(f"Calculating the hitting set for the action {self.action_name}.")

    def calculate_sample_information_gain(self, new_lifted_sample: Set[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_lifted_sample: the new sample to calculate the information gain of.
        :return: the information gain of the new sample.
        """
        if new_lifted_sample.issubset(self.cannot_be_preconditions):
            return 0

        if new_lifted_sample.issuperset(self.preconditions_superset):
            return 0

        return 1  # TODO: implement this

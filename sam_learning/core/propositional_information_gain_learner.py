"""Module to calculate the information gain of the propositional part of an action."""
import logging
from typing import Set

from pddl_plus_parser.models import Predicate


class PropositionalInformationGainLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    preconditions_superset: Set[Predicate]
    cannot_be_preconditions: Set[Predicate]
    must_be_preconditions: Set[Set[Predicate]]

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.preconditions_superset = set()
        self.cannot_be_preconditions = set()
        self.must_be_preconditions = set()

    def add_positive_sample(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action was executed successfully.
        """
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        if len(self.preconditions_superset) == 0:
            self.preconditions_superset = {predicate for predicate in predicates_in_state}
            return

        if len(self.cannot_be_preconditions) == 0:
            self.cannot_be_preconditions = {predicate for predicate in self.preconditions_superset}
            self.cannot_be_preconditions.difference_update(predicates_in_state)

        self.preconditions_superset.intersection_update(predicates_in_state)
        not_preconditions = self.preconditions_superset.difference(predicates_in_state)
        self.cannot_be_preconditions.update(not_preconditions)

        for not_precondition in not_preconditions:
            for must_be_precondition in self.must_be_preconditions:
                if not_precondition in must_be_precondition:
                    self.must_be_preconditions.remove(must_be_precondition)

    def add_negative_sample(self, predicates_in_state: Set[Predicate]) -> None:
        """Adds a negative sample to the samples list used to create the action's precondition.

        :param predicates_in_state: the predicates observed in the state in which the action could not be applied.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        preconditions_not_in_state = self.preconditions_superset.difference(predicates_in_state)
        self.must_be_preconditions.add(preconditions_not_in_state)

    def calculate_sample_information_gain(self, new_lifted_sample: Set[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_lifted_sample: the new sample to calculate the information gain of.
        :return: the information gain of the new sample.
        """
        if new_lifted_sample.issubset(self.cannot_be_preconditions):
            return 0

        if new_lifted_sample.issubset(self.preconditions_superset):
            return 0

        return 1  # TODO: implement this

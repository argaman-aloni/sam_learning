"""Module that stores amd learns an action's numeric state fluents."""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Set

import numpy as np
from pddl_plus_parser.models import PDDLFunction, Precondition, NumericalExpressionTree, ConditionalEffect

from sam_learning.core.convex_hull_learner import ConvexHullLearner
from sam_learning.core.linear_regression_learner import LinearRegressionLearner

np.seterr(divide='ignore', invalid='ignore')


class NumericFluentStateStorage:
    """Stores and learned the numeric state fluents of a single action."""

    logger: logging.Logger
    previous_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.
    next_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        self.logger = logging.getLogger(__name__)
        self.previous_state_storage = defaultdict(list)
        self.next_state_storage = defaultdict(list)
        self.convex_hull_learner = ConvexHullLearner(action_name, domain_functions)
        self.linear_regression_learner = LinearRegressionLearner(action_name, domain_functions)

    def add_to_previous_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the previous state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        for state_fluent_lifted_str, state_fluent_data in state_fluents.items():
            self.previous_state_storage[state_fluent_lifted_str].append(state_fluent_data.value)

    def add_to_next_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the next state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        for state_fluent_lifted_str, state_fluent_data in state_fluents.items():
            self.next_state_storage[state_fluent_lifted_str].append(state_fluent_data.value)
            if len(self.previous_state_storage.get(state_fluent_lifted_str, [])) != \
                    len(self.next_state_storage[state_fluent_lifted_str]):
                self.logger.debug("This is a case where effects create new fluents - should adjust the previous state.")
                self.previous_state_storage[state_fluent_lifted_str].append(0)

    def filter_out_inconsistent_state_variables(self) -> None:
        """Filters out fluents that appear only in part of the states since they are not safe.

        :return: only the safe state variables that appear in *all* states.
        """
        max_function_len = max([len(values) for values in self.previous_state_storage.values()])
        self.previous_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                       self.previous_state_storage.items() if len(state_values) == max_function_len}
        self.next_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                   self.next_state_storage.items() if len(state_values) == max_function_len}

    def construct_safe_linear_inequalities(
            self, relevant_fluents: Optional[List[str]] = None) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: The precondition that contains the linear inequalities.
        """

        return self.convex_hull_learner.construct_safe_linear_inequalities(
            self.previous_state_storage, relevant_fluents)

    def construct_assignment_equations(self, allow_unsafe_learning: bool = False) -> Union[
        Tuple[List[ConditionalEffect], Precondition], Set[NumericalExpressionTree],
        Tuple[Set[NumericalExpressionTree], Precondition]]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :param allow_unsafe_learning: whether to allow learning from unsafe data.
        :return: the constructed assignment statements.
        """
        return self.linear_regression_learner.construct_assignment_equations(
            self.previous_state_storage, self.next_state_storage, allow_unsafe_learning=allow_unsafe_learning)

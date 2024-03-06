"""Module that stores amd learns an action's numeric state fluents."""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from pddl_plus_parser.models import PDDLFunction, Precondition, NumericalExpressionTree

from sam_learning.core.numeric_learning.convex_hull_learner import ConvexHullLearner
from sam_learning.core.numeric_learning.linear_regression_learner import LinearRegressionLearner

np.seterr(divide="ignore", invalid="ignore")


class NumericFluentStateStorage:
    """Stores and learned the numeric state fluents of a single action."""

    logger: logging.Logger
    previous_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.
    next_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        self.action_name = action_name
        self.logger = logging.getLogger(__name__)
        self.previous_state_storage = defaultdict(list)
        self.next_state_storage = defaultdict(list)
        self.convex_hull_learner = ConvexHullLearner(action_name, domain_functions)
        self.linear_regression_learner = LinearRegressionLearner(action_name, domain_functions)

    def add_to_previous_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the previous state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        previous_state_values = {
            state_fluent_lifted_str: state_fluent_data.value for state_fluent_lifted_str, state_fluent_data in state_fluents.items()
        }
        for state_fluent_lifted_str, state_fluent_data in previous_state_values.items():
            self.previous_state_storage[state_fluent_lifted_str].append(state_fluent_data)

        self.linear_regression_learner.add_new_observation(previous_state_values, store_in_prev_state=True)

    def add_to_next_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the next state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        next_state_values = {state_fluent_lifted_str: state_fluent_data.value for state_fluent_lifted_str, state_fluent_data in state_fluents.items()}
        self.linear_regression_learner.add_new_observation(next_state_values, store_in_prev_state=False)

    def filter_out_inconsistent_state_variables(self) -> None:
        """Filters out fluents that appear only in part of the states since they are not safe.

        :return: only the safe state variables that appear in *all* states.
        """
        max_function_len = max([len(values) for values in self.previous_state_storage.values()])
        new_prev_state_storage = defaultdict(list)
        for lifted_function, state_values in self.previous_state_storage.items():
            if len(state_values) == max_function_len:
                new_prev_state_storage[lifted_function].extend(state_values)

        self.previous_state_storage = new_prev_state_storage

        new_next_state_storage = defaultdict(list)
        for lifted_function, state_values in self.next_state_storage.items():
            if len(state_values) == max_function_len:
                new_next_state_storage[lifted_function].extend(state_values)

        self.next_state_storage = new_next_state_storage

    def construct_safe_linear_inequalities(self, relevant_fluents: Optional[List[str]] = None) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: The precondition that contains the linear inequalities.
        """
        self.logger.info("Constructing the safe linear inequalities.")
        return self.convex_hull_learner.construct_safe_linear_inequalities(self.previous_state_storage, relevant_fluents)

    def construct_assignment_equations(self) -> Tuple[Set[NumericalExpressionTree], Optional[Precondition], bool]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :return: the constructed assignment statements.
        """
        self.logger.info("Constructing the assignment equations.")
        return self.linear_regression_learner.construct_assignment_equations()

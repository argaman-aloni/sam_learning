"""Module that stores amd learns an action's numeric state fluents."""
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pddl_plus_parser.models import PDDLFunction, Precondition, NumericalExpressionTree

from sam_learning.core.numeric_learning.incremental_convex_hull_learner import IncrementalConvexHullLearner
from sam_learning.core.numeric_learning.numeric_fluent_learner_algorithm import NumericFluentStateStorage

np.seterr(divide="ignore", invalid="ignore")


class IncrementalNumericFluentStateStorage(NumericFluentStateStorage):
    """Stores and learned the numeric state fluents of a single action."""

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        super().__init__(action_name, domain_functions)
        self.logger = logging.getLogger(__name__)
        self.convex_hull_learner = IncrementalConvexHullLearner(action_name, domain_functions)

    def add_to_previous_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the previous state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        super().add_to_previous_state_storage(state_fluents)
        prev_state_values = {
            state_fluent_lifted_str: state_fluent_data.value
            for state_fluent_lifted_str, state_fluent_data in state_fluents.items()
        }
        self.convex_hull_learner.add_new_point(prev_state_values)

    def construct_safe_linear_inequalities(self, relevant_fluents: Optional[List[str]] = None) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: The precondition that contains the linear inequalities.
        """
        self.logger.info("Constructing the safe linear inequalities.")
        return self.convex_hull_learner.construct_convex_hull_inequalities()

    def construct_assignment_equations(self) -> Tuple[Set[NumericalExpressionTree], Optional[Precondition], bool]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :return: the constructed assignment statements.
        """
        self.logger.info("Constructing the assignment equations.")
        return self.linear_regression_learner.construct_assignment_equations()

"""Learns fluents using Oblique trees."""
from typing import List, Tuple, Union

from pddl_plus_parser.models import Domain, Observation
from stree import Stree
from stree.Splitter import Snode

from sam_learning.core import LearnerDomain
from sam_learning.core.learning_types import ConditionType
from sam_learning.core.numeric_utils import prettify_coefficients, prettify_floating_point_number
from sam_learning.core.unsafe_numeric_fluents_learning_base import UnsafeFluentsLearning, CLASS_COLUMN

TOLERANCE = 0.01


class ObliqueTreeFluentsLearning(UnsafeFluentsLearning):
    """Learns models creating an oblique tree and iterating over its calculated nodes."""

    def __init__(self, action_name: str, polynomial_degree: int = 0,
                 partial_domain: Union[Domain, LearnerDomain] = None):
        super().__init__(action_name, polynomial_degree, partial_domain)

    def _iterate_over_tree(self, node: Snode, features_names: List[str],
                           coefficients_path: List[List[float]], intercepts_path: List[float]) -> List[str]:
        """Iterates over the tree and creates the preconditions from the SVC nodes.

        Notice:
            All the routes are connected with an OR condition while the nodes on each route are connected though AND.

        :param node: the node that is currently being traversed.
        :param coefficients_path: the path containing the coefficient parameters of the current route.
        :param intercepts_path: the path containing the intercept parameters of the current route.
        :return: the inequality conditions to be concatenated with an OR condition.
        """
        if node.is_leaf():
            if node._class == 1:
                self.logger.debug("A route has been completed. "
                                  "All nodes on the route are connected with an AND condition.")
                constraint_strings = self._create_inequality_constraint_strings(
                    features_names, coefficients_path, intercepts_path)
                return constraint_strings

            return []

        coefficients_path.append(prettify_coefficients(list(node.get_classifier().coef_[0])))
        intercepts_path.append(prettify_floating_point_number(node.get_classifier().intercept_[0]))

        # Node->down == the right child for some unknown reason
        right_child_node = node.get_down()
        # Node->down == the left child for some unknown reason
        left_child_node = node.get_up()
        left_constraint_strings = self._iterate_over_tree(
            left_child_node, features_names, coefficients_path, intercepts_path)
        right_constraint_strings = self._iterate_over_tree(
            right_child_node, features_names, coefficients_path, intercepts_path)

        coefficients_path.pop()
        intercepts_path.pop()

        return left_constraint_strings + right_constraint_strings

    def learn_preconditions(self, positive_observations: List[Observation],
                            negative_observations: List[Observation]) -> Tuple[List[str], ConditionType]:
        """Learning the preconditions of an action using oblique tree technique.

        Notice:
            Oblique tree uses SVM as the base for the calculation. For each SVC node it is defined that
            the goal is defined by sign(w*f(x) + b). The SVC node is defined by the function f(x)
            is the identity function.

        :return: the list of preconditions to be connected with an OR statement.
        """
        self.logger.info("Learning the preconditions of the action using oblique tree.")
        dataframe = self._create_pre_state_classification_dataset(positive_observations, negative_observations)
        stree = Stree(random_state=42, max_depth=5, splitter="cfs").fit(
            dataframe.loc[:, dataframe.columns != CLASS_COLUMN], dataframe[CLASS_COLUMN])
        self.logger.debug("The tree has been built.")
        intercepts_path = []
        coefficients_path = []
        return self._iterate_over_tree(stree.tree_, dataframe.columns.values.tolist(),
                                       coefficients_path, intercepts_path), ConditionType.disjunctive

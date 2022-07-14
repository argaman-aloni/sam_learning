"""Learns fluents using SVM."""
import itertools
import logging
from collections import defaultdict
from typing import Dict, List, NoReturn

import numpy
import pandas as pd
from pddl_plus_parser.models import PDDLFunction, Domain, Observation, ObservedComponent
from stree import Stree, Siterator
from stree.Splitter import Snode

from sam_learning.core import NumericFunctionMatcher

CLASS_COLUMN = "class"

TOLERANCE = 0.01
C = 23


class ObliqueTreeFluentsLearning:
    """Learns models by using linear SVM classifier and evaluating the coefficients of the fluents."""

    polynom_degree: int
    function_matcher: NumericFunctionMatcher

    def __init__(self, action_name: str, polynomial_degree: int = 0, partial_domain: Domain = None):
        self.polynom_degree = polynomial_degree
        self.function_matcher = NumericFunctionMatcher(partial_domain)
        self.logger = logging.getLogger(__name__)
        super().__init__(action_name, polynomial_degree)

    def _add_lifted_functions_to_dataset(self, component: ObservedComponent,
                                         dataset: Dict[str, List[float]]) -> NoReturn:
        """Adds lifted functions to the dataset.

        :param component: the current observed component, i.e. the current trajectory triplet.
        :param dataset: the dataset to which the lifted functions are added.
        """
        matches = self.function_matcher.match_state_functions(component.grounded_action_call,
                                                              component.previous_state.state_fluents)
        for fluent_name, func in matches.items():
            dataset[fluent_name].append(func.value)

    def create_prestate_dataset(self, positive_observations: List[Observation],
                                negative_observations: List[Observation]) -> pd.DataFrame:
        """Creates a dataset for the previous states.

        :param positive_observations: the positive observations.
        :param negative_observations: the negative observations.
        :return: the dataframe for the previous states with both the positive and the negative observations.
        """
        pre_state_data = defaultdict(list)
        for observation in positive_observations:
            for component in observation.components:
                self._add_lifted_functions_to_dataset(component, pre_state_data)
                pre_state_data["class"].append(1)

        for observation in negative_observations:
            for component in observation.components:
                self._add_lifted_functions_to_dataset(component, pre_state_data)
                pre_state_data["class"].append(-1)

        return pd.DataFrame.from_dict(pre_state_data)

    def _create_polynomial_string_recursive(self, fluents: List[str]) -> str:
        """

        :param fluents:
        :return:
        """
        if len(fluents) == 1:
            return fluents[0]

        return f"(* {fluents[0]} {self._create_polynomial_string_recursive(fluents[1:])})"

    def _create_monomial_string(self, fluents: List[str]) -> str:
        """

        :param fluents:
        :return:
        """
        return self._create_polynomial_string_recursive(fluents)

    def _reduce_num_variables(self, input_df: pd.DataFrame, observation_class: List[int]) -> pd.DataFrame:
        """

        :param input_df:
        :param observation_class:
        :return:
        """
        feature_selection_obj = SelectPercentile(chi2, percentile=65)
        feature_selection_obj.fit(input_df, observation_class)
        selected_feature_names = feature_selection_obj.get_feature_names_out(list(self.positive_observations.keys()))
        x_new = feature_selection_obj.transform(input_df)
        return pd.DataFrame(x_new, columns=selected_feature_names)

    def _add_polynom(self, state_fluents: Dict[str, PDDLFunction], is_positive_observation: bool = True) -> NoReturn:
        """

        :param state_fluents:
        :param is_positive_observation:
        :return:
        """
        observations_store = self.positive_observations if is_positive_observation else self.negative_observations
        if self.polynom_degree == 1:
            for first_fluent, second_fluent in itertools.combinations(list(state_fluents.keys()), r=2):
                multiplied_fluent = self._create_monomial_string([first_fluent, second_fluent])
                observations_store[multiplied_fluent].append(
                    state_fluents[first_fluent].value * state_fluents[second_fluent].value)
            return

        for degree in range(2, self.polynom_degree + 1):
            for fluent_combination in itertools.combinations_with_replacement(
                    list(state_fluents.keys()), r=degree):
                polynomial_fluent = self._create_monomial_string(list(fluent_combination))
                values = [state_fluents[fluent].value for fluent in fluent_combination]
                observations_store[polynomial_fluent].append(numpy.prod(values))

    def _iterate_over_tree(self, node: Snode, features_names: List[str],
                           coefficients_path: List[List[float]], intercepts_path: List[float]) -> List[str]:
        """

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

        coefficients_path.append(list(node.get_classifier().coef_[0]))
        intercepts_path.append(node.get_classifier().intercept_[0])
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
                            negative_observations: List[Observation]) -> str:
        """Learning the preconditions of an action using oblique tree technique.

        Notice:
            Oblique tree uses SVM as the base for the calculation. For each SVC node it is defined that
            the goal is defined by sign(w*f(x) + b). The SVC node is defined by the function f(x)
            is the identity function.

        :return:
        """
        dataframe = self.create_prestate_dataset(positive_observations, negative_observations)
        stree = Stree(C=C, random_state=42).fit(dataframe.loc[:, dataframe.columns != CLASS_COLUMN],
                                                dataframe[CLASS_COLUMN])
        self.logger.debug("The tree has been built.")
        intercepts_path = []
        coefficients_path = []
        all_constraints = ["(or\n"]
        all_constraints.extend(self._iterate_over_tree(stree.tree_, dataframe.columns.values.tolist(),
                                                       coefficients_path, intercepts_path))
        all_constraints.append(")")
        return "".join(all_constraints)

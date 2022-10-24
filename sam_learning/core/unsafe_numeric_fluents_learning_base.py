"""Base class for the unsafe numeric fluents learning approches."""
import itertools
import logging
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy
import pandas as pd
from pddl_plus_parser.models import PDDLFunction, Domain, Observation, ObservedComponent
from sklearn.linear_model import LinearRegression

from sam_learning.core import LearnerDomain
from sam_learning.core.learning_types import ConditionType
from sam_learning.core.numeric_function_matcher import NumericFunctionMatcher
from sam_learning.core.numeric_utils import construct_multiplication_strings, construct_non_circular_assignment, \
    construct_linear_equation_string, prettify_coefficients, prettify_floating_point_number

CLASS_COLUMN = "class"


class UnsafeFluentsLearning(ABC):
    """Learns models by using different machine learning approaches (in inheriting class)."""

    polynom_degree: int
    function_matcher: NumericFunctionMatcher
    logger: logging.Logger
    action_name: str

    def __init__(self, action_name: str, polynomial_degree: int = 0,
                 partial_domain: Union[Domain, LearnerDomain] = None):
        self.polynom_degree = polynomial_degree
        self.function_matcher = NumericFunctionMatcher(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name

    def _create_monomial_string(self, fluents: List[str]) -> str:
        """Creates a string representing the monomial of the input fluents.

        :param fluents: the fluents that make up the monomial.
        :return: the monomial string.
        """
        if len(fluents) == 1:
            return fluents[0]

        return f"(* {fluents[0]} {self._create_monomial_string(fluents[1:])})"

    def _add_polynomial(self, lifted_fluents: Dict[str, PDDLFunction], dataset: Dict[str, List[float]]) -> None:
        """Adds a polynomial to the dataset.

        :param lifted_fluents: the lifted fluents that are to be added to the polynomial dataset.
        :param dataset: the dataset containing the fluents values.
        """
        if self.polynom_degree == 1:
            for first_fluent, second_fluent in itertools.combinations(list(lifted_fluents.keys()), r=2):
                multiplied_fluent = self._create_monomial_string([first_fluent, second_fluent])
                dataset[multiplied_fluent].append(
                    lifted_fluents[first_fluent].value * lifted_fluents[second_fluent].value)
            return

        for degree in range(2, self.polynom_degree + 1):
            for fluent_combination in itertools.combinations_with_replacement(
                    list(lifted_fluents.keys()), r=degree):
                polynomial_fluent = self._create_monomial_string(list(fluent_combination))
                values = [lifted_fluents[fluent].value for fluent in fluent_combination]
                dataset[polynomial_fluent].append(numpy.prod(values))

    def _add_lifted_post_state_fluent_to_dataset(
            self, component: ObservedComponent, dataset: Dict[str, List[float]]) -> None:
        """Adds the post state fluents to the dataset.

        :param component: the component that is currently being processed.
        :param dataset: the post state dataset to which the fluents are added.
        """
        matches = self.function_matcher.match_state_functions(component.grounded_action_call,
                                                              component.next_state.state_fluents)
        for fluent_name, func in matches.items():
            dataset[fluent_name].append(func.value)

    def _add_lifted_functions_to_dataset(self, component: ObservedComponent,
                                         dataset: Dict[str, List[float]]) -> None:
        """Adds lifted functions to the dataset.

        :param component: the current observed component, i.e. the current trajectory triplet.
        :param dataset: the dataset to which the lifted functions are added.
        """
        matches = self.function_matcher.match_state_functions(component.grounded_action_call,
                                                              component.previous_state.state_fluents)
        for fluent_name, func in matches.items():
            dataset[fluent_name].append(func.value)

        if self.polynom_degree > 0:
            self._add_polynomial(matches, dataset)

    @staticmethod
    def _filter_out_inconsistent_state_variables(dataset: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Filters out fluents that appear only in part of the states since they are not safe.

        :return: only the safe state variables that appear in *all* states.
        """
        max_function_len = max([len(values) for values in dataset.values()])
        return {lifted_function: state_values for lifted_function, state_values in
                dataset.items() if len(state_values) == max_function_len}

    def _create_pre_state_classification_dataset(self, positive_observations: List[Observation],
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

        filtered_dataset = self._filter_out_inconsistent_state_variables(pre_state_data)
        return pd.DataFrame.from_dict(filtered_dataset)

    def _construct_linear_equation_string(self, multiplication_parts: List[str]) -> str:
        """Construct the addition parts of the linear equation string.

        :param multiplication_parts: the multiplication function strings that are multiplied by the coefficient.
        :return: the string representing the sum of the linear variables.
        """
        if len(multiplication_parts) == 1:
            return multiplication_parts[0]

        inner_layer = self._construct_linear_equation_string(multiplication_parts[1:])
        return f"(+ {multiplication_parts[0]} {inner_layer})"

    def _create_inequality_constraint_strings(self, features_names: List[str], coefficients_path: List[List[float]],
                                              intercepts_path: List[float], should_be_also_equal: bool = True) -> List[
        str]:
        """Creates the string representing f(x) + b >= 0 for each of the linear equations.

        Notice:
            Each path is an AND condition.

        :param features_names: the names of the features that were used in the SVC node.
        :param coefficients_path: the coefficients of the SVC classifiers in each node in the path.
        :param intercepts_path: the intercepts of the SVC classifiers in each node in the path.
        :return: a list of strings representing the inequality constraints.
        """
        inequalities = set()
        for node_coefficients, intercept_value in zip(coefficients_path, intercepts_path):
            multiplication_functions = construct_multiplication_strings(node_coefficients, features_names)
            if len(multiplication_functions) == 0:
                continue

            coefficients_string = self._construct_linear_equation_string(multiplication_functions)
            equality_sign = ">=" if should_be_also_equal else ">"
            inequalities.add(f"({equality_sign} (+ {coefficients_string} {intercept_value}) 0.0)")

        if len(inequalities) == 0:
            return []

        ordered_inequalities = ["(and"] + list(inequalities) + [")"]
        return ordered_inequalities

    def _construct_regression_data(
            self, positive_observations: List[Observation]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Constructs the dictionaries needed to run the regression algorithm on the positive observations' data.

        :param positive_observations: the positive observations.
        :return: the dictionaries needed to run the regression algorithm on the positive observations' data.
        """
        post_state_data = defaultdict(list)
        pre_state_data = defaultdict(list)
        for observation in positive_observations:
            for component in observation.components:
                self._add_lifted_functions_to_dataset(component, pre_state_data)
                self._add_lifted_post_state_fluent_to_dataset(component, post_state_data)
        pre_state_filtered_dataset = self._filter_out_inconsistent_state_variables(pre_state_data)
        post_state_filtered_dataset = self._filter_out_inconsistent_state_variables(post_state_data)
        return post_state_filtered_dataset, pre_state_filtered_dataset

    def _solve_regression(self, dataframe: pd.DataFrame, label_column_name: str) -> Dict[str, float]:
        """Solves the regression problem and returns the resulting effect coefficients and intercept.

        :param dataframe: the dataset to be used for the regression.
        :param label_column_name: the name of the column that contains the post state values used as label.
        :return:
        """
        reg = LinearRegression()
        reg.fit(dataframe.loc[:, dataframe.columns != label_column_name], dataframe[label_column_name])
        self.logger.debug("Regression algorithm trained. Returning the coefficients and the intercept...")
        coefficients_map = {
            column_name: coeff for column_name, coeff in zip(reg.feature_names_in_, prettify_coefficients(reg.coef_))
        }
        coefficients_map["(dummy)"] = prettify_floating_point_number(reg.intercept_)
        return coefficients_map

    def learn_preconditions(self, positive_observations: List[Observation],
                            negative_observations: List[Observation]) -> Tuple[List[str], ConditionType]:
        """Learning the preconditions using a machine learning approach defined in the inherited class.

        :return: the list of preconditions to be connected with an OR statement.
        """
        raise NotImplemented()

    def learn_effects(self, positive_observations: List[Observation]) -> List[str]:
        """Learning the effects using linear regression algorithm.

        :return: the list of effects' equations.
        """
        assignment_statements = []
        post_state_values, pre_state_values = self._construct_regression_data(positive_observations)
        for fluent_name in post_state_values:
            self.logger.debug(f"Building the regression dataframe for {fluent_name}!")
            if all(post_value - pre_value == 0 for pre_value, post_value in
                   zip(pre_state_values[fluent_name], post_state_values[fluent_name])):
                self.logger.debug(f"{fluent_name} is not changed by the action {self.action_name}!")
                continue

            label_column_name = f"{fluent_name}_post_value"
            pre_state_values[label_column_name] = post_state_values[fluent_name]
            regression_dataset = pd.DataFrame.from_dict(pre_state_values)
            coefficients_map = self._solve_regression(regression_dataset, label_column_name)
            if fluent_name in coefficients_map:
                assignment_statements.append(
                    construct_non_circular_assignment(
                        fluent_name, coefficients_map, pre_state_values[fluent_name][0],
                        post_state_values[fluent_name][0]))

                continue

            multiplication_functions = construct_multiplication_strings(
                list(coefficients_map.values()), list(coefficients_map.keys()))
            constructed_right_side = construct_linear_equation_string(multiplication_functions)
            assignment_statements.append(f"(assign {fluent_name} {constructed_right_side})")

        return assignment_statements

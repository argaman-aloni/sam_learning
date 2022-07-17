"""Base class for the unsafe numeric fluents learning approches."""
import itertools
import logging
from abc import ABC
from collections import defaultdict
from typing import Dict, List, NoReturn, Tuple, Union

import numpy
import pandas as pd
from pddl_plus_parser.models import PDDLFunction, Domain, Observation, ObservedComponent

from sam_learning.core import LearnerDomain
from sam_learning.core.learning_types import ConditionType
from sam_learning.core.numeric_function_matcher import NumericFunctionMatcher
from sam_learning.core.numeric_utils import construct_multiplication_strings

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

    def _add_polynomial(self, lifted_fluents: Dict[str, PDDLFunction], dataset: Dict[str, List[float]]) -> NoReturn:
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
                                              intercepts_path: List[float]) -> List[str]:
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
            coefficients_string = self._construct_linear_equation_string(multiplication_functions)
            inequalities.add(f"(>= (+ {coefficients_string} {intercept_value}) 0.0)")

        ordered_inequalities = ["(and"] + list(inequalities) + [")"]
        return ordered_inequalities

    def learn_preconditions(self, positive_observations: List[Observation],
                            negative_observations: List[Observation]) -> Tuple[List[str], ConditionType]:
        """Learning the preconditions using a machine learning approch defined in the inherited class.

        :return: the list of preconditions to be connected with an OR statement.
        """
        raise NotImplemented()

"""Module that stores an action's numeric state fluents and handles its access."""
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, NoReturn, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy
from pddl_plus_parser.models import PDDLFunction
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError

from sam_learning.core.exceptions import NotSafeActionError


class ConditionType(Enum):
    injunctive = 1
    disjunctive = 2


class NumericFluentStateStorage:
    """Stores the numeric state fluents of a single action."""

    logger: logging.Logger
    action_name: str
    previous_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.
    next_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.previous_state_storage = defaultdict(list)
        self.next_state_storage = defaultdict(list)

    def _construct_multipliction_strings(self, coefficients_vector: np.ndarray,
                                         function_variables: List[str]) -> List[str]:
        """Constructs the strings representing the multiplications of the function variables with the coefficient.

        :param coefficients_vector: the coefficient that multiplies the function vector.
        :param function_variables: the name of the numeric fluents that are being used.
        :return: the representation of the fluents multiplied by the coefficients.
        """
        return [f"(* {func} {coefficient})" for func, coefficient in zip(function_variables, coefficients_vector)]

    def _construct_linear_equation_string(self, multiplication_parts: List[str]) -> str:
        """Construct the addition parts of the linear equation string.

        :param multiplication_parts: the multiplication function strings that are multiplied by the coefficient.
        :return: the string representing the sum of the linear variables.
        """
        if len(multiplication_parts) == 1:
            return multiplication_parts[0]

        inner_layer = self._construct_linear_equation_string(multiplication_parts[1:])
        return f"(+ {multiplication_parts[0]} {inner_layer})"

    def _solve_function_linear_equations(self, values_matrix: np.ndarray,
                                         function_post_values: np.ndarray) -> np.ndarray:
        """Solves the linear equations using a matrix form.

        Note: the equation Ax=b is solved as: x = inverse(A)*b.

        :param values_matrix: the A matrix that contains the previous values of the function variables.
        :param function_post_values: the resulting values after the linear change.
        :return: the vector representing the coefficients for the function variables.
        """
        try:
            padded_matrix = np.pad(values_matrix, [(0, 0), (0, 1)], constant_values=1)
            num_dimensions = padded_matrix.shape[1]
            _, indices = sympy.Matrix(padded_matrix).T.rref()
            independent_matrix_rows = np.array([padded_matrix[index] for index in indices])
            independent_next_values = np.array([function_post_values[index] for index in indices])

            A = independent_matrix_rows[:num_dimensions]
            b = independent_next_values[:num_dimensions]

            return np.linalg.solve(A, b)

        except np.LinAlgError:
            failure_reason = "Could not solve input equations, thus making the action unsafe to use."
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason)

    def _convert_to_array_format(self, storage_name: str) -> np.ndarray:
        """Converts the storage to a numpy array format so that scipy functions would be able to use it.

        :param storage_name: the name of the storage to take the values from.
        :return: the array containing the functions' values.
        """
        if storage_name == "previous_state":
            storage = self.previous_state_storage
        else:
            storage = self.next_state_storage

        array = list(map(list, itertools.zip_longest(*storage.values(), fillvalue=None)))
        return np.array(array)

    def _construct_pddl_inequality_scheme(self, coefficient_matrix: np.ndarray, border_points: np.ndarray) -> List[str]:
        """Construct the inequality strings in the appropriate PDDL format.

        :param coefficient_matrix: the matrix containing the coefficient vectors for each inequality.
        :param border_points: the convex hull point which ensures that Ax <= b.
        :return: the inequalities PDDL formatted strings.
        """
        inequalities = []
        for inequality_coefficients, border_point in zip(coefficient_matrix, border_points):
            multiplication_functions = self._construct_multipliction_strings(inequality_coefficients,
                                                                             list(self.previous_state_storage.keys()))
            constructed_left_side = self._construct_linear_equation_string(multiplication_functions)
            inequalities.append(f"(<= {constructed_left_side} {border_point})")

        return inequalities

    def _create_disjunctive_preconditions(self, previous_state_matrix: np.ndarray) -> Tuple[List[str], ConditionType]:
        """Create the disjunctive representation of the preconditions.

        :param previous_state_matrix: the matrix containing the previous state values.
        :return: a disjunctive representation for the precondition in case a convex hull cannot be created.
        """
        functions_equality_strings = []
        if previous_state_matrix.shape[0] == 1:
            self.logger.debug("There is only one state, creating a single precondition")
            for function_variable, value in zip(self.previous_state_storage, previous_state_matrix[0]):
                functions_equality_strings.append(f"(= {function_variable} {value})")

            concatenated_str = " ".join(functions_equality_strings)
            return [concatenated_str], ConditionType.injunctive

        injunctive_conditions = []
        for state_values in previous_state_matrix:
            for function_variable, value in zip(self.previous_state_storage, state_values):
                functions_equality_strings.append(f"(= {function_variable} {value})")

            concatenated_str = " ".join(functions_equality_strings)
            injunctive_conditions.append(f"(and {concatenated_str})")
            functions_equality_strings = []

        return injunctive_conditions, ConditionType.disjunctive

    def _create_convex_hull_linear_inequalities(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :param points: the points that represent the values of the function in the states of the observations.
        :return: the matrix representing the inequalities of the planes created by the convex hull.

        Note: the returned values represents the linear inequalities of the convex hull, i.e.,  Ax <= b.
        """
        try:
            hull = ConvexHull(points)
            num_dimensions = points.shape[1]
            if num_dimensions == 2:
                _ = convex_hull_plot_2d(hull)
                plt.show()

            A = hull.equations[:, :num_dimensions]
            b = -hull.equations[:, num_dimensions]
            return A, b

        except QhullError:
            failure_reason = "Convex hull encountered an error condition and no solution was found"
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason)

    def add_to_previous_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> NoReturn:
        """Adds the matched lifted state fluents to the previous state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        for state_fluent_lifted_str, state_fluent_data in state_fluents.items():
            self.previous_state_storage[state_fluent_lifted_str].append(state_fluent_data.value)

    def add_to_next_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> NoReturn:
        """Adds the matched lifted state fluents to the next state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        for state_fluent_lifted_str, state_fluent_data in state_fluents.items():
            self.next_state_storage[state_fluent_lifted_str].append(state_fluent_data.value)

    def filter_out_inconsistent_state_variables(self) -> NoReturn:
        """

        :return:
        """
        max_function_len = max([len(values) for values in self.previous_state_storage.values()])
        self.previous_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                       self.previous_state_storage.items() if len(state_values) == max_function_len}
        self.next_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                   self.next_state_storage.items() if len(state_values) == max_function_len}

    def construct_safe_linear_inequalities(self) -> Tuple[List[str], ConditionType]:
        """

        :return:
        """
        num_required_dimensions = len(self.previous_state_storage) + 1
        previous_state_matrix = self._convert_to_array_format("previous_state")
        if previous_state_matrix.shape[0] < num_required_dimensions:
            return self._create_disjunctive_preconditions(previous_state_matrix)

        A, b = self._create_convex_hull_linear_inequalities(previous_state_matrix)
        inequalities_strs = self._construct_pddl_inequality_scheme(A, b)
        return inequalities_strs, ConditionType.injunctive

    def construct_assignment_equations(self) -> List[str]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :return: the constructed assignment statements.
        """
        assignment_statements = []
        for lifted_function, next_state_values in self.next_state_storage.items():
            # check if the action changed the value from the previous state at all.
            if not any([(next_value - prev_value) != 0 for
                        prev_value, next_value in zip(self.previous_state_storage[lifted_function],
                                                      next_state_values)]):
                continue

            # check if all of the values consist of a change to a constant value C.
            if len(set(self.next_state_storage[lifted_function])) <= 1:
                const_assigned_value = self.next_state_storage[lifted_function][0]
                return [f"(assign {lifted_function} {const_assigned_value})"]

            self._validate_safe_equation_solving(lifted_function)

            function_post_values = np.array(next_state_values)
            values_matrix = self._convert_to_array_format("previous_state")
            coefficient_vector = self._solve_function_linear_equations(values_matrix, function_post_values)

            functions_including_dummy = list(self.previous_state_storage.keys()) + ["(dummy)"]
            multiplication_functions = self._construct_multipliction_strings(
                coefficient_vector, functions_including_dummy)
            constructed_right_side = self._construct_linear_equation_string(multiplication_functions)
            assignment_statements.append(f"(assign {lifted_function} {constructed_right_side})")

        return assignment_statements

    def _validate_safe_equation_solving(self, lifted_function):
        num_variables = len(self.next_state_storage)
        num_equations = len(self.previous_state_storage[lifted_function])
        # validate that it is possible to solve linear equations at all.
        if num_equations < num_variables + 1:
            failure_reason = "Cannot solve linear equations when too little input equations given."
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason)

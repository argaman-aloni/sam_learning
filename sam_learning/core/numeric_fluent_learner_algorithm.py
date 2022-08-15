"""Module that stores amd learns an action's numeric state fluents."""
import itertools
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NoReturn, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy
from pddl_plus_parser.models import PDDLFunction
from scipy import stats
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.qhull import QhullError
from sklearn.linear_model import LinearRegression

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import EquationSolutionType, ConditionType
from sam_learning.core.numeric_utils import prettify_coefficients, construct_multiplication_strings, \
    construct_linear_equation_string, construct_non_circular_assignment

EPSILON = 1e-10
LEGAL_LEARNING_SCORE = 1.00
np.seterr(divide='ignore', invalid='ignore')


class NumericFluentStateStorage:
    """Stores and learned the numeric state fluents of a single action."""

    logger: logging.Logger
    action_name: str
    previous_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.
    next_state_storage: Dict[str, List[float]]  # lifted function str -> numeric values.

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.previous_state_storage = defaultdict(list)
        self.next_state_storage = defaultdict(list)
        # TODO: remove this once the action is fully tested.
        self.convex_hull_error_file_path = Path(os.environ["CONVEX_HULL_ERROR_PATH"])

    def _validate_legal_equations(self, values_matrix: np.ndarray) -> NoReturn:
        """Validates that there are enough independent equations which enable for a single solution for the equation.

        :param values_matrix: the matrix constructed based on the observations.
        """
        num_dimensions = values_matrix.shape[1]
        _, indices = sympy.Matrix(values_matrix).T.rref()
        independent_rows_matrix = np.array([values_matrix[index] for index in indices])
        if independent_rows_matrix.shape[0] >= num_dimensions:
            return

        failure_reason = f"There are too few independent rows of data! " \
                         f"cannot solve linear equations for action - {self.action_name}!"
        self.logger.warning(failure_reason)
        raise NotSafeActionError(self.action_name, failure_reason, EquationSolutionType.not_enough_data)

    def _validate_safe_equation_solving(self, lifted_function: str) -> NoReturn:
        """Conducts a first validation on whether it is safe to solve the linear equations.

        :param lifted_function: the number of the function to be validated.
        """
        num_variables = len(self.previous_state_storage)
        num_equations = len(self.next_state_storage[lifted_function])
        # validate that it is possible to solve linear equations at all.
        if num_equations < num_variables or num_equations == num_variables == 1:
            failure_reason = f"Cannot solve linear equations when too little input equations given for " \
                             f"action - {self.action_name}."
            self.logger.warning(failure_reason)
            raise NotSafeActionError(
                self.action_name, failure_reason, EquationSolutionType.not_enough_data)

    def _solve_function_linear_equations(self, values_matrix: np.ndarray,
                                         function_post_values: np.ndarray) -> Tuple[List[float], float]:
        """Solves the linear equations using a matrix form.

        Note: the equation Ax=b is solved as: x = inverse(A)*b.

        :param values_matrix: the A matrix that contains the previous values of the function variables.
        :param function_post_values: the resulting values after the linear change.
        :return: the vector representing the coefficients for the function variables and the learning score (R^2).
        """
        regressor = LinearRegression()
        regressor.fit(values_matrix, function_post_values)
        learning_score = regressor.score(values_matrix, function_post_values)
        if learning_score < LEGAL_LEARNING_SCORE:
            reason = "The learned effects are not safe since the R^2 is not high enough."
            self.logger.warning(reason)
            raise NotSafeActionError(self.action_name, reason, EquationSolutionType.no_solution_found)

        coefficients = list(regressor.coef_) + [regressor.intercept_]
        coefficients = prettify_coefficients(coefficients)
        return coefficients, learning_score

    def _convert_to_array_format(self, storage_name: str, relevant_fluents: Optional[List[str]] = None,
                                 should_filter_repetitive_values: bool = True) -> np.ndarray:
        """Converts the storage to a numpy array format so that scipy functions would be able to use it.

        :param storage_name: the name of the storage to take the values from.
        :param relevant_fluents: the relevant fluents to take the values from.
        :param should_filter_repetitive_values: whether to filter out the repetitive values.
        :return: the array containing the functions' values.
        """
        if storage_name != "previous_state":
            storage = self.next_state_storage

        elif relevant_fluents is not None:
            storage_fluents = [fluent for fluent in relevant_fluents if fluent in self.previous_state_storage]
            storage = {fluent: self.previous_state_storage[fluent] for fluent in storage_fluents}

        else:
            storage = self.previous_state_storage

        array = list(map(list, itertools.zip_longest(*storage.values(), fillvalue=None)))
        if should_filter_repetitive_values:
            return np.unique(np.array(array), axis=0)

        return np.array(array)

    def _construct_pddl_inequality_scheme(self, coefficient_matrix: np.ndarray, border_points: np.ndarray,
                                          relevant_fluents: Optional[List[str]] = None) -> List[str]:
        """Construct the inequality strings in the appropriate PDDL format.

        :param coefficient_matrix: the matrix containing the coefficient vectors for each inequality.
        :param border_points: the convex hull point which ensures that Ax <= b.
        :param relevant_fluents: the fluents relevant to the creation of the preconditions if exists, if not,
            should be ALL the previous state variables.
        :return: the inequalities PDDL formatted strings.
        """
        inequalities = set()
        for inequality_coefficients, border_point in zip(coefficient_matrix, border_points):
            function_names = relevant_fluents or self.previous_state_storage.keys()
            multiplication_functions = construct_multiplication_strings(inequality_coefficients, function_names)
            constructed_left_side = construct_linear_equation_string(multiplication_functions)
            inequalities.add(f"(<= {constructed_left_side} {border_point})")

        return list(inequalities)

    def _create_disjunctive_preconditions(
            self, previous_state_matrix: np.ndarray,
            equality_conditions=None) -> Tuple[List[str], ConditionType]:
        """Create the disjunctive representation of the preconditions.

        :param previous_state_matrix: the matrix containing the previous state values.
        :return: a disjunctive representation for the precondition in case a convex hull cannot be created.
        """
        if equality_conditions is None:
            equality_conditions = []
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
            if equality_conditions is not None and len(equality_conditions) > 0:
                concatenated_str += " " + " ".join(equality_conditions)

            injunctive_conditions.append(f"(and {concatenated_str})")
            functions_equality_strings = []

        return injunctive_conditions, ConditionType.disjunctive

    def _create_convex_hull_linear_inequalities(self, points: np.ndarray,
                                                display_mode: bool = False) -> tuple[List[List[float]], List[float]]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :param points: the points that represent the values of the function in the states of the observations.
        :return: the matrix representing the inequalities of the planes created by the convex hull.

        Note: the returned values represents the linear inequalities of the convex hull, i.e.,  Ax <= b.
        """
        try:
            hull = ConvexHull(points)
            num_dimensions = points.shape[1]
            self._display_convex_hull(display_mode, hull, num_dimensions)
            A = hull.equations[:, :num_dimensions]
            b = -hull.equations[:, num_dimensions]
            return [prettify_coefficients(row) for row in A], prettify_coefficients(b)

        except (QhullError, ValueError) as e:
            with open(self.convex_hull_error_file_path, "at") as error_file:
                error_file.write(f"{e}\n")

            failure_reason = "Convex hull encountered an error condition and no solution was found"
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason, EquationSolutionType.convex_hull_not_found)

    def _display_convex_hull(self, display_mode: bool, hull: ConvexHull, num_dimensions: int) -> NoReturn:
        """Displays the convex hull in as a plot.

        :param display_mode: whether to display the plot.
        :param hull: the convex hull to display.
        :param num_dimensions: the number of dimensions of the original data.
        """
        if num_dimensions == 2 and display_mode:
            _ = convex_hull_plot_2d(hull)
            plt.title(f"{self.action_name} - convex hull")
            plt.show()

    def _remove_duplicated_variables(self, silent: bool = False) -> Dict[str, str]:
        """removes variables that are basically duplication of other variables. This happens in some domains.

        :param silent: whether to print the removed variables.
        :return: the mapping between the removed function to the one that is identical to it.
        """
        duplicated_numeric_functions = []
        duplicate_map = {}
        for function1, function2 in itertools.combinations(self.previous_state_storage, 2):
            if self.previous_state_storage[function1] == self.previous_state_storage[function2]:
                duplicated_numeric_functions.append(function2)
                duplicate_map[function2] = function1

        if not silent:
            for func in duplicated_numeric_functions:
                self.previous_state_storage.pop(func, None)

        return duplicate_map

    def _construct_single_dimension_inequalities(self, relevant_fluent: str, equality_strs: List[str] = []) -> Tuple[
        List[str], ConditionType]:
        """Construct a single dimension precondition representation.

        :param relevant_fluent: the fluent only fluent that is relevant to the preconditions' creation.
        :param equality_strs: the equality conditions that are already present in the preconditions.
        :return: the preconditions string and the condition type.
        """
        min_value = min(self.previous_state_storage.get(relevant_fluent, [0]))
        max_value = max(self.previous_state_storage.get(relevant_fluent, [0]))
        conditions = [f"(>= {relevant_fluent} {min_value})", f"(<= {relevant_fluent} {max_value})"]
        conditions.extend(equality_strs)
        return conditions, ConditionType.injunctive

    def _filter_constant_features(self, previous_state_matrix: np.ndarray, relevant_fluents: List[str]) -> Tuple[
        np.ndarray, List[str], List[str]]:
        """Filters out fluents that contain only constant values since they do not contribute to the convex hull.

        :param previous_state_matrix: the matrix of the previous state values.
        :return: the filtered matrix and the equality strings, i.e. the strings of the values that should be equal.
        """
        self.logger.info("Filtering out features that are equal to a constant value...")
        equal_columns = np.all(previous_state_matrix == previous_state_matrix[0, :], axis=0)
        columns_to_remove = [i for i, is_equal in enumerate(equal_columns) if is_equal]
        if len(columns_to_remove) == 0:
            self.logger.debug("No columns with only single constant value found found!")
            return previous_state_matrix, [], []

        equal_fluent_strs = []
        self.logger.debug(f"The columns {columns_to_remove} is containing only constant values, removing it.")
        removed_fluents = []
        for column_index in columns_to_remove:
            fluent_name = relevant_fluents[column_index]
            equal_fluent_strs.append(f"(= {fluent_name} {self.previous_state_storage[fluent_name][0]})")
            removed_fluents.append(fluent_name)

        filtered_matrix = np.delete(previous_state_matrix, columns_to_remove, axis=1)
        self.logger.info(f"Removed the fluents {removed_fluents}.")
        return filtered_matrix, equal_fluent_strs, removed_fluents

    @staticmethod
    def _extract_numeric_linear_coefficient(function1_values: List[float], function2_values: List[float]) -> float:
        """Extract the first real numeric value from the two lists of values since there might be NaN values.

        :param function1_values: the values of the first function.
        :param function2_values: the values of the second function.
        :return: the first numeric divisor.
        """
        linear_coeff = 0
        division_res = (np.array(function1_values) / np.array(function2_values))
        for value in division_res:
            if not math.isnan(value):
                linear_coeff = value
                break

        return linear_coeff

    def _detect_linear_dependent_features(self, previous_state_matrix: np.ndarray,
                                          remained_fluents: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Removes features that are identical to one another since they prevent from the convex hull from working.

        Notice:
            This method does not ignore the identical features but adds a new condition stating that these
            features must be equal.

        :param previous_state_matrix: the matrix of the previous state values.
        :param remained_fluents: the list of fluents remained after removing constant features.
        :return: the matrix after the filtering process ended and the filtered matrix and the equality strings, i.e.
            the strings of the values that are identical.
        """
        self.logger.info(f"Detecting linear dependent features and removing them for action - {self.action_name}")
        linear_dependent_fluent_strs = []
        fluents_indexes_to_remove = []
        removed_fluents = []
        filtered_matrix = previous_state_matrix.copy()
        for function1, function2 in itertools.combinations(self.previous_state_storage, 2):
            if function1 not in remained_fluents or function2 not in remained_fluents:
                continue

            function1_values = self.previous_state_storage[function1]
            function2_values = self.previous_state_storage[function2]
            correlation_coeff, p_value = stats.pearsonr(function1_values, function2_values)
            if abs(correlation_coeff - 1) <= EPSILON:
                self.logger.debug(f"The two functions {function1} and {function2} are linearly dependent.")
                # extracting the linear dependency parameter
                linear_coeff = self._extract_numeric_linear_coefficient(function1_values, function2_values)
                linear_dependent_fluent_strs.append(f"(= {function1} (* {linear_coeff} {function2}))")
                fluents_indexes_to_remove.append(remained_fluents.index(function2))
                removed_fluents.append(function2)

        if len(fluents_indexes_to_remove) > 0:
            filtered_matrix = np.delete(previous_state_matrix, fluents_indexes_to_remove, axis=1)

        return filtered_matrix, linear_dependent_fluent_strs, removed_fluents

    def _filter_all_convex_hull_inconsistencies(
            self, previous_state_matrix: np.ndarray,
            relevant_fluents: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
        """Filters out features that might prevent from the convex hull algorithm to work properly.

        :param previous_state_matrix: the matrix of the previous state containing numeric values.
        :param relevant_fluents: the fluents that are relevant for the current action.
        :return: the equality strings, the filtered matrix and the remaining fluents that will be used to create
         the convex hull.
        """
        non_convexed_conditions = []
        self.logger.debug(f"Now checking to see if there are columns that linearly dependent.")
        if previous_state_matrix.shape[0] == 1:
            self.logger.debug("There is only one sample of information to learn from, "
                              "not enough to learn of linear dependency!")
            return [], previous_state_matrix, relevant_fluents

        no_constant_columns_matrix, equality_strs, removed_fluents = self._filter_constant_features(
            previous_state_matrix, relevant_fluents)
        relevant_fluents = [fluent for fluent in relevant_fluents if fluent not in removed_fluents]
        filtered_previous_state_matrix, column_equality_strs, removed_fluents = \
            self._detect_linear_dependent_features(no_constant_columns_matrix, relevant_fluents)
        remained_fluents = [fluent for fluent in relevant_fluents if fluent not in removed_fluents]
        equality_strs.extend(column_equality_strs)
        if len(equality_strs) > 0:
            non_convexed_conditions.extend(equality_strs)

        return non_convexed_conditions, filtered_previous_state_matrix, remained_fluents

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
            if len(self.previous_state_storage.get(state_fluent_lifted_str, [])) != \
                    len(self.next_state_storage[state_fluent_lifted_str]):
                self.logger.debug("This is a case where effects create new fluents - should adjust the previous state.")
                self.previous_state_storage[state_fluent_lifted_str].append(0)

    def filter_out_inconsistent_state_variables(self) -> NoReturn:
        """Filters out fluents that appear only in part of the states since they are not safe.

        :return: only the safe state variables that appear in *all* states.
        """
        max_function_len = max([len(values) for values in self.previous_state_storage.values()])
        self.previous_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                       self.previous_state_storage.items() if len(state_values) == max_function_len}
        self.next_state_storage = {lifted_function: state_values for lifted_function, state_values in
                                   self.next_state_storage.items() if len(state_values) == max_function_len}

    def construct_safe_linear_inequalities(
            self, relevant_fluents: Optional[List[str]] = None) -> Tuple[List[str], ConditionType]:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        if relevant_fluents is None:
            relevant_fluents = self.previous_state_storage.keys()

        if len(relevant_fluents) == 1:
            self.logger.debug("Only one dimension is needed in the preconditions!")
            return self._construct_single_dimension_inequalities(relevant_fluents[0])

        num_required_dimensions = len(relevant_fluents) + 1 if relevant_fluents is not None else \
            len(self.previous_state_storage.keys()) + 1

        previous_state_matrix = self._convert_to_array_format("previous_state", relevant_fluents)
        equality_strs, filtered_previous_state_matrix, remained_fluents = self._filter_all_convex_hull_inconsistencies(
            previous_state_matrix, relevant_fluents)

        if filtered_previous_state_matrix.shape[0] < num_required_dimensions:
            return self._create_disjunctive_preconditions(filtered_previous_state_matrix, equality_strs)

        if filtered_previous_state_matrix.shape[1] < 2:
            return self._construct_single_dimension_inequalities(remained_fluents[0], equality_strs)

        A, b = self._create_convex_hull_linear_inequalities(filtered_previous_state_matrix, display_mode=False)
        inequalities_strs = self._construct_pddl_inequality_scheme(A, b, remained_fluents)
        inequalities_strs.extend(equality_strs)

        return inequalities_strs, ConditionType.injunctive

    def construct_assignment_equations(self) -> List[str]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :return: the constructed assignment statements.
        """
        self.logger.info(f"Constructing the fluent assignment equations for action {self.action_name}.")
        assignment_statements = []
        duplicate_map = self._remove_duplicated_variables()
        for lifted_function, next_state_values in self.next_state_storage.items():
            self._validate_safe_equation_solving(lifted_function)
            function_post_values = np.array(next_state_values)
            values_matrix = self._convert_to_array_format("previous_state", should_filter_repetitive_values=False)
            self._validate_legal_equations(values_matrix)
            self.logger.debug("After validating that the learning process is safe then trying to see if the "
                              "action affects the numeric fluent.")

            searched_function = lifted_function if lifted_function not in duplicate_map else duplicate_map[
                lifted_function]
            if not any([(next_val - prev_val) != 0 for
                        prev_val, next_val in zip(self.previous_state_storage[searched_function], next_state_values)]):
                self.logger.debug(f"The action {self.action_name} does not affect the fluent - {lifted_function}")
                continue

            coefficient_vector, learning_score = self._solve_function_linear_equations(values_matrix,
                                                                                       function_post_values)
            self.logger.debug(f"Learned the coefficients for the numeric equations with r^2 score of {learning_score}")

            functions_including_dummy = list(self.previous_state_storage.keys()) + ["(dummy)"]
            if coefficient_vector[list(self.previous_state_storage.keys()).index(searched_function)] != 0:
                self.logger.debug("the assigned party is a part of the equation, "
                                  "cannot use circular dependency so changing the format!")
                coefficients_map = {lifted_func: coef for lifted_func, coef in
                                    zip(functions_including_dummy, coefficient_vector)}
                if lifted_function not in coefficients_map:
                    coefficients_map[lifted_function] = coefficients_map[searched_function]

                assignment_statements.append(
                    construct_non_circular_assignment(lifted_function,
                                                      coefficients_map,
                                                      self.previous_state_storage[searched_function][0],
                                                      self.next_state_storage[lifted_function][0]))
                continue

            multiplication_functions = construct_multiplication_strings(
                coefficient_vector, functions_including_dummy)
            if len(multiplication_functions) == 0:
                self.logger.debug("The algorithm designated a vector of zeros to the equation "
                                  "which means that there are not coefficients. Continuing.")
                continue

            constructed_right_side = construct_linear_equation_string(multiplication_functions)
            assignment_statements.append(f"(assign {lifted_function} {constructed_right_side})")

        return assignment_statements

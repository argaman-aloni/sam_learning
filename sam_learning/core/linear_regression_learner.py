"""This module contains the LinearRegressionLearner class."""
import logging
from typing import Optional, List, Dict, Tuple, Union, Set

import numpy as np
import sympy
from pandas import DataFrame
from pddl_plus_parser.models import Precondition, NumericalExpressionTree, PDDLFunction, ConditionalEffect
from sklearn.linear_model import LinearRegression

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import EquationSolutionType, ConditionType
from sam_learning.core.numeric_utils import detect_linear_dependent_features, construct_linear_equation_string, \
    construct_non_circular_assignment, construct_multiplication_strings, prettify_coefficients, \
    construct_numeric_conditions, construct_numeric_effects

LABEL_COLUMN = "label"
NEXT_STATE_PREFIX = "next_state_"  # prefix for the next state variables.
SUBTRACTION_COLUMN = "subtraction"
POST_NEXT_STATE_PREFIX_INDEX = len(NEXT_STATE_PREFIX)

LEGAL_LEARNING_SCORE = 1.00


class LinearRegressionLearner:

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.domain_functions = domain_functions

    def _validate_legal_equations(self, values_df: DataFrame, allow_unsafe_learning: bool = True) -> None:
        """Validates that there are enough independent equations which enable for a single solution for the equation.

        :param values_df: the matrix constructed based on the observations.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        """
        values_matrix = values_df.to_numpy()
        num_dimensions = values_matrix.shape[1] + 1
        num_rows = values_matrix.shape[0]
        values_matrix_with_bias = np.c_[values_matrix, np.ones(num_rows)]
        _, indices = sympy.Matrix(values_matrix_with_bias).T.rref()
        independent_rows_matrix = np.array([values_matrix_with_bias[index] for index in indices])
        if independent_rows_matrix.shape[0] >= num_dimensions:
            return

        failure_reason = f"There are too few independent rows of data! " \
                         f"cannot solve linear equations for action - {self.action_name}!"
        self.logger.warning(failure_reason)
        if allow_unsafe_learning:
            return

        raise NotSafeActionError(self.action_name, failure_reason, EquationSolutionType.not_enough_data)

    def _solve_function_linear_equations(
            self, values_matrix: np.ndarray, function_post_values: np.ndarray,
            allow_unsafe_learning: bool = True) -> Tuple[List[float], float]:
        """Solves the linear equations using a matrix form.

        Note: the equation Ax=b is solved as: x = inverse(A)*b.

        :param values_matrix: the A matrix that contains the previous values of the function variables.
        :param function_post_values: the resulting values after the linear change.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        :return: the vector representing the coefficients for the function variables and the learning score (R^2).
        """
        regressor = LinearRegression()
        regressor.fit(values_matrix, function_post_values)
        learning_score = regressor.score(values_matrix, function_post_values)
        if learning_score < LEGAL_LEARNING_SCORE:
            reason = "The learned effects are not safe since the R^2 is not high enough."
            self.logger.warning(reason)
            if not allow_unsafe_learning:
                raise NotSafeActionError(self.action_name, reason, EquationSolutionType.no_solution_found)

        coefficients = list(regressor.coef_) + [regressor.intercept_]
        coefficients = prettify_coefficients(coefficients)
        return coefficients, learning_score

    def _compute_non_constant_change(
            self, lifted_function: str, regression_matrix: DataFrame, allow_unsafe_learning: bool = True) -> Optional[
        str]:
        """Computes the change in the function value based on the previous values of the function.

        Note: We assume in this stage that the change results from a polynomial function of the previous values.

        :param lifted_function: the function to compute the change for.
        :param regression_matrix: the matrix containing the previous values of the function and the resulting values.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        :return: the string representing the polynomial function change in PDDL+ format.
        """
        regression_array = np.array(regression_matrix.loc[:, regression_matrix.columns != LABEL_COLUMN])
        function_post_values = np.array(regression_matrix[LABEL_COLUMN])
        coefficient_vector, learning_score = self._solve_function_linear_equations(
            regression_array, function_post_values, allow_unsafe_learning)
        self.logger.debug(f"Learned the coefficients for the numeric equations with r^2 score of {learning_score}")

        functions_and_dummy = list(regression_matrix.columns[:-1]) + ["(dummy)"]
        if lifted_function in regression_matrix.columns and coefficient_vector[
            list(regression_matrix.columns).index(lifted_function)] != 0:
            self.logger.debug("the assigned party is a part of the equation, "
                              "cannot use circular dependency so changing the format!")
            coefficients_map = {lifted_func: coef for lifted_func, coef in
                                zip(functions_and_dummy, coefficient_vector)}
            return construct_non_circular_assignment(lifted_function,
                                                     coefficients_map,
                                                     regression_matrix[lifted_function][0],
                                                     regression_matrix[LABEL_COLUMN][0])

        multiplication_functions = construct_multiplication_strings(coefficient_vector, functions_and_dummy)
        if len(multiplication_functions) == 0:
            self.logger.debug("The algorithm designated a vector of zeros to the equation "
                              "which means that there are not coefficients. Continuing.")
            return None

        constructed_right_side = construct_linear_equation_string(multiplication_functions)
        return f"(assign {lifted_function} {constructed_right_side})"

    def _construct_safe_conditional_effect(self, combined_data: DataFrame) -> ConditionalEffect:
        """Constructs a safe conditional effect when the there is not enough data to learn the effect.

        :param combined_data: the data frame containing the previous and next state values.
        :return: the assignment statements and the additional conditions.
        """
        assignment_statements = []
        additional_conditions = []
        for fluent in combined_data.columns:
            if fluent.startswith(NEXT_STATE_PREFIX):
                if combined_data[fluent[POST_NEXT_STATE_PREFIX_INDEX:]].iloc[0] != combined_data[fluent].iloc[0]:
                    self.logger.debug("The next state changed from the previous state to the next state.")
                    assignment_statements.append(
                        f"(assign {fluent[POST_NEXT_STATE_PREFIX_INDEX:]} {combined_data[fluent].iloc[0]})")

            else:  # the fluent is belongs to the previous state
                if combined_data[fluent].iloc[0] != 0:
                    self.logger.debug("The next state fluent might be dependent on the current "
                                      "state fluent since it is not zero")
                    additional_conditions.append(f"(= {fluent} {combined_data[fluent].iloc[0]})")

        conditional_effect = ConditionalEffect()
        conditional_effect.antecedents.root = construct_numeric_conditions(
            additional_conditions, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)
        conditional_effect.numeric_effects = construct_numeric_effects(assignment_statements, self.domain_functions)
        return conditional_effect

    def _action_not_affects_fluent(self, lifted_function: str, regression_matrix: DataFrame) -> bool:
        """Validates that the action affects the given lifted function.

        :param lifted_function: the name of the lifted function to check.
        :param regression_matrix: the regression matrix to check.
        :return: Whether the action affects the given lifted function.
        """
        self.logger.debug("Checking if the action affects the given lifted function...")
        subtraction_df = regression_matrix[LABEL_COLUMN] - regression_matrix[lifted_function]
        unique_values = subtraction_df.unique()
        return len(unique_values) == 1 and unique_values[0] == 0

    @staticmethod
    def _combine_states_data(prev_state: Dict[str, List[float]], next_state: Dict[str, List[float]]) -> DataFrame:
        """Combines the previous and next states data into a single dataframe.

        :return: the combined dataframe.
        """
        combined_data = prev_state.copy()
        combined_data.update({f"{NEXT_STATE_PREFIX}{fluent_name}": fluent_values for fluent_name, fluent_values in
                              next_state.items()})
        return DataFrame(combined_data).drop_duplicates()

    @staticmethod
    def _compute_constant_change(lifted_function: str, regression_matrix: DataFrame) -> Optional[str]:
        """Computes a constant change in the function value based on the previous values of the function.

        :param lifted_function: the function to compute the change for.
        :param regression_matrix: the matrix containing the previous values of the function and the resulting values.
        :return: the string representing the constant change in PDDL+ format if it exists.
        """
        unique_values = (regression_matrix[LABEL_COLUMN] - regression_matrix[lifted_function]).unique()
        if len(unique_values) == 1:
            diff = unique_values[0]
            return f"(decrease {lifted_function} {abs(diff)})" if diff < 0 else f"(increase {lifted_function} {diff})"

        return None

    def construct_assignment_equations(
            self, previous_state_data: Dict[str, List[float]],
            next_state_data: Dict[str, List[float]],
            allow_unsafe_learning: bool = True) -> Union[Tuple[
        Set[NumericalExpressionTree], Optional[Precondition]], ConditionalEffect]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :param previous_state_data: the data of the previous state.
        :param next_state_data: the data of the next state.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        :return: the constructed effects with the possibly additional preconditions.
        """
        self.logger.info(f"Constructing the fluent assignment equations for action {self.action_name}.")
        combined_data = self._combine_states_data(previous_state_data, next_state_data)
        if combined_data.shape[0] == 1:
            self.logger.info(f"The action {self.action_name} contains a single unique observation!")
            restrictive_conditional_effect = self._construct_safe_conditional_effect(combined_data)
            return restrictive_conditional_effect

        assignment_statements = []
        additional_conditions = set()
        labels = [f"{NEXT_STATE_PREFIX}{fluent_name}" for fluent_name in next_state_data.keys()]
        features = list(previous_state_data.keys())
        for feature_fluent, label_fluent in zip(features, labels):
            features_df = combined_data.copy()[features]
            regression_matrix, conditions, dependent_columns = detect_linear_dependent_features(
                features_df, feature_fluent)
            additional_conditions.update(conditions)

            self._validate_legal_equations(regression_matrix, allow_unsafe_learning=allow_unsafe_learning)
            regression_matrix[LABEL_COLUMN] = combined_data[label_fluent]
            if self._action_not_affects_fluent(feature_fluent, regression_matrix):
                self.logger.info(f"The action {self.action_name} does not affect the fluent - {feature_fluent}")
                continue

            constant_change = self._compute_constant_change(feature_fluent, regression_matrix)
            if constant_change is not None:
                assignment_statements.append(constant_change)
                continue

            non_constant_change = self._compute_non_constant_change(
                feature_fluent, regression_matrix, allow_unsafe_learning)
            if non_constant_change is not None:
                assignment_statements.append(non_constant_change)
                continue

        if len(additional_conditions) == 0:
            return construct_numeric_effects(assignment_statements, self.domain_functions), None

        return construct_numeric_effects(assignment_statements, self.domain_functions), \
            construct_numeric_conditions(list(additional_conditions), ConditionType.conjunctive, self.domain_functions)

"""This module contains the LinearRegressionLearner class."""
import logging
from typing import Optional, List, Dict, Tuple, Union, Set

import numpy as np
import sympy
from multipledispatch import dispatch
from pandas import DataFrame, Series
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

    @staticmethod
    def _combine_states_data(prev_state: Dict[str, List[float]], next_state: Dict[str, List[float]]) -> DataFrame:
        """Combines the previous and next states data into a single dataframe.

        :return: the combined dataframe.
        """
        combined_data = prev_state.copy()
        combined_data.update({f"{NEXT_STATE_PREFIX}{fluent_name}": fluent_values for fluent_name, fluent_values in
                              next_state.items()})
        dataframe = DataFrame(combined_data).fillna(0)
        dataframe.drop_duplicates(inplace=True)
        return dataframe

    @staticmethod
    def _remove_constant_zero_fluents(combined_data: DataFrame) -> Tuple[DataFrame, List[str], List[str]]:
        """Removes the fluents that have a constant value of zero from the combined data.

        :param combined_data: the combined data to remove the fluents from.
        :return: the combined data without the zero fluents and the list of the removed fluents as additional
            preconditions to the action.
        """
        zero_fluents = []
        removed_fluents = []
        filtered_data_df = combined_data.copy()
        for fluent in filtered_data_df.columns:
            if fluent.startswith(NEXT_STATE_PREFIX):
                continue

            column_values = filtered_data_df[fluent]
            if all(value == 0 for value in column_values):
                zero_fluents.append(f"(= {fluent} 0)")
                filtered_data_df.drop(columns=[fluent], inplace=True)
                removed_fluents.append(fluent)

        return filtered_data_df, zero_fluents, removed_fluents

    def _validate_legal_equations(self, values_df: DataFrame) -> bool:
        """Validates that there are enough independent equations which enable for a single solution for the equation.

        :param values_df: the matrix constructed based on the observations.
        """
        values_matrix = values_df.to_numpy()
        num_dimensions = values_matrix.shape[1] + 1  # +1 for the bias.
        num_rows = values_matrix.shape[0]
        values_matrix_with_bias = np.c_[values_matrix, np.ones(num_rows)]
        _, pivot_cols = sympy.Matrix(values_matrix_with_bias).rref()
        if len(pivot_cols) >= num_dimensions:
            return True

        failure_reason = f"There are too few independent rows of data! " \
                         f"cannot solve linear equations for action - {self.action_name}!"
        self.logger.warning(failure_reason)

        return False

    def _solve_regression_problem(self, values_matrix: np.ndarray, function_post_values: np.ndarray,
                                  allow_unsafe_learning: bool = True) -> Tuple[List[float], float]:
        """Solves the polynomial equations using a matrix form.

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

    def _solve_safe_independent_equations(
            self, lifted_function: str, regression_df: DataFrame, allow_unsafe_learning: bool = False) -> Optional[
        str]:
        """Computes the change in the function value based on the previous values of the function.

        Note: We assume in this stage that the change results from a polynomial function of the previous values.

        :param lifted_function: the function to compute the change for.
        :param regression_df: the matrix containing the previous values of the function and the resulting values.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        :return: the string representing the polynomial function change in PDDL+ format.
        """
        regression_array = np.array(regression_df.loc[:, regression_df.columns != LABEL_COLUMN])
        function_post_values = np.array(regression_df[LABEL_COLUMN])
        coefficient_vector, learning_score = self._solve_regression_problem(
            regression_array, function_post_values, allow_unsafe_learning)
        self.logger.debug(f"Learned the coefficients for the numeric equations with r^2 score of {learning_score}")

        functions_and_dummy = list(regression_df.columns[:-1]) + ["(dummy)"]
        if lifted_function in regression_df.columns and coefficient_vector[
            list(regression_df.columns).index(lifted_function)] != 0:
            self.logger.debug("the assigned party is a part of the equation, "
                              "cannot use circular dependency so changing the format!")
            coefficients_map = {lifted_func: coef for lifted_func, coef in
                                zip(functions_and_dummy, coefficient_vector)}
            return construct_non_circular_assignment(lifted_function,
                                                     coefficients_map,
                                                     regression_df[lifted_function][0],
                                                     regression_df[LABEL_COLUMN][0])

        multiplication_functions = construct_multiplication_strings(coefficient_vector, functions_and_dummy)
        if len(multiplication_functions) == 0:
            self.logger.debug("The algorithm designated a vector of zeros to the equation "
                              "which means that there are not coefficients. Continuing.")
            return None

        constructed_right_side = construct_linear_equation_string(multiplication_functions)
        return f"(assign {lifted_function} {constructed_right_side})"

    def _construct_restrictive_effect_from_textual_data(
            self, additional_conditions: List[str],
            assignment_statements: List[str]) -> Optional[Tuple[Optional[ConditionalEffect], Precondition]]:
        """Constructs a conditional effect from the textual data.

        :param additional_conditions: the added preconditions and the antecedents to the conditional effects.
        :param assignment_statements: the assignment statements to add to the conditional effect.
        :return: the conditional effect and the restrictive precondition.
        """
        if len(assignment_statements) == 0:
            self.logger.debug("The algorithm could not find any changes in the state variables. Continuing.")
            return None, construct_numeric_conditions(
                additional_conditions, ConditionType.conjunctive, self.domain_functions)

        conditional_effect = ConditionalEffect()
        conditional_effect.antecedents.root = construct_numeric_conditions(
            additional_conditions, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)
        conditional_effect.numeric_effects = construct_numeric_effects(assignment_statements, self.domain_functions)
        restrictive_precondition = construct_numeric_conditions(
            additional_conditions, ConditionType.conjunctive, self.domain_functions)
        return conditional_effect, restrictive_precondition

    @dispatch(Series)
    def _construct_safe_conditional_effect_for_single_sample(
            self, sample_data: Series) -> Optional[Tuple[Optional[ConditionalEffect], Precondition]]:
        """Constructs a conditional effect for a single sample.

        :param sample_data: the sample to construct the conditional effect for.
        :return: the conditional effect and the precondition for the conditional effect.
        """
        assignment_statements = []
        additional_conditions = []
        for fluent in sample_data.index.tolist():
            if fluent.startswith(NEXT_STATE_PREFIX):
                if sample_data[fluent[POST_NEXT_STATE_PREFIX_INDEX:]] != sample_data[fluent]:
                    self.logger.debug("The value of the fluent changed by applying the action on the previous state.")
                    assignment_statements.append(
                        f"(assign {fluent[POST_NEXT_STATE_PREFIX_INDEX:]} {sample_data[fluent]})")

            else:  # the fluent is belongs to the previous state
                additional_conditions.append(f"(= {fluent} {sample_data[fluent]})")

        self.logger.debug("Constructing a restrictive effect from the input samples.")
        return self._construct_restrictive_effect_from_textual_data(additional_conditions, assignment_statements)

    @dispatch(DataFrame)
    def _construct_safe_conditional_effect_for_single_sample(
            self, sample_data: DataFrame) -> Optional[Tuple[Optional[ConditionalEffect], Precondition]]:
        """Constructs a conditional effect for a single sample.

        :param sample_data: the sample to construct the conditional effect for.
        :return: the conditional effect and the precondition for the conditional effect.
        """
        assignment_statements = []
        additional_conditions = []
        for fluent in sample_data.columns:
            if fluent.startswith(NEXT_STATE_PREFIX):
                if sample_data[fluent[POST_NEXT_STATE_PREFIX_INDEX:]].iloc[0] != sample_data[fluent].iloc[0]:
                    self.logger.debug("The value of the fluent changed by applying the action on the previous state.")
                    assignment_statements.append(
                        f"(assign {fluent[POST_NEXT_STATE_PREFIX_INDEX:]} {sample_data[fluent].iloc[0]})")

            else:  # the fluent is belongs to the previous state
                additional_conditions.append(f"(= {fluent} {sample_data[fluent].iloc[0]})")

        return self._construct_restrictive_effect_from_textual_data(additional_conditions, assignment_statements)

    def _construct_multiple_safe_conditional_effects(
            self, combined_data: DataFrame) -> Tuple[List[ConditionalEffect], Precondition]:
        """Constructs a safe conditional effect and the additional preconditions needed for the action
            when the there is not enough data to learn the effect.

        :param combined_data: the data frame containing the previous and next state values.
        :return: the conditional effects combined with the disjunctive precondition that we need to add.
        """
        disjunctive_precondition = Precondition("or")
        conditional_effects = []
        if combined_data.shape[0] == 1:
            conditional_effect, restrictive_precondition = \
                self._construct_safe_conditional_effect_for_single_sample(combined_data)
            return [conditional_effect], restrictive_precondition

        for index, row in combined_data.iterrows():
            single_row_result = self._construct_safe_conditional_effect_for_single_sample(row)
            if single_row_result is None:
                continue

            conditional_effect, restrictive_precondition = single_row_result
            disjunctive_precondition.add_condition(restrictive_precondition)
            if conditional_effect is not None:
                conditional_effects.append(conditional_effect)

        if len(conditional_effects) == 0:
            self.logger.debug("The algorithm could not find any changes in the state variables. Continuing.")
            return [], disjunctive_precondition

        return conditional_effects, disjunctive_precondition

    def _action_not_affects_fluent(self, lifted_function: str, testing_dataframe: DataFrame) -> bool:
        """Validates that the action affects the given lifted function.

        :param lifted_function: the name of the lifted function to check.
        :param testing_dataframe: the dataframe containing the previous and next state values of the relevant feature.
        :return: Whether the action affects the given lifted function.
        """
        self.logger.debug("Checking if the action affects the given lifted function...")
        subtraction_df = testing_dataframe[LABEL_COLUMN] - testing_dataframe[lifted_function]
        unique_values = subtraction_df.unique()
        return len(unique_values) == 1 and unique_values[0] == 0

    def construct_assignment_equations(
            self, previous_state_data: Dict[str, List[float]],
            next_state_data: Dict[str, List[float]],
            allow_unsafe_learning: bool = False) -> Union[
        Tuple[List[ConditionalEffect], Precondition], Set[NumericalExpressionTree],
        Tuple[Set[NumericalExpressionTree], Precondition]]:
        """Constructs the assignment statements for the action according to the changed value functions.

        :param previous_state_data: the data of the previous state.
        :param next_state_data: the data of the next state.
        :param allow_unsafe_learning: whether to allow unsafe learning.
        :return: the constructed effects with the possibly additional preconditions.
        """
        assignment_statements = []
        self.logger.info(f"Constructing the fluent assignment equations for action {self.action_name}.")
        combined_data = self._combine_states_data(previous_state_data, next_state_data)
        if combined_data.shape[0] == 1:
            self.logger.info(f"The action {self.action_name} contains a single unique observation!")
            return self._construct_multiple_safe_conditional_effects(combined_data)

        # The dataset contains more than one observation.
        self.logger.debug("Removing fluents that are constant zero...")
        filtered_df, zero_fluent_conditions, zero_fluents = self._remove_constant_zero_fluents(combined_data)
        features_df = filtered_df.copy()[[k for k in previous_state_data.keys() if k in filtered_df.columns]]
        regression_df, linear_dep_conditions, dependent_columns = detect_linear_dependent_features(features_df)

        combined_conditions = linear_dep_conditions + zero_fluent_conditions
        if not self._validate_legal_equations(regression_df):
            self.logger.info(f"The action is not safe to learn, creating a restrictive representation!")
            return self._construct_multiple_safe_conditional_effects(combined_data)

        self.logger.debug("The action is safe to learn, constructing the assignment equations...")
        tagged_next_state_fluents = [f"{NEXT_STATE_PREFIX}{fluent_name}" for fluent_name in next_state_data.keys()]
        tested_fluents_names = [fluent_name for fluent_name in next_state_data.keys()]
        for feature_fluent, tagged_fluent in zip(tested_fluents_names, tagged_next_state_fluents):
            change_df = combined_data.copy()[[feature_fluent, tagged_fluent]]
            change_df.rename(columns={tagged_fluent: LABEL_COLUMN}, inplace=True)
            if self._action_not_affects_fluent(feature_fluent, change_df):
                self.logger.info(f"The action {self.action_name} does not affect the fluent - {feature_fluent}")
                continue

            regression_df[LABEL_COLUMN] = combined_data[tagged_fluent]
            polynomial_equation = self._solve_safe_independent_equations(
                feature_fluent, regression_df, allow_unsafe_learning=allow_unsafe_learning)

            if polynomial_equation is not None:
                assignment_statements.append(polynomial_equation)

        if len(assignment_statements) == 0:
            self.logger.debug("The algorithm could not find any effects by the action.")
            return set()

        if len(combined_conditions) == 0:
            return construct_numeric_effects(assignment_statements, self.domain_functions)

        return construct_numeric_effects(assignment_statements, self.domain_functions), \
            construct_numeric_conditions(combined_conditions, ConditionType.conjunctive, self.domain_functions)

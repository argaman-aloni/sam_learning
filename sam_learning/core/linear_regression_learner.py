"""This module contains the LinearRegressionLearner class."""
import logging
from typing import Optional, List, Dict, Tuple, Set

import numpy as np
from pandas import DataFrame
from pddl_plus_parser.models import Precondition, NumericalExpressionTree, PDDLFunction
from sklearn.linear_model import LinearRegression

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import EquationSolutionType, ConditionType
from sam_learning.core.numeric_utils import detect_linear_dependent_features, construct_linear_equation_string, \
    construct_non_circular_assignment, construct_multiplication_strings, prettify_coefficients, \
    construct_numeric_conditions, construct_numeric_effects, filter_constant_features, get_num_independent_equations

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

    def _validate_legal_equations(self, values_df: DataFrame) -> bool:
        """Validates that there are enough independent equations which enable for a single solution for the equation.

        :param values_df: the matrix constructed based on the observations.
        """
        if len(values_df) == 1:
            return False

        num_dimensions = len(values_df.columns.tolist()) + 1  # +1 for the bias.
        num_independent_rows = get_num_independent_equations(values_df)
        if num_independent_rows >= num_dimensions:
            return True

        failure_reason = f"There are too few independent rows of data! " \
                         f"cannot create a linear equation from the current data for action - {self.action_name}!"
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
            reason = f"The learned effects are not safe since the R^2 is not high enough. " \
                     f"Got R^2 of {learning_score} and expected {LEGAL_LEARNING_SCORE}! " \
                     f"Action {self.action_name} is not safe!"
            self.logger.warning(reason)
            if not allow_unsafe_learning:
                raise NotSafeActionError(self.action_name, reason, EquationSolutionType.no_solution_found)

        coefficients = list(regressor.coef_) + [regressor.intercept_]
        coefficients = prettify_coefficients(coefficients)
        self.logger.debug(f"Learned the coefficients for the numeric equations with r^2 score of {learning_score}")
        return coefficients, learning_score

    def _solve_linear_equations(
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

        if all([coef == 0 for coef in coefficient_vector]) and len(regression_df[LABEL_COLUMN].unique()) == 1 \
                and regression_df[LABEL_COLUMN].unique() == 0:
            self.logger.debug("The algorithm designated a vector of zeros to the equation "
                              "which means that there are not coefficients and the label itself is also zero. "
                              "Assuming assignment function of the value 0.0.")
            return f"(assign {lifted_function} {0.0})"

        functions_and_dummy = list(regression_df.columns[:-1]) + ["(dummy)"]
        if lifted_function in regression_df.columns and \
                coefficient_vector[list(regression_df.columns).index(lifted_function)] == 1 and \
                all([coefficient_vector[list(regression_df.columns).index(function)] == 0
                     for function in regression_df.columns if function != lifted_function]):
            self.logger.info(f"The value of the {lifted_function} is the same before and "
                             f"after the application of the action. The action does not affect {lifted_function}!")
            return None

        if lifted_function in regression_df.columns and coefficient_vector[
            list(regression_df.columns).index(lifted_function)] != 0:
            self.logger.debug("the assigned party is a part of the equation, "
                              "cannot use circular dependency so changing the format!")
            coefficients_map = {lifted_func: coef for lifted_func, coef in
                                zip(functions_and_dummy, coefficient_vector)}

            previous_value = next_value = 0.0
            for i in range(len(regression_df)):
                previous_value = regression_df.iloc[i][lifted_function]
                next_value = regression_df.iloc[i][LABEL_COLUMN]
                if previous_value != next_value:
                    break

            return construct_non_circular_assignment(lifted_function,
                                                     coefficients_map,
                                                     previous_value,
                                                     next_value)

        multiplication_functions = construct_multiplication_strings(coefficient_vector, functions_and_dummy)
        if len(multiplication_functions) == 0:
            self.logger.debug("The algorithm designated a vector of zeros to the equation "
                              "which means that there are not coefficients. Continuing.")
            return None

        constructed_right_side = construct_linear_equation_string(multiplication_functions)
        return f"(assign {lifted_function} {constructed_right_side})"

    @staticmethod
    def _extract_const_conditions(
            precondition_statements: List[List[str]], additional_conditions: List[str] = None) -> List[str]:
        """Extract conditions that are constant for all the samples.

        Note:
            These preconditions that appear in all samples will be considered axioms and will be added the
            general preconditions to create a more compact version of the action's preconditions.

        :param precondition_statements: the precondition statements.
        :return: the extracted constant conditions.
        """
        const_preconditions = set(additional_conditions) if additional_conditions else set()
        for conjunction_statement in precondition_statements:
            for condition in conjunction_statement:
                if all([condition in other_conjunction for other_conjunction in precondition_statements]):
                    const_preconditions.add(condition)
                    continue

        for conjunction_statement in precondition_statements:
            for const_condition in const_preconditions:
                if const_condition in conjunction_statement:
                    conjunction_statement.remove(const_condition)

        return list(const_preconditions)

    def _construct_restrictive_numeric_preconditions(
            self, combined_data: DataFrame, additional_conditions: List[str] = None) -> Precondition:
        """Constructs the restrictive numeric preconditions for the action.

        Note:
            if there is only one sample we create an AND condition, otherwise we create an OR condition.

        :param combined_data: the combined data of the previous and next states.
        :return: the restrictive numeric preconditions.
        """
        self.logger.debug(f"Constructing the restrictive numeric preconditions for the action {self.action_name}.")
        precondition_statements = []
        for index, row in combined_data.iterrows():
            additional_conditions = []
            for fluent in row.index.tolist():
                if fluent.startswith(NEXT_STATE_PREFIX):
                    continue

                else:  # the fluent is belongs to the previous state
                    additional_conditions.append(f"(= {fluent} {row[fluent]})")

            precondition_statements.append(additional_conditions)

        if len(precondition_statements) == 1:
            return construct_numeric_conditions(
                precondition_statements[0], ConditionType.conjunctive, self.domain_functions)

        const_preconditions = self._extract_const_conditions(precondition_statements, additional_conditions)
        combined_preconditions = None
        if len(const_preconditions) > 0:
            combined_preconditions = construct_numeric_conditions(
                const_preconditions, ConditionType.conjunctive, self.domain_functions)

        dijsunctive_preconditions = Precondition("or")
        for precondition_statement in precondition_statements:
            dijsunctive_preconditions.add_condition(construct_numeric_conditions(
                precondition_statement, ConditionType.conjunctive, self.domain_functions))

        if combined_preconditions is not None:
            combined_preconditions.add_condition(dijsunctive_preconditions)
            return combined_preconditions

        return dijsunctive_preconditions

    def _construct_effect_from_single_sample(self, sample_data: DataFrame) -> Set[NumericalExpressionTree]:
        """constructs the effect from a single sample.

        :return: the constructed numeric effect.
        """
        assignment_statements = []
        for fluent in sample_data.columns:
            if not fluent.startswith(NEXT_STATE_PREFIX):
                continue

            if sample_data[fluent[POST_NEXT_STATE_PREFIX_INDEX:]].iloc[0] != sample_data[fluent].iloc[0]:
                self.logger.debug("The value of the fluent changed by applying the action on the previous state.")
                assignment_statements.append(
                    f"(assign {fluent[POST_NEXT_STATE_PREFIX_INDEX:]} {sample_data[fluent].iloc[0]})")

        return construct_numeric_effects(assignment_statements, self.domain_functions)

    def _filter_out_constant_effects(
            self, constant_features: List[str], combined_data_df: DataFrame, filtered_df: DataFrame) -> List[str]:
        """

        :param constant_features:
        :param combined_data_df:
        :param filtered_df:
        :return:
        """
        removed_next_state_variables = []
        for constant_feature in constant_features:
            if combined_data_df[constant_feature].equals(combined_data_df[f"{NEXT_STATE_PREFIX}{constant_feature}"]):
                self.logger.info(f"The value of the {constant_feature} is the same before and "
                                 f"after the application of the action. The value is constant so it is not needed!")
                filtered_df.drop(f"{NEXT_STATE_PREFIX}{constant_feature}", axis=1, inplace=True)
                removed_next_state_variables.append(constant_feature)

        return removed_next_state_variables

    def construct_assignment_equations(
            self, previous_state_data: Dict[str, List[float]],
            next_state_data: Dict[str, List[float]],
            allow_unsafe_learning: bool = False) -> Tuple[Set[NumericalExpressionTree], Optional[Precondition], bool]:
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
            return self._construct_effect_from_single_sample(combined_data), \
                self._construct_restrictive_numeric_preconditions(combined_data), False

        # The dataset contains more than one observation.
        self.logger.debug("Removing fluents that are constant zero...")
        tagged_next_state_fluents = [f"{NEXT_STATE_PREFIX}{fluent_name}" for fluent_name in next_state_data.keys()]
        filtered_df, constant_features_conditions, constant_features = filter_constant_features(
            combined_data, columns_to_ignore=tagged_next_state_fluents)
        features_df = filtered_df.copy()[[k for k in previous_state_data.keys() if k in filtered_df.columns]]
        regression_df, linear_dep_conditions, dependent_columns = detect_linear_dependent_features(features_df)
        combined_conditions = linear_dep_conditions + constant_features_conditions
        tested_fluents_names = [fluent_name for fluent_name in next_state_data.keys()]
        is_safe_to_learn = self._validate_legal_equations(regression_df)
        for feature_fluent, tagged_fluent in zip(tested_fluents_names, tagged_next_state_fluents):
            regression_df[LABEL_COLUMN] = combined_data[tagged_fluent]
            polynomial_equation = self._solve_linear_equations(
                feature_fluent, regression_df, allow_unsafe_learning=allow_unsafe_learning)

            if polynomial_equation is not None:
                assignment_statements.append(polynomial_equation)

        if not is_safe_to_learn:
            return construct_numeric_effects(assignment_statements, self.domain_functions), \
                self._construct_restrictive_numeric_preconditions(features_df, combined_conditions), False

        return construct_numeric_effects(assignment_statements, self.domain_functions), None, True

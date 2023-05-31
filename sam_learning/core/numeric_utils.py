"""Utility functions for handling and presenting numeric data."""
import itertools
from typing import Union, List, Dict, Tuple, Optional, Set

import math
import numpy as np
import sympy
from pandas import DataFrame, Series
from pddl_plus_parser.lisp_parsers import PDDLTokenizer
from pddl_plus_parser.models import Precondition, PDDLFunction, construct_expression_tree, NumericalExpressionTree

from sam_learning.core.learning_types import ConditionType

EPSILON = 1e-10


def prettify_floating_point_number(number: float) -> float:
    """Converts the floating point number into a prettier form so that the created equations would be more presentable.

    :param number: the RAW number received from the learning process.
    :return: the prettified version of the number.
    """
    return round(number, 2)


def construct_multiplication_strings(coefficients_vector: Union[np.ndarray, List[float]],
                                     function_variables: List[str]) -> List[str]:
    """Constructs the strings representing the multiplications of the function variables with the coefficient.

    :param coefficients_vector: the coefficient that multiplies the function vector.
    :param function_variables: the name of the numeric fluents that are being used.
    :return: the representation of the fluents multiplied by the coefficients.
    """
    product_components = []
    for func, coefficient in zip(function_variables, coefficients_vector):
        if coefficient == 0.0:
            continue

        if func == "(dummy)":
            product_components.append(f"{prettify_floating_point_number(coefficient)}")

        else:
            product_components.append(f"(* {func} {prettify_floating_point_number(coefficient)})")

    return product_components


def prettify_coefficients(coefficients: List[float]) -> List[float]:
    """Converts the coefficients into a prettier form so that the created equations would be more presentable.

    :param coefficients: the RAW coefficients received from the linear regression.
    :return: the prettified version of the coefficients.
    """
    coefficients = [coef if abs(coef) > EPSILON else 0.0 for coef in coefficients]
    prettified_coefficients = [round(value, 2) for value in coefficients]
    return prettified_coefficients


def construct_linear_equation_string(multiplication_parts: List[str]) -> str:
    """Construct the addition parts of the linear equation string.

    :param multiplication_parts: the multiplication function strings that are multiplied by the coefficient.
    :return: the string representing the sum of the linear variables.
    """
    if len(multiplication_parts) == 1:
        return multiplication_parts[0]

    inner_layer = construct_linear_equation_string(multiplication_parts[1:])
    return f"(+ {multiplication_parts[0]} {inner_layer})"


def construct_non_circular_assignment(lifted_function: str, coefficients_map: Dict[str, float],
                                      previous_value: float, next_value: float) -> str:
    """Changes circular assignment statements to be non-circular.

    Note:
        Since numeric solvers don't approve circular dependencies we need format the assignment operations to be
        in the form of increase / decrease.

    :param lifted_function: the assigned variable.
    :param coefficients_map: the calculated coefficient map.
    :param previous_value: the numeric value of the function prior to the action's execution.
    :param next_value: the numeric value of the function after the action's execution.
    :return: the formatted string without circular dependencies.
    """
    normalized_coefficients = {k: v / coefficients_map[lifted_function] for k, v in
                               coefficients_map.items() if k != lifted_function and v != 0}
    if len(normalized_coefficients) == 1:
        normalized_coefficients = {k: abs(v) for k, v in normalized_coefficients.items()}

    multiplication_functions = construct_multiplication_strings(
        list(normalized_coefficients.values()), list(normalized_coefficients.keys()))
    constructed_right_side = construct_linear_equation_string(multiplication_functions)

    if previous_value < next_value:
        return f"(increase {lifted_function} {constructed_right_side})"

    return f"(decrease {lifted_function} {constructed_right_side})"


def extract_numeric_linear_coefficient(function1_values: Series, function2_values: Series) -> float:
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


def filter_constant_features(input_df: DataFrame, column_to_ignore: Optional[str] = None) -> Tuple[
    DataFrame, List[str], List[str]]:
    """Filters out fluents that contain only constant values since they do not contribute to the convex hull.

    :param input_df: the matrix of the previous state values.
    :param column_to_ignore:
    :return: the filtered matrix and the equality strings, i.e. the strings of the values that should be equal.
    """
    equal_fluent_strs, removed_fluents = [], []
    result_ft = input_df.copy()
    for col in [col for col in input_df.columns if col != column_to_ignore]:
        if len(input_df[col].unique()) == 1:
            equal_fluent_strs.append(f"(= {col} {input_df[col].unique()[0]})")
            removed_fluents.append(col)
            result_ft.drop(col, inplace=True, axis=1)

    return result_ft, equal_fluent_strs, removed_fluents


def detect_linear_dependent_features(data_matrix: DataFrame, column_to_ignore: Optional[str] = None) -> Tuple[
    DataFrame, List[str], Dict[str, str]]:
    """Detects linear dependent features and adds the equality constraints to the problem.

    :param data_matrix: the matrix of the previous state values.
    :param column_to_ignore: the column to ignore.
    :return: the filtered matrix and the equality strings, i.e. the strings of the values that should be equal and the
        column to column mapping.
    """
    additional_conditions = []
    dependent_columns = {}
    filtered_matrix, equality_strs, _ = filter_constant_features(data_matrix.copy(), column_to_ignore)
    additional_conditions.extend(equality_strs)
    for col1, col2 in itertools.combinations(filtered_matrix.columns, 2):
        if col1 == column_to_ignore or col2 == column_to_ignore:
            continue

        reduced_form, inds = sympy.Matrix(filtered_matrix[[col1, col2]].values).rref()
        if len(inds) == 2:
            continue

        linear_coeff = extract_numeric_linear_coefficient(filtered_matrix[col1], filtered_matrix[col2])
        additional_conditions.append(f"(= {col1} (* {linear_coeff} {col2}))")
        filtered_matrix[col1] = filtered_matrix[col2] * linear_coeff
        dependent_columns[col2] = col1

    filtered_matrix = filtered_matrix[[col for col in filtered_matrix.columns if col not in dependent_columns]]
    return filtered_matrix, additional_conditions, dependent_columns


def construct_numeric_conditions(
        conditions: List[str], condition_type: ConditionType,
        domain_functions: Dict[str, PDDLFunction]) -> Precondition:
    """Construct the numeric conditions from the given equality conditions.

    :param conditions: the condition strings to create the preconditions from.
    :param condition_type: the type of the conditions ('and' or 'or').
    :param domain_functions: the domain functions to use for the numeric conditions.
    :return: the constructed numeric precondition.
    """
    precondition_type = "and" if condition_type == ConditionType.conjunctive else "or"
    constructed_precondition = Precondition(precondition_type)
    for condition in conditions:
        numeric_tokenizer = PDDLTokenizer(pddl_str=condition)
        tokens = numeric_tokenizer.parse()
        numeric_expression = construct_expression_tree(tokens, domain_functions)
        constructed_precondition.add_condition(NumericalExpressionTree(numeric_expression))

    return constructed_precondition


def construct_numeric_effects(
        effects: List[str], domain_functions: Dict[str, PDDLFunction]) -> Set[NumericalExpressionTree]:
    """Construct the numeric effects for the given set of input strings.

    :param effects: the effect strings to create the numeric effects from.
    :param domain_functions: the domain functions to use for the numeric effects.
    :return: the constructed numeric precondition.
    """
    numeric_effects = []
    for condition in effects:
        numeric_tokenizer = PDDLTokenizer(pddl_str=condition)
        tokens = numeric_tokenizer.parse()
        numeric_expression = construct_expression_tree(tokens, domain_functions)
        numeric_effects.append(numeric_expression)

    return {NumericalExpressionTree(expr) for expr in numeric_effects}

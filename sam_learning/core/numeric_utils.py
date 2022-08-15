"""Utility functions for handling and presenting numeric data."""
from typing import Union, List, Dict

import numpy as np

EPSILON = 1e-10


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
            product_components.append(f"{coefficient}")

        else:
            product_components.append(f"(* {func} {coefficient})")

    return product_components


def prettify_coefficients(coefficients: List[float]) -> List[float]:
    """Converts the coefficients into a prettier form so that the created equations would be more presentable.

    :param coefficients: the RAW coefficients received from the linear regression.
    :return: the prettified version of the coefficients.
    """
    coefficients = [coef if abs(coef) > EPSILON else 0.0 for coef in coefficients]
    prettified_coefficients = [round(value, 2) for value in coefficients]
    return prettified_coefficients


def prettify_floating_point_number(number: float) -> float:
    """Converts the floating point number into a prettier form so that the created equations would be more presentable.

    :param number: the RAW number received from the learning process.
    :return: the prettified version of the number.
    """
    return round(number, 2)


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

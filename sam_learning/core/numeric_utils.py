"""Utility functions for handling and presenting numeric data."""
from typing import Union, List

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

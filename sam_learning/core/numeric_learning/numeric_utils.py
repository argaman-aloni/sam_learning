"""Utility functions for handling and presenting numeric data."""
import itertools
from typing import Union, List, Dict, Tuple, Optional, Set

import math
import matplotlib.pyplot as plt
import numpy as np
import sympy
from pandas import DataFrame, Series
from pddl_plus_parser.lisp_parsers import PDDLTokenizer
from pddl_plus_parser.models import Precondition, PDDLFunction, construct_expression_tree, NumericalExpressionTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.feature_selection import VarianceThreshold

from sam_learning.core.learning_types import ConditionType

EPSILON = 1e-4


def get_num_independent_equations(data_matrix: DataFrame) -> int:
    """Calculates the number of independent equations in the given matrix.

    :param data_matrix: the matrix of the previous state values.
    :return: the number of independent equations.
    """
    values_matrix = data_matrix.to_numpy()
    num_rows = values_matrix.shape[0]
    values_matrix_with_bias = np.c_[values_matrix, np.ones(num_rows)]
    _, pivot_cols = sympy.Matrix(values_matrix_with_bias).rref()
    return len(pivot_cols)


def prettify_floating_point_number(number: float) -> float:
    """Converts the floating point number into a prettier form so that the created equations would be more presentable.

    :param number: the RAW number received from the learning process.
    :return: the prettified version of the number.
    """
    return int(number) if abs(number - int(number)) < EPSILON else number


def construct_multiplication_strings(coefficients_vector: Union[np.ndarray, List[float]], function_variables: List[str]) -> List[str]:
    """Constructs the strings representing the multiplications of the function variables with the coefficient.

    :param coefficients_vector: the coefficient that multiplies the function vector.
    :param function_variables: the name of the numeric fluents that are being used.
    :return: the representation of the fluents multiplied by the coefficients.
    """
    product_components = []
    for func, coefficient in zip(function_variables, coefficients_vector):
        if abs(round(coefficient, 4)) <= EPSILON:
            continue

        if func == "(dummy)":
            product_components.append(f"{prettify_floating_point_number(round(coefficient, 4))}")

        elif coefficient == 1.0:
            product_components.append(func)

        else:
            product_components.append(f"(* {func} {prettify_floating_point_number(round(coefficient, 4))})")

    return product_components


def prettify_coefficients(coefficients: List[float]) -> List[float]:
    """Converts the coefficients into a prettier form so that the created equations would be more presentable.

    :param coefficients: the RAW coefficients received from the linear regression.
    :return: the prettified version of the coefficients.
    """
    coefficients = [coef if abs(coef) > EPSILON else 0.0 for coef in coefficients]
    prettified_coefficients = [round(value, 4) for value in coefficients]
    return prettified_coefficients


def construct_projected_variable_strings(
    function_variables: List[str], shift_point: Union[np.ndarray, List[float]], projection_basis: Union[np.ndarray, List[List[float]]],
) -> List[str]:
    """Constructs the strings representing the multiplications of the function variables with the coefficient.

    :param function_variables: the name of the numeric fluents that are being used.
    :param shift_point: the point in which the data was shifted according to.
    :param projection_basis: the basis in which the data was projected to.
    :return: the new variable names after applying the PCA model transformation
    """
    shifted_by_mean = []
    for func, shift_value in zip(function_variables, shift_point):
        component_function = func if shift_value == 0.0 else f"(- {func} {prettify_floating_point_number(round(shift_value, 2))})"
        shifted_by_mean.append(component_function)

    sum_of_product_by_components = []
    for row in range(len(projection_basis)):
        product_by_components_row = []
        for shifted, component in zip(shifted_by_mean, projection_basis[row]):
            if abs(round(component, 4)) <= EPSILON:
                continue

            if component == 1.0:
                product_by_components_row.append(shifted)
                continue

            product_by_components_row.append(f"(* {shifted} {prettify_floating_point_number(round(component, 4))})")

        sum_of_product_by_components.append(construct_linear_equation_string(product_by_components_row))

    return sum_of_product_by_components


def construct_linear_equation_string(multiplication_parts: List[str]) -> str:
    """Construct the addition parts of the linear equation string.

    :param multiplication_parts: the multiplication function strings that are multiplied by the coefficient.
    :return: the string representing the sum of the linear variables.
    """
    if len(multiplication_parts) == 1:
        return multiplication_parts[0]

    inner_layer = construct_linear_equation_string(multiplication_parts[1:])
    return f"(+ {multiplication_parts[0]} {inner_layer})"


def construct_non_circular_assignment(lifted_function: str, coefficients_map: Dict[str, float], previous_value: float, next_value: float) -> str:
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
    if previous_value < next_value:
        coefficients_map[lifted_function] = coefficients_map[lifted_function] - 1

    else:
        coefficients_map = {k: -v for k, v in coefficients_map.items()}
        coefficients_map[lifted_function] = coefficients_map[lifted_function] + 1

    multiplication_functions = construct_multiplication_strings(list(coefficients_map.values()), list(coefficients_map.keys()))
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
    denominator = np.where(function2_values == 0, 1e-9, function2_values)
    division_res = np.array(function1_values) / np.array(denominator)
    for value in division_res:
        if not math.isnan(value) and not math.isinf(value):
            linear_coeff = value
            break

    return prettify_floating_point_number(round(linear_coeff, 4))


def filter_constant_features(input_df: DataFrame, columns_to_ignore: Optional[List[str]] = []) -> Tuple[DataFrame, List[str], List[str]]:
    """Filters out fluents that contain only constant values since they do not contribute to the convex hull.

    :param input_df: the matrix of the previous state values.
    :param columns_to_ignore: the list of columns that should be ignored.
    :return: the filtered matrix and the equality strings, i.e. the strings of the values that should be equal.
    """
    equal_fluent_strs, removed_fluents = [], []
    relevant_columns = [col for col in input_df.columns if col not in columns_to_ignore]
    try:
        selector = VarianceThreshold().set_output(transform="pandas")
        result_df = selector.fit_transform(input_df[relevant_columns])
        filtered_columns = [column for column in relevant_columns if column not in result_df.columns]
        for col in filtered_columns:
            equal_fluent_strs.append(f"(= {col} {input_df[col].unique()[0]})")
            removed_fluents.append(col)

        return result_df, equal_fluent_strs, removed_fluents

    except ValueError:
        # no feature in X meets the variance threshold 0.0 (i.e. all features are constant)
        for col in input_df.columns:
            equal_fluent_strs.append(f"(= {col} {input_df[col].unique()[0]})")

        return DataFrame(), equal_fluent_strs, input_df.columns


def detect_linear_dependent_features(data_matrix: DataFrame) -> Tuple[DataFrame, List[str], Dict[str, str]]:
    """Detects linear dependent features and adds the equality constraints to the problem.

    The idea is: put together these column vectors as a matrix and calculate its row-echelon form.
    If the row-echelon form is diagonal with only ones, the set of vectors is independent, otherwise, it is dependent.

    :param data_matrix: the matrix of the previous state values.
    :return: the filtered matrix and the equality strings, i.e. the strings of the values that should be equal and the
        column to column mapping.
    """
    additional_conditions = []
    dependent_columns = {}

    data_matrix_copy = data_matrix.copy()
    for col1, col2 in itertools.combinations(data_matrix_copy.columns, 2):
        reduced_form, _ = sympy.Matrix(data_matrix_copy[[col1, col2]].values).rref()
        diagonal_required_result = np.array([[1, 0], [0, 1]])
        if np.array_equal(diagonal_required_result, np.diag(np.diag(reduced_form))):
            continue  # The columns are independent

        independent_column, dependent_column = col1, col2
        linear_coeff = extract_numeric_linear_coefficient(data_matrix_copy[dependent_column], data_matrix_copy[independent_column])
        additional_conditions.append(f"(= {dependent_column} (* {linear_coeff} {independent_column}))")
        dependent_columns[dependent_column] = independent_column

    filtered_matrix = data_matrix_copy[[col for col in data_matrix_copy.columns if col not in dependent_columns]]
    return filtered_matrix, additional_conditions, dependent_columns


def construct_numeric_conditions(
    conditions: List[str], condition_type: ConditionType, domain_functions: Dict[str, PDDLFunction]
) -> Optional[Precondition]:
    """Construct the numeric conditions from the given equality conditions.

    :param conditions: the condition strings to create the preconditions from.
    :param condition_type: the type of the conditions ('and' or 'or').
    :param domain_functions: the domain functions to use for the numeric conditions.
    :return: the constructed numeric precondition.
    """
    if len(conditions) == 0:
        return None

    precondition_type = "and" if condition_type == ConditionType.conjunctive else "or"
    constructed_precondition = Precondition(precondition_type)
    precondition_parts = []
    for condition in conditions:
        numeric_tokenizer = PDDLTokenizer(pddl_str=condition)
        tokens = numeric_tokenizer.parse()
        numeric_expression = construct_expression_tree(tokens, domain_functions)
        precondition_parts.append(NumericalExpressionTree(numeric_expression))

    for part in precondition_parts:
        constructed_precondition.add_condition(part)

    return constructed_precondition


def construct_numeric_effects(effects: List[str], domain_functions: Dict[str, PDDLFunction]) -> Set[NumericalExpressionTree]:
    """Construct the numeric effects for the given set of input strings.

    :param effects: the effect strings to create the numeric effects from.
    :param domain_functions: the domain functions to use for the numeric effects.
    :return: the constructed numeric precondition.
    """
    if len(effects) == 0:
        return set()

    numeric_effects = []
    for condition in effects:
        numeric_tokenizer = PDDLTokenizer(pddl_str=condition)
        tokens = numeric_tokenizer.parse()
        numeric_expression = construct_expression_tree(tokens, domain_functions)
        numeric_effects.append(numeric_expression)

    return {NumericalExpressionTree(expr) for expr in numeric_effects}


def extended_gram_schmidt(input_basis_vectors: List[List[float]], eigen_vectors: Optional[List[List[float]]] = None) -> List[List[float]]:
    """Runs the extended Gram-Schmidt algorithm on the input basis vectors.

    Note:
        The algorithm is extended in the that it can handle the additional input eigen vectors and return the
        orthonormal basis vectors of the input basis vectors and eigen vectors.

    :param input_basis_vectors: The input basis vectors.
    :param eigen_vectors: The eigen vectors - optional and can be none.
    :return: The orthonormal basis vectors of the input basis vectors and eigen vectors.
    """
    non_normal_vectors = eigen_vectors.copy() if eigen_vectors else []
    normal_vectors = []
    for vector in input_basis_vectors:
        # Gram Schmidt magic
        projected_vector = vector - np.sum([(np.dot(vector, b) / np.linalg.norm(b) ** 2) * np.array(b) for b in non_normal_vectors], axis=0)
        if not (np.absolute(projected_vector) > EPSILON).any():
            continue

        non_normal_vectors.append(projected_vector.tolist())
        normal_vectors.append((projected_vector / np.linalg.norm(projected_vector)).tolist())

    return normal_vectors


def display_convex_hull_2d(action_name: str, hull: ConvexHull) -> None:
    """Displays the convex hull in 2D as a plot.

    :param action_name: the name of the action with its convex hull displayed.
    :param hull: the convex hull to display.
    """
    plt.title(f"{action_name} - convex hull")
    _ = convex_hull_plot_2d(hull)
    plt.show()


def display_convex_hull_3d(action_name: str, hull: ConvexHull) -> None:
    """Displays the convex hull in 3D as a plot.

    :param action_name: the name of the action with its convex hull displayed.
    :param hull: the convex hull to display.
    """
    plt.title(f"{action_name} - convex hull")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Repeat the first point to create a closed shape
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], hull.points[simplex, 2], "b-")

    plt.show()


def display_convex_hull(action_name: str, display_mode: bool, hull: ConvexHull) -> None:
    """Displays the convex hull in as a plot.

    :param action_name: the name of the action with its convex hull displayed.
    :param display_mode: whether to display the plot.
    :param hull: the convex hull to display.
    """
    if not display_mode:
        return

    dimensionality = hull.points.shape[1]
    if dimensionality == 2:
        display_convex_hull_2d(action_name, hull)

    elif dimensionality == 3:
        display_convex_hull_3d(action_name, hull)


def create_monomials(domain_functions: List[str], polynom_degree: int = 0) -> List[List[str]]:
    """Creates the monomials from the state fluents.

    :return: the monomials from the state fluents.
    """
    monomials = list([item] for item in domain_functions)
    if polynom_degree == 0:
        return monomials

    if polynom_degree == 1:
        for first_fluent, second_fluent in itertools.combinations(domain_functions, r=2):
            monomial = sorted([first_fluent, second_fluent])
            monomials.append(monomial)

    else:
        for degree in range(2, polynom_degree + 1):
            for fluent_combination in itertools.combinations_with_replacement(domain_functions, r=degree):
                monomial = sorted(list(fluent_combination))
                monomials.append(monomial)

    return monomials


def _create_polynomial_string_recursive(fluents: List[str]) -> str:
    """Creates the polynomial string representing the equation recursively.

    :param fluents: the numeric fluents to create the polynomial string from.
    :return: the polynomial string representing the equation.
    """
    if len(fluents) == 1:
        return fluents[0]

    return f"(* {fluents[0]} {_create_polynomial_string_recursive(fluents[1:])})"


def create_polynomial_string(fluents: List[str]) -> str:
    """The auxiliary function that creates the polynomial string representing the equation.

    :param fluents: the numeric fluents to create the polynomial string from.
    :return: the polynomial string representing the equation.
    """
    return _create_polynomial_string_recursive(fluents)

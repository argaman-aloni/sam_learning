"""Tests for numeric_utils.py"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sam_learning.core.numeric_learning.numeric_utils import (
    extract_numeric_linear_coefficient,
    construct_projected_variable_strings,
    extended_gram_schmidt,
)


def test_extract_numeric_linear_coefficient_does_not_output_infinity_for_zero_division():
    """Test that the function does not output infinity for zero division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_does_not_output_nan_for_illegal_division():
    """Test that the function does not output nan for illegal division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_outputs_correct_coefficient_when_division_is_legal():
    """Test that the function outputs the correct coefficient when the division is legal."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 1


def test_construct_projected_variable_strings_returns_correct_strings_representing_the_projected_format():
    function_variables = ["(x)", "(y)", "(z)"]
    X = np.array([[-1, -1, 1], [-2, -1, 3], [-3, -2, 23], [1, 1, 31], [2, 1, 12], [3, 2, 7]])
    pca = PCA(n_components=2)
    pca.fit(X)

    pca_variable_strings = construct_projected_variable_strings(function_variables, pca.mean_, pca.components_)
    print(pca_variable_strings)
    assert len(pca_variable_strings) == 2  # 2 components


def test_construct_projected_variable_strings_returns_correct_strings_representing_the_projected_format():
    function_variables = ["(x)", "(y)", "(z)", "w"]
    X = np.array([[1, 0, 17, 3], [5, 5, 7, 3], [2, 4, 9, 7]])
    shift = np.array([-1, 2, 3, 12])

    pca_variable_strings = construct_projected_variable_strings(function_variables, shift, X)
    print(f"x' = {pca_variable_strings[0]}")
    print(f"y' = {pca_variable_strings[1]}")
    print(f"z' = {pca_variable_strings[2]}")

    diagonal_eye = [list(vector) for vector in np.eye(X.shape[1])]
    orthnormal_span = extended_gram_schmidt(diagonal_eye, [[1, 0, 17, 3], [5, 5, 7, 3], [2, 4, 9, 7]])
    print(f"orthnormal_span = {orthnormal_span}")
    # print(pca_variable_strings)
    # assert len(pca_variable_strings) == 2  # 2 components


def test_construct_projected_variable_strings_returns_correct_strings_representing_the_projected_format_when_components_are_equal_to_variables():
    function_variables = ["(x)", "(y)", "(z)"]
    X = np.array([[-1, -1, 1], [-2, -1, 3], [-3, -2, 23], [1, 1, 31], [2, 1, 12], [3, 2, 7]])
    pca = PCA(n_components=3)
    pca.fit(X)

    pca_variable_strings = construct_projected_variable_strings(function_variables, pca.mean_, pca.components_)
    assert len(pca_variable_strings) == 3  # 3 components
    print(pca_variable_strings)


def test_construct_projected_variable_strings_returns_correct_numbers_according_to_construction_calculation_for_simple_case():
    function_variables = ["(x)", "(y)", "(z)"]
    mean = np.array([2, 2, 2])
    pca_components = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    pca_variable_strings = construct_projected_variable_strings(function_variables, mean, pca_components)
    assert len(pca_variable_strings) == 3  # 3 components
    assert pca_variable_strings[0] == "(- (x) 2)"
    assert pca_variable_strings[1] == "(- (y) 2)"
    assert pca_variable_strings[2] == "(- (z) 2)"
    print(pca_variable_strings)


def test_construct_pca_variable_strings_returns_correct_numbers_according_to_construction_calculation_for_complex_case():
    function_variables = ["(x)", "(y)", "(z)"]
    mean = np.array([2, 2, 2])
    pca_components = np.array([[1, 8, 32.5], [12, 1, 33], [3, 2, 1]])

    pca_variable_strings = construct_projected_variable_strings(function_variables, mean, pca_components)
    assert len(pca_variable_strings) == 3  # 3 components
    assert pca_variable_strings[0] == "(+ (- (x) 2) (+ (* (- (y) 2) 8) (* (- (z) 2) 32.5)))"
    assert pca_variable_strings[1] == "(+ (* (- (x) 2) 12) (+ (- (y) 2) (* (- (z) 2) 33)))"
    assert pca_variable_strings[2] == "(+ (* (- (x) 2) 3) (+ (* (- (y) 2) 2) (- (z) 2)))"


def test_extended_gram_schmidt_on_no_shift_base_returns_correct_lower_dimensional_points():
    projections = extended_gram_schmidt([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert len(projections) == 2
    assert projections == [[0, 1, 0], [1, 0, 0]]


def test_extended_gram_schmidt_on_no_shift_base_returns_correct_lower_dimensional_points_with_correct_rational_values():
    projections = extended_gram_schmidt([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    normalized_projection = []
    for vector in projections:
        normalized_projection.append([round(val, 3) for val in vector])

    assert len(projections) == 2
    assert normalized_projection == [[0, 0.707, 0.707], [0, 0.707, -0.707]]


def test_extended_gram_schmidt_on_no_shift_when_given_orthonormal_partial_base_returns_complementary_base_to_points():
    projections = extended_gram_schmidt([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0]])
    assert len(projections) == 1
    assert projections == [[0, 0, 1]]


def test_extended_gram_schmidt_on_no_shift_with_base_correctly_returns_the_missing_vector():
    projections = extended_gram_schmidt([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    extended_standard_base = np.concatenate((np.eye(3), np.array(projections)), axis=0).tolist()
    complementary_projection = extended_gram_schmidt(extended_standard_base, projections)
    assert len(complementary_projection) == 1
    assert complementary_projection == [[1, 0, 0]]


def test_extended_gram_schmidt_on_no_shift_with_base_correctly_returns_the_missing_vector_with_more_complicated_setting():
    projections = extended_gram_schmidt([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    extended_standard_base = [[18, 2, 19], [2, 7, 1], [29, 31, -1]]
    complementary_projection = extended_gram_schmidt(extended_standard_base, projections)
    assert len(complementary_projection) == 1
    assert complementary_projection == [[1.0, -9.868649107779169e-17, 0.0]]


def test_extended_gram_schmidt_on_no_shift_with_base_with_more_rows_than_columns_still_returns_2x2_output():
    projections = extended_gram_schmidt([[-19.0, 32.0], [14.0, 52.0], [28.0, 12.0], [-7.0, 13.0]])
    assert len(projections) == 2
    assert all([len(vector) == 2 for vector in projections])
    assert (
        np.sum(
            np.array(projections)
            - np.array([[-0.510538754155436, 0.859854743840735], [0.859854743840735, 0.510538754155436]])
        )
        < 0.0001
    )
    print(projections)


def test_extended_gram_schmidt_with_no_base_returns_correct_orthonormal_matrix():
    projections = extended_gram_schmidt([[1, 1, 1], [2, 1, 0], [5, 1, 3]])
    assert len(projections) == 3
    assert (
        np.sum(
            np.array(projections)
            - np.array(
                [
                    [0.577350269189626, 0.577350269189626, 0.577350269189626],
                    [0.707106781186548, 0, -0.707106781186548],
                    [0.408248290463863, -0.816496580927726, 0.408248290463863],
                ]
            )
        )
        < 0.0001
    )


def test_extended_gram_schmidt_with_paper_example_returns_correct_values():
    projections = extended_gram_schmidt([[0, 0, 0], [-1, 1, 0], [-1, 0, 1]])
    print(projections)
    assert len(projections) == 2

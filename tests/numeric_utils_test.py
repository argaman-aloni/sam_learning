"""Tests for numeric_utils.py"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sam_learning.core.numeric_utils import extract_numeric_linear_coefficient, construct_projected_variable_strings


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
    assert len(pca_variable_strings) == 2  # 2 components


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


"""Tests for numeric_utils.py"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sam_learning.core.numeric_utils import extract_numeric_linear_coefficient, construct_pca_variable_strings


def test_extract_numeric_linear_coefficient_does_not_output_infinity_for_zero_division():
    """Test that the function does not output infinity for zero division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_does_not_output_nan_for_illegal_division():
    """Test that the function does not output nan for illegal division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_outputs_correct_coefficient_when_division_is_legal():
    """Test that the function outputs the correct coefficient when the division is legal."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 1


def test_construct_pca_variable_strings_returns_correct_strings_representing_the_pca_format():
    function_variables = ["(x)", "(y)", "(z)"]
    X = np.array([[-1, -1, 1], [-2, -1, 3], [-3, -2, 23], [1, 1, 31], [2, 1, 12], [3, 2, 7]])
    pca = PCA(n_components=2)
    pca.fit(X)

    pca_variable_strings = construct_pca_variable_strings(function_variables, pca.mean_, pca.components_)
    assert len(pca_variable_strings) == 2  # 2 components


def test_construct_pca_variable_strings_returns_correct_strings_representing_the_pca_format_when_components_are_equal_to_variables():
    function_variables = ["(x)", "(y)", "(z)"]
    X = np.array([[-1, -1, 1], [-2, -1, 3], [-3, -2, 23], [1, 1, 31], [2, 1, 12], [3, 2, 7]])
    pca = PCA(n_components=3)
    pca.fit(X)

    pca_variable_strings = construct_pca_variable_strings(function_variables, pca.mean_, pca.components_)
    assert len(pca_variable_strings) == 3  # 3 components
    print(pca_variable_strings)

"""Tests for numeric_utils.py"""
import pandas as pd

from sam_learning.core.numeric_utils import extract_numeric_linear_coefficient


def test_extract_numeric_linear_coefficient_does_not_output_infinity_for_zero_division():
    """Test that the function does not output infinity for zero division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_does_not_output_nan_for_illegal_division():
    """Test that the function does not output nan for illegal division."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([0, 0, 0])) == 0


def test_extract_numeric_linear_coefficient_outputs_correct_coefficient_when_division_is_legal():
    """Test that the function outputs the correct coefficient when the division is legal."""
    assert extract_numeric_linear_coefficient(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])) == 1

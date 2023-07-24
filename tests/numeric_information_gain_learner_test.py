"""Module tests for the numeric information gaining process."""
import pandas as pd
import pytest
from pddl_plus_parser.models import PDDLFunction

from sam_learning.core import NumericInformationGainLearner

TEST_ACTION_NAME = 'test_action'
TEST_FUNCTION_NAMES = ["(x)", "(y)", "(z)", "(w)"]


@pytest.fixture
def numeric_information_gain_learner() -> NumericInformationGainLearner:
    domain_functions = {
        "(x)": PDDLFunction(name="x", signature={}),
        "(y)": PDDLFunction(name="y", signature={}),
        "(z)": PDDLFunction(name="z", signature={}),
        "(w)": PDDLFunction(name="w", signature={})
    }
    return NumericInformationGainLearner(TEST_ACTION_NAME, domain_functions)


def test_add_positive_sample_adds_new_sample_to_the_existing_positive_dataframe(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    numeric_information_gain_learner.positive_samples_df = test_dataframe
    assert len(numeric_information_gain_learner.positive_samples_df) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    numeric_information_gain_learner.add_positive_sample(new_sample)
    assert len(numeric_information_gain_learner.positive_samples_df) == 4
    assert len(numeric_information_gain_learner.positive_samples_df.columns) == 4


def test_add_positive_sample_does_not_add_to_negative_samples(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    numeric_information_gain_learner.positive_samples_df = test_dataframe
    assert len(numeric_information_gain_learner.positive_samples_df) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    numeric_information_gain_learner.add_positive_sample(new_sample)
    assert len(numeric_information_gain_learner.negative_samples_df) == 0


def testadd_negative_sample_adds_new_sample_to_the_existing_negative_dataframe(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    numeric_information_gain_learner.negative_samples_df = test_dataframe
    assert len(numeric_information_gain_learner.negative_samples_df) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    numeric_information_gain_learner.add_negative_sample(new_sample)
    assert len(numeric_information_gain_learner.negative_samples_df) == 4
    assert len(numeric_information_gain_learner.negative_samples_df.columns) == 4


def testadd_negative_sample_does_not_add_to_positive_samples(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    numeric_information_gain_learner.negative_samples_df = test_dataframe
    assert len(numeric_information_gain_learner.negative_samples_df) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    numeric_information_gain_learner.add_negative_sample(new_sample)
    assert len(numeric_information_gain_learner.positive_samples_df) == 0


def test_remove_false_negative_sample_removes_the_correct_sample_from_the_negative_samples(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 5, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 3, 3],
        "(w)": [1, 9, 3]
    })
    numeric_information_gain_learner.negative_samples_df = test_dataframe
    sample_to_remove = {}
    row = test_dataframe.iloc[1]
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(row[function_name])
        sample_to_remove[function_name] = new_func

    numeric_information_gain_learner.remove_false_negative_sample(sample_to_remove)
    assert len(numeric_information_gain_learner.negative_samples_df) == 2
    assert len(numeric_information_gain_learner.negative_samples_df.columns) == 4
    assert numeric_information_gain_learner.negative_samples_df.loc[0, "(x)"] == 1
    assert numeric_information_gain_learner.negative_samples_df.loc[0, "(y)"] == 1
    assert numeric_information_gain_learner.negative_samples_df.loc[0, "(z)"] == 1
    assert numeric_information_gain_learner.negative_samples_df.loc[0, "(w)"] == 1
    assert numeric_information_gain_learner.negative_samples_df.loc[2, "(x)"] == 3
    assert numeric_information_gain_learner.negative_samples_df.loc[2, "(y)"] == 3
    assert numeric_information_gain_learner.negative_samples_df.loc[2, "(z)"] == 3
    assert numeric_information_gain_learner.negative_samples_df.loc[2, "(w)"] == 3


def test_remove_false_negative_sample_does_not_remove_anything_if_the_sample_is_not_in_the_negative_samples(
        numeric_information_gain_learner: NumericInformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 5, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 3, 3],
        "(w)": [1, 9, 3]
    })
    numeric_information_gain_learner.negative_samples_df = test_dataframe
    sample_to_remove = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(12)
        sample_to_remove[function_name] = new_func

    numeric_information_gain_learner.remove_false_negative_sample(sample_to_remove)
    assert len(numeric_information_gain_learner.negative_samples_df) == 3
    assert len(numeric_information_gain_learner.negative_samples_df.columns) == 4


def test_in_hull_captures_that_a_point_is_in_a_convex_hull_in_a_2d_plane(
        numeric_information_gain_learner: NumericInformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [0.5]
    })
    convex_hull_array = hull_df.to_numpy()
    point_to_test_array = point_to_test.to_numpy()
    assert numeric_information_gain_learner._in_hull(point_to_test_array, convex_hull_array)


def test_in_hull_captures_that_a_point_is_not_in_a_convex_hull_in_a_2d_plane(
        numeric_information_gain_learner: NumericInformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [2],
        "(y)": [2]
    })
    convex_hull_array = hull_df.to_numpy()
    point_to_test_array = point_to_test.to_numpy()
    assert not numeric_information_gain_learner._in_hull(point_to_test_array, convex_hull_array)


def test_in_hull_captures_that_more_than_one_point_is_in_2d_convex_hull(
        numeric_information_gain_learner: NumericInformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5, 0.6, 0.7, 2, 10],
        "(y)": [0.5, 0.6, 0.7, 2, 10]
    })
    convex_hull_array = hull_df.to_numpy()
    points_to_test_array = points_to_test.to_numpy()
    assert numeric_information_gain_learner._in_hull(points_to_test_array, convex_hull_array)


def test_calculate_information_gain_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_negative(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [1, 2],
        "(y)": [1, 2],
        "(z)": [1, 2],
        "(w)": [1, 2]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [3, 4],
        "(y)": [3, 4],
        "(z)": [3, 4],
        "(w)": [3, 4]
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4)
        new_sample[function_name] = new_func

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) == 0


def test_calculate_information_gain_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_positive(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [1, 2],
        "(y)": [1, 2],
        "(z)": [1, 2],
        "(w)": [1, 2]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [3, 4],
        "(y)": [3, 4],
        "(z)": [3, 4],
        "(w)": [3, 4]
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1)
        new_sample[function_name] = new_func

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) == 0


def test_calculate_information_gain_when_not_enough_points_to_create_convex_hull_verifies_that_the_point_was_not_observed_in_the_negative_and_positive_samples(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [1, 2],
        "(y)": [1, 2],
        "(z)": [1, 2],
        "(w)": [1, 2]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [3, 4],
        "(y)": [3, 4],
        "(z)": [3, 4],
        "(w)": [3, 4]
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(index)
        new_sample[function_name] = new_func

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) > 0


def test_calculate_information_gain_when_can_create_convex_hull_and_point_is_in_ch_returns_that_the_point_is_not_informative(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [3, 4],
        "(y)": [3, 4],
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}
    for function_name in ["(x)", "(y)"]:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_sample[function_name] = new_func

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) == 0


def test_calculate_information_gain_returns_zero_when_new_point_combined_with_positive_points_creates_a_hull_that_includes_a_negative_sample(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [0.5, 4],
        "(y)": [1.5, 4],
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(2.0)
    new_sample["(y)"] = y_function

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) == 0


def test_calculate_information_gain_returns_value_greater_than_zero_when_new_point_should_be_informative(
        numeric_information_gain_learner: NumericInformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [1.5],
    })
    numeric_information_gain_learner.positive_samples_df = positive_samples_df
    numeric_information_gain_learner.negative_samples_df = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(-0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(-0.5)
    new_sample["(y)"] = y_function

    assert numeric_information_gain_learner.calculate_sample_information_gain(new_sample) > 0

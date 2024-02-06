"""Module test for the consistent model validator."""
from typing import List

import pandas as pd
import pytest
from pddl_plus_parser.models import Predicate, Domain

from sam_learning.core import NumericConsistencyValidator

TEST_ACTION_NAME = 'test_action'
TEST_FUNCTION_NAMES = ["(x)", "(y)", "(z)", "(w)"]
TEST_PREDICATE_NAMES = ["p", "q", "r", "s"]


@pytest.fixture
def test_predicates() -> List[Predicate]:
    predicates = []
    for p in TEST_PREDICATE_NAMES:
        predicates.append(Predicate(name=p, signature={}, is_positive=True))
        predicates.append(Predicate(name=p, signature={}, is_positive=False))

    return predicates


@pytest.fixture
def consistent_model_validator() -> NumericConsistencyValidator:
    model_validator = NumericConsistencyValidator(action_name=TEST_ACTION_NAME)
    model_validator.init_numeric_dataframes(valid_lifted_functions=TEST_FUNCTION_NAMES)
    return model_validator


@pytest.fixture
def depot_numeric_model_validator(depot_domain: Domain) -> NumericConsistencyValidator:
    model_validator = NumericConsistencyValidator(action_name="drive")
    model_validator.init_numeric_dataframes(valid_lifted_functions=list(depot_domain.functions.keys()))
    return model_validator


def test_locate_sample_in_df_locates_that_samples_exists_in_the_dataframe_and_returns_its_index(
        consistent_model_validator: NumericConsistencyValidator):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    sample_to_locate = [1, 1, 1, 1]
    assert consistent_model_validator._locate_sample_in_df(sample_to_locate, test_dataframe) == 0


def test_locate_sample_in_df_can_understand_that_sample_does_not_exists_in_dataframe_and_return_minus_one(
        consistent_model_validator: NumericConsistencyValidator):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    sample_to_locate = [1, 2, 1, 2]
    assert consistent_model_validator._locate_sample_in_df(sample_to_locate, test_dataframe) == -1


def test_in_hull_captures_that_a_point_is_in_a_convex_hull_in_a_2d_plane(
        consistent_model_validator: NumericConsistencyValidator):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [0.5]
    })
    assert consistent_model_validator._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_a_point_is_not_in_a_convex_hull_in_a_2d_plane(
        consistent_model_validator: NumericConsistencyValidator):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [2],
        "(y)": [2]
    })
    assert not consistent_model_validator._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_2d_convex_hull(
        consistent_model_validator: NumericConsistencyValidator):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5, 0.6, 0.7, 2, 10],
        "(y)": [0.5, 0.6, 0.7, 2, 10]
    })
    assert consistent_model_validator._in_hull(points_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_1d_convex_hull(
        consistent_model_validator: NumericConsistencyValidator):
    hull_df = pd.DataFrame({
        "(x)": [0, 1],
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5, 0.6, 0.7],
    })
    assert consistent_model_validator._in_hull(points_to_test, hull_df)


def test_in_hull_captures_that_point_is_in_1d_hull_when_given_only_one_test_point(
        consistent_model_validator: NumericConsistencyValidator):
    hull_df = pd.DataFrame({
        "(x)": [0, 1],
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5],
    })
    assert consistent_model_validator._in_hull(points_to_test, hull_df)


def test_can_determine_effects_effects_perfectly_returns_true_when_indicator_is_set_to_true_and_does_not_do_any_calculations(
        consistent_model_validator: NumericConsistencyValidator):
    consistent_model_validator._effects_learned_perfectly = True
    assert consistent_model_validator.can_determine_numeric_effects_perfectly()


def test_can_determine_effects_effects_perfectly_returns_false_when_only_one_sample_is_observed(
        consistent_model_validator: NumericConsistencyValidator):
    positive_samples_df = pd.DataFrame({
        "(x)": [1],
        "(y)": [0]
    })
    consistent_model_validator.numeric_positive_samples = positive_samples_df
    assert not consistent_model_validator.can_determine_numeric_effects_perfectly()


def test_can_determine_effects_effects_perfectly_returns_false_when_the_indicator_is_not_set_and_there_is_not_enough_data_to_learn_the_effects(
        consistent_model_validator: NumericConsistencyValidator):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1],
        "(y)": [2, 1]
    })
    consistent_model_validator.numeric_positive_samples = positive_samples_df
    assert not consistent_model_validator.can_determine_numeric_effects_perfectly()

def test_can_determine_effects_effects_perfectly_returns_true_when_the_indicator_is_not_set_and_there_is_enough_data_to_learn_the_effects(
        depot_numeric_model_validator: NumericConsistencyValidator):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]
    })
    depot_numeric_model_validator.numeric_positive_samples = positive_samples_df
    assert depot_numeric_model_validator.can_determine_numeric_effects_perfectly()

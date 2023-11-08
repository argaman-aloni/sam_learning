"""Module tests for the numeric information gaining process."""
import random
from typing import List

import pandas as pd
import pytest
from pddl_plus_parser.models import PDDLFunction, Predicate, Domain, Observation

from sam_learning.core import InformationGainLearner

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
def information_gain_learner_no_predicates() -> InformationGainLearner:
    information_gain = InformationGainLearner(action_name=TEST_ACTION_NAME)
    information_gain.init_dataframes(valid_lifted_functions=TEST_FUNCTION_NAMES, lifted_predicates=[])
    return information_gain


@pytest.fixture
def information_gain_learner_with_predicates(test_predicates: List[Predicate]) -> InformationGainLearner:
    information_gain = InformationGainLearner(action_name=TEST_ACTION_NAME)
    information_gain.init_dataframes(valid_lifted_functions=TEST_FUNCTION_NAMES,
                                     lifted_predicates=[p.untyped_representation for p in test_predicates])
    return information_gain


@pytest.fixture
def depot_numeric_information_gain_learner(depot_domain: Domain) -> InformationGainLearner:
    information_gain = InformationGainLearner(action_name="drive")
    information_gain.init_dataframes(
        valid_lifted_functions=list(depot_domain.functions.keys()),
        lifted_predicates=[p.untyped_representation for p in depot_domain.predicates.values()])
    return information_gain


def test_locate_sample_in_df_locates_that_samples_exists_in_the_dataframe_and_returns_its_index(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    sample_to_locate = [1, 1, 1, 1]
    assert information_gain_learner_no_predicates._locate_sample_in_df(sample_to_locate, test_dataframe) == 0


def test_locate_sample_in_df_can_understand_that_sample_does_not_exists_in_dataframe_and_return_minus_one(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    sample_to_locate = [1, 2, 1, 2]
    assert information_gain_learner_no_predicates._locate_sample_in_df(sample_to_locate, test_dataframe) == -1


def test_add_positive_sample_adds_new_sample_to_the_existing_positive_dataframe(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = test_dataframe
    assert len(information_gain_learner_no_predicates.numeric_positive_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    information_gain_learner_no_predicates.add_positive_sample(new_sample, [])
    assert len(information_gain_learner_no_predicates.numeric_positive_samples) == 4
    assert len(information_gain_learner_no_predicates.numeric_positive_samples.columns) == 4


def test_add_positive_sample_does_not_add_to_negative_samples(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = test_dataframe
    assert len(information_gain_learner_no_predicates.numeric_positive_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    information_gain_learner_no_predicates.add_positive_sample(new_sample, [])
    assert len(information_gain_learner_no_predicates.numeric_negative_samples) == 0


def test_add_positive_sample_when_domain_contains_numeric_and_discrete_parts_removes_columns_that_are_no_longer_relevant_for_calculation(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    assert len(information_gain_learner_with_predicates.lifted_predicates) == 8
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 8
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(
        TEST_FUNCTION_NAMES)

    information_gain_learner_with_predicates.add_positive_sample(new_numeric_sample, new_discrete_sample)
    assert len(information_gain_learner_with_predicates.lifted_predicates) == 4
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 4
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(
        TEST_FUNCTION_NAMES)


def test_add_positive_sample_when_domain_contains_numeric_and_discrete_parts_adds_new_sample_to_the_existing_positive_dataframe_with_the_correct_values(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df) == 0
    assert len(information_gain_learner_with_predicates.numeric_positive_samples) == 0
    information_gain_learner_with_predicates.add_positive_sample(new_numeric_sample, new_discrete_sample)
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df) == 1
    assert len(information_gain_learner_with_predicates.numeric_positive_samples) == 1
    discrete_positive_sample = information_gain_learner_with_predicates.positive_discrete_sample_df.iloc[0]
    numeric_positive_sample = information_gain_learner_with_predicates.numeric_positive_samples.iloc[0]
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        assert numeric_positive_sample[function_name] == 4 + index

    for index, predicate in enumerate(new_discrete_sample):
        assert discrete_positive_sample[predicate.untyped_representation] == 1


def test_add_negative_sample_when_domain_contains_numeric_and_discrete_parts_does_not_change_the_dimensionality_of_the_positive_dataframes(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    assert len(information_gain_learner_with_predicates.lifted_predicates) == 8
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 8
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(
        TEST_FUNCTION_NAMES)

    information_gain_learner_with_predicates.add_negative_sample(new_numeric_sample, new_discrete_sample)
    assert len(information_gain_learner_with_predicates.lifted_predicates) == 8
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 8
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(
        TEST_FUNCTION_NAMES)


def test_add_negative_sample_when_domain_contains_numeric_and_discrete_parts_the_values_are_being_added_to_the_negative_dataframe_correctly(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.add_negative_sample(new_numeric_sample, new_discrete_sample)
    discrete_negative_sample = information_gain_learner_with_predicates.negative_combined_sample_df.iloc[0]
    numeric_negative_sample = information_gain_learner_with_predicates.numeric_negative_samples.iloc[0]
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        assert numeric_negative_sample[function_name] == 4 + index

    for predicate in test_predicates:
        if predicate in new_discrete_sample:
            assert discrete_negative_sample[predicate.untyped_representation] == 1
        else:
            assert discrete_negative_sample[predicate.untyped_representation] == 0


def test_add_negative_sample_adds_new_sample_to_the_existing_negative_numeric_dataframe(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    information_gain_learner_no_predicates.numeric_negative_samples = test_dataframe
    assert len(information_gain_learner_no_predicates.numeric_negative_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    information_gain_learner_no_predicates.add_negative_sample(new_sample, [])
    assert len(information_gain_learner_no_predicates.numeric_negative_samples) == 4
    assert len(information_gain_learner_no_predicates.numeric_negative_samples.columns) == 4


def test_add_negative_sample_does_not_add_to_positive_samples(
        information_gain_learner_no_predicates: InformationGainLearner):
    test_dataframe = pd.DataFrame({
        "(x)": [1, 2, 3],
        "(y)": [1, 2, 3],
        "(z)": [1, 2, 3],
        "(w)": [1, 2, 3]
    })
    information_gain_learner_no_predicates.numeric_negative_samples = test_dataframe
    assert len(information_gain_learner_no_predicates.numeric_negative_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    information_gain_learner_no_predicates.add_negative_sample(new_sample, [])
    assert len(information_gain_learner_no_predicates.numeric_positive_samples) == 0


def test_in_hull_captures_that_a_point_is_in_a_convex_hull_in_a_2d_plane(
        information_gain_learner_no_predicates: InformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [0.5]
    })
    assert information_gain_learner_no_predicates._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_a_point_is_not_in_a_convex_hull_in_a_2d_plane(
        information_gain_learner_no_predicates: InformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    point_to_test = pd.DataFrame({
        "(x)": [2],
        "(y)": [2]
    })
    assert not information_gain_learner_no_predicates._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_2d_convex_hull(
        information_gain_learner_no_predicates: InformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5, 0.6, 0.7, 2, 10],
        "(y)": [0.5, 0.6, 0.7, 2, 10]
    })
    assert information_gain_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_1d_convex_hull(
        information_gain_learner_no_predicates: InformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 1],
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5, 0.6, 0.7],
    })
    assert information_gain_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_validate_action_discrete_preconditions_hold_in_state_when_predicates_are_valid_returns_true(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    information_gain_learner_with_predicates.lifted_predicates = [
        p.untyped_representation for p in test_predicates[:-1]]
    assert information_gain_learner_with_predicates._validate_action_discrete_preconditions_hold_in_state(
        test_predicates)


def test_validate_action_discrete_preconditions_hold_in_state_when_predicates_are_are_partial_from_required_returns_false(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    
    information_gain_learner_with_predicates.lifted_predicates = [
        p.untyped_representation for p in test_predicates]
    assert not information_gain_learner_with_predicates._validate_action_discrete_preconditions_hold_in_state(
        test_predicates[:-1])


def test_in_hull_captures_that_point_is_in_1d_hull_when_given_only_one_test_point(
        information_gain_learner_no_predicates: InformationGainLearner):
    hull_df = pd.DataFrame({
        "(x)": [0, 1],
    })
    points_to_test = pd.DataFrame({
        "(x)": [0.5],
    })
    assert information_gain_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_is_non_informative_safe_returns_true_when_the_point_is_in_the_positive_samples(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_sample[function_name] = new_func

    assert information_gain_learner_no_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_false_when_the_point_is_not_in_the_ch(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1.5)
        new_sample[function_name] = new_func

    assert not information_gain_learner_no_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_false_when_the_point_is_not_in_range_when_only_one_function_exists(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1],

    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1.5)
        new_sample[function_name] = new_func

    assert not information_gain_learner_no_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_true_when_the_point_is_in_range_when_only_one_function_exists(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1],

    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1)
        new_sample[function_name] = new_func

    assert information_gain_learner_no_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_true_when_the_predicates_in_the_state_are_superset_of_existing_predicates(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1, 0, 1],
        "(y)": [0, 0, 1, 1],

    })
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    assert information_gain_learner_with_predicates._is_non_informative_safe(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample)


def test_is_non_informative_safe_returns_false_when_predicates_are_contained_in_state_but_the_numeric_part_is_not_safe(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(2.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1, 0, 1],
        "(y)": [0, 0, 1, 1],

    })
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    assert not information_gain_learner_with_predicates._is_non_informative_safe(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=test_predicates)


def test_is_non_informative_safe_returns_false_when_some_predicates_are_missing_but_the_ch_is_correct(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    positive_samples_df = pd.DataFrame({
        "(x)": [0, 1, 0, 1],
        "(y)": [0, 0, 1, 1],

    })
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in test_predicates]
    assert not information_gain_learner_with_predicates._is_non_informative_safe(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample)


def test_is_non_informative_safe_returns_action_informative_when_not_observed_before(
        depot_numeric_information_gain_learner: InformationGainLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    state_predicates = []
    for predicate_set in initial_state.state_predicates.values():
        state_predicates.extend(predicate_set)

    assert not depot_numeric_information_gain_learner._is_non_informative_safe(
        new_numeric_sample=initial_state.state_fluents, new_propositional_sample=state_predicates)


def test_is_non_informative_safe_returns_false_when_the_relevant_fluents_is_empty_and_the_predicates_are_all_in_the_preconditions_and_the_effects_are_not_determined_perfectly(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not information_gain_learner_with_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample,
        new_propositional_sample=[p for p in test_predicates if p in new_discrete_sample],
        relevant_numeric_features=[])


def test_is_non_informative_safe_returns_true_when_the_relevant_fluents_is_empty_and_the_predicates_are_all_in_the_preconditions_and_the_effects_are_determined_perfectly(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func
    information_gain_learner_with_predicates._effects_learned_perfectly = True
    assert information_gain_learner_with_predicates._is_non_informative_safe(
        new_numeric_sample=new_sample,
        new_propositional_sample=[p for p in test_predicates if p in new_discrete_sample],
        relevant_numeric_features=[])


def test_is_non_informative_unsafe_returns_true_when_a_negative_point_is_in_the_ch_created_from_the_combined_model(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })
    negative_samples_df = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [1.5],
    })

    information_gain_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_no_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_unsafe_returns_true_when_a_negative_point_is_in_the_ch_created_from_the_combined_model_and_the_discrete_model_is_correct(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    negative_discrete_data = {p.untyped_representation: [1] for p in new_discrete_sample}

    negative_samples_df = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [1.5],
        **negative_discrete_data
    })
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]

    information_gain_learner_with_predicates.negative_combined_sample_df = negative_samples_df
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_with_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample, new_propositional_sample=new_discrete_sample)


def test_is_non_informative_unsafe_returns_true_when_a_non_of_the_negative_samples_match_the_discrete_part_of_the_model(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    negative_discrete_data = {p.untyped_representation: [int(index % 2 == 0)] for index, p in
                              enumerate(new_discrete_sample)}

    negative_samples_df = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [1.5],
        **negative_discrete_data
    })
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]

    information_gain_learner_with_predicates.numeric_negative_samples = negative_samples_df
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not information_gain_learner_with_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample, new_propositional_sample=new_discrete_sample)


def test_is_non_informative_unsafe_returns_true_when_the_relevant_fluents_is_empty_and_the_predicates_are_not_in_the_state(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_with_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample,
        new_propositional_sample=[p for p in test_predicates if p not in new_discrete_sample],
        relevant_numeric_features=[])


def test_is_non_informative_unsafe_returns_true_when_the_relevant_fluents_is_empty_and_some_predicates_are_in_the_state(
        information_gain_learner_with_predicates: InformationGainLearner, test_predicates: List[Predicate]):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_with_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample,
        new_propositional_sample=random.choices(new_discrete_sample, k=2),
        relevant_numeric_features=[])


def test_is_non_informative_unsafe_returns_false_when_no_negative_point_is_in_the_ch_created_from_the_combined_model(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.

    })
    negative_samples_df = pd.DataFrame({
        "(x)": [2.5, 3.5],
        "(y)": [4.5, 5.5],
    })

    information_gain_learner_no_predicates.negative_samples_df = negative_samples_df
    information_gain_learner_no_predicates.positive_samples_df = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not information_gain_learner_no_predicates._is_non_informative_unsafe(
        new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_negative(
        information_gain_learner_no_predicates: InformationGainLearner):
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
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4)
        new_sample[function_name] = new_func

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_positive(
        information_gain_learner_no_predicates: InformationGainLearner):
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
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1)
        new_sample[function_name] = new_func

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_that_the_point_was_not_observed_in_the_negative_and_positive_samples(
        information_gain_learner_no_predicates: InformationGainLearner):
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
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(index)
        new_sample[function_name] = new_func

    assert information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_can_create_convex_hull_and_point_is_in_ch_returns_that_the_point_is_not_informative(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [3, 4],
        "(y)": [3, 4],
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for function_name in ["(x)", "(y)"]:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_sample[function_name] = new_func

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_returns_zero_when_new_point_combined_with_positive_points_creates_a_hull_that_includes_a_negative_sample(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [0.5, 4],
        "(y)": [1.5, 4],
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(2.0)
    new_sample["(y)"] = y_function

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_returns_value_greater_than_zero_when_new_point_should_be_informative(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1]  # The convex hull is a square.
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [0.5],
        "(y)": [1.5],
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(-0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(-0.5)
    new_sample["(y)"] = y_function

    assert information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_the_convex_hull_only_on_the_relevant_part_when_consts_are_equal(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 1, 0, 1],
        "(z)": [0, 0, 0, 0],
        "(w)": [1, 1, 1, 1]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [2.5],
        "(y)": [1.5],
        "(z)": [1.0],
        "(w)": [1.1]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(0.5)
    new_sample["(y)"] = y_function
    z_function = PDDLFunction(name="(z)", signature={})
    z_function.set_value(0.0)
    new_sample["(z)"] = z_function
    w_function = PDDLFunction(name="(w)", signature={})
    w_function.set_value(1.0)
    new_sample["(w)"] = w_function

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_min_max_values_only_on_the_relevant_part_when_consts_are_equal(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 0, 2, 2],
        "(z)": [0, 0, 0, 0],
        "(w)": [1, 1, 1, 1]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [2.5],
        "(y)": [1.5],
        "(z)": [1.0],
        "(w)": [1.1]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(0.75)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(1.5)
    new_sample["(y)"] = y_function
    z_function = PDDLFunction(name="(z)", signature={})
    z_function.set_value(0.0)
    new_sample["(z)"] = z_function
    w_function = PDDLFunction(name="(w)", signature={})
    w_function.set_value(1.0)
    new_sample["(w)"] = w_function

    assert not information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_min_max_values_only_on_the_relevant_part_when_consts_are_equal_and_value_out_of_range(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 0, 2, 2],
        "(z)": [0, 0, 0, 0],
        "(w)": [1, 1, 1, 1]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [2.5],
        "(y)": [1.5],
        "(z)": [1.0],
        "(w)": [1.1]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(1.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(3)
    new_sample["(y)"] = y_function
    z_function = PDDLFunction(name="(z)", signature={})
    z_function.set_value(0.0)
    new_sample["(z)"] = z_function
    w_function = PDDLFunction(name="(w)", signature={})
    w_function.set_value(1.0)
    new_sample["(w)"] = w_function

    assert information_gain_learner_no_predicates.is_sample_informative(new_sample, [])


def test_is_sample_informative_when_using_relevant_features_with_a_single_features_only_examines_only_that_feature_for_the_informative_calculation(
        information_gain_learner_no_predicates: InformationGainLearner):
    positive_samples_df = pd.DataFrame({
        "(x)": [0, 0, 1, 1],
        "(y)": [0, 0, 2, 2],
        "(z)": [0, 0, 0, 0],
        "(w)": [1, 1, 1, 1]
    })
    negative_samples_df = pd.DataFrame({
        "(x)": [2.5],
        "(y)": [1.5],
        "(z)": [1.0],
        "(w)": [1.1]
    })
    information_gain_learner_no_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(1.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(3)
    new_sample["(y)"] = y_function
    z_function = PDDLFunction(name="(z)", signature={})
    z_function.set_value(0.0)
    new_sample["(z)"] = z_function
    w_function = PDDLFunction(name="(w)", signature={})
    w_function.set_value(1.0)
    new_sample["(w)"] = w_function

    assert not information_gain_learner_no_predicates.is_sample_informative(
        new_sample, [], relevant_numeric_features=["(z)"])

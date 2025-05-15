"""Module tests for the numeric information gaining process."""
import random
from typing import List, Set, Dict

import pandas as pd
import pytest
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Predicate, Domain, Observation

from sam_learning.core import VocabularyCreator
from sam_learning.core.online_learning import IncrementalConvexHullLearner, InformationStatesLearner, OnlineDiscreteModelLearner

TEST_ACTION_NAME = "test_action"
TEST_FUNCTION_NAMES = ["(x)", "(y)", "(z)", "(w)"]
TEST_PREDICATE_NAMES = ["p", "q", "r", "s"]


@pytest.fixture
def lifted_depot_vocabulary(depot_domain: Domain) -> Set[Predicate]:
    return VocabularyCreator().create_lifted_vocabulary(domain=depot_domain, possible_parameters=depot_domain.actions["drive"].signature)


@pytest.fixture
def online_discrete_model_learner(lifted_depot_vocabulary: Set[Predicate]) -> OnlineDiscreteModelLearner:
    return OnlineDiscreteModelLearner(TEST_ACTION_NAME, lifted_depot_vocabulary)


@pytest.fixture
def parameter_bound_function_vocabulary(depot_domain: Domain) -> Dict[str, PDDLFunction]:
    return VocabularyCreator().create_lifted_functions_vocabulary(domain=depot_domain, possible_parameters=depot_domain.actions["drive"].signature)


@pytest.fixture
def incremental_convex_hull_learner(parameter_bound_function_vocabulary: Dict[str, PDDLFunction]) -> IncrementalConvexHullLearner:
    return IncrementalConvexHullLearner(TEST_ACTION_NAME, parameter_bound_function_vocabulary)


@pytest.fixture
def informative_states_learner_no_predicates(incremental_convex_hull_learner: IncrementalConvexHullLearner) -> InformationStatesLearner:
    information_gain = InformationStatesLearner(
        action_name=TEST_ACTION_NAME,
        discrete_model_learner=OnlineDiscreteModelLearner(action_name=TEST_ACTION_NAME, lifted_predicates=set()),
        convex_hull_learner=incremental_convex_hull_learner,
    )
    return information_gain


@pytest.fixture
def informative_states_learner_only_discrete(online_discrete_model_learner: OnlineDiscreteModelLearner) -> InformationStatesLearner:
    information_gain = InformationStatesLearner(
        action_name=TEST_ACTION_NAME,
        discrete_model_learner=online_discrete_model_learner,
        convex_hull_learner=IncrementalConvexHullLearner(TEST_ACTION_NAME, {}),
    )
    return information_gain


@pytest.fixture
def depot_informative_states_learner(
    online_discrete_model_learner: OnlineDiscreteModelLearner, incremental_convex_hull_learner: IncrementalConvexHullLearner
) -> InformationStatesLearner:
    information_gain = InformationStatesLearner(
        action_name="drive", discrete_model_learner=online_discrete_model_learner, convex_hull_learner=incremental_convex_hull_learner,
    )
    return information_gain


def test_create_combined_sample_when_domain_does_not_have_predicates_and_only_functions_data_creates_a_dataframe_with_the_correct_columns(
    informative_states_learner_no_predicates: InformationStatesLearner, parameter_bound_function_vocabulary: Dict[str, PDDLFunction]
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.name] = new_func

    combined_df = informative_states_learner_no_predicates._create_combined_sample_data(new_numeric_sample, set())
    assert len(combined_df) == 1


def test_locate_sample_in_df_locates_that_samples_exists_in_the_dataframe_and_returns_its_index(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    sample_to_locate = [1, 1, 1, 1]
    assert informative_states_learner_no_predicates._locate_sample_in_df(sample_to_locate, test_dataframe) == 0


def test_locate_sample_in_df_can_understand_that_sample_does_not_exists_in_dataframe_and_return_minus_one(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    sample_to_locate = [1, 2, 1, 2]
    assert informative_states_learner_no_predicates._locate_sample_in_df(sample_to_locate, test_dataframe) == -1


def test_add_positive_sample_adds_new_sample_to_the_existing_positive_dataframe(informative_states_learner_no_predicates: InformationStatesLearner):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    informative_states_learner_no_predicates.numeric_positive_samples = test_dataframe
    assert len(informative_states_learner_no_predicates.numeric_positive_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    informative_states_learner_no_predicates.add_positive_sample(new_sample, [])
    assert len(informative_states_learner_no_predicates.numeric_positive_samples) == 4
    assert len(informative_states_learner_no_predicates.numeric_positive_samples.columns) == 4


def test_add_positive_sample_does_not_add_to_negative_samples(informative_states_learner_no_predicates: InformationStatesLearner):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    informative_states_learner_no_predicates.numeric_positive_samples = test_dataframe
    assert len(informative_states_learner_no_predicates.numeric_positive_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    informative_states_learner_no_predicates.add_positive_sample(new_sample, [])
    assert len(informative_states_learner_no_predicates.numeric_negative_samples) == 0


def test_add_positive_sample_when_domain_contains_numeric_and_discrete_parts_removes_columns_that_are_no_longer_relevant_for_calculation(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 8
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(TEST_FUNCTION_NAMES)

    information_gain_learner_with_predicates.add_positive_sample(new_numeric_sample, new_discrete_sample)
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 4
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(TEST_FUNCTION_NAMES)


def test_add_positive_sample_when_domain_contains_numeric_and_discrete_parts_adds_new_sample_to_the_existing_positive_dataframe_with_the_correct_values(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
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
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
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
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(TEST_FUNCTION_NAMES)

    information_gain_learner_with_predicates.add_negative_sample(new_numeric_sample, new_discrete_sample)
    assert len(information_gain_learner_with_predicates.lifted_predicates) == 8
    assert len(information_gain_learner_with_predicates.positive_discrete_sample_df.columns.tolist()) == 8
    assert len(information_gain_learner_with_predicates.numeric_positive_samples.columns.tolist()) == len(TEST_FUNCTION_NAMES)


def test_add_negative_sample_when_domain_contains_numeric_and_discrete_parts_the_values_are_being_added_to_the_negative_dataframe_correctly(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
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
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    informative_states_learner_no_predicates.numeric_negative_samples = test_dataframe
    assert len(informative_states_learner_no_predicates.numeric_negative_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    informative_states_learner_no_predicates.add_negative_sample(new_sample, [])
    assert len(informative_states_learner_no_predicates.numeric_negative_samples) == 4
    assert len(informative_states_learner_no_predicates.numeric_negative_samples.columns) == 4


def test_add_negative_sample_does_not_add_to_positive_samples(informative_states_learner_no_predicates: InformationStatesLearner):
    test_dataframe = pd.DataFrame({"(x)": [1, 2, 3], "(y)": [1, 2, 3], "(z)": [1, 2, 3], "(w)": [1, 2, 3]})
    informative_states_learner_no_predicates.numeric_negative_samples = test_dataframe
    assert len(informative_states_learner_no_predicates.numeric_negative_samples) == 3
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_sample[function_name] = new_func

    informative_states_learner_no_predicates.add_negative_sample(new_sample, [])
    assert len(informative_states_learner_no_predicates.numeric_positive_samples) == 0


def test_in_hull_captures_that_a_point_is_in_a_convex_hull_in_a_2d_plane(informative_states_learner_no_predicates: InformationStatesLearner):
    hull_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    point_to_test = pd.DataFrame({"(x)": [0.5], "(y)": [0.5]})
    assert informative_states_learner_no_predicates._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_a_point_is_not_in_a_convex_hull_in_a_2d_plane(informative_states_learner_no_predicates: InformationStatesLearner):
    hull_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    point_to_test = pd.DataFrame({"(x)": [2], "(y)": [2]})
    assert not informative_states_learner_no_predicates._in_hull(point_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_2d_convex_hull(informative_states_learner_no_predicates: InformationStatesLearner):
    hull_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    points_to_test = pd.DataFrame({"(x)": [0.5, 0.6, 0.7, 2, 10], "(y)": [0.5, 0.6, 0.7, 2, 10]})
    assert informative_states_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_in_hull_captures_that_more_than_one_point_is_in_1d_convex_hull(informative_states_learner_no_predicates: InformationStatesLearner):
    hull_df = pd.DataFrame({"(x)": [0, 1],})
    points_to_test = pd.DataFrame({"(x)": [0.5, 0.6, 0.7],})
    assert informative_states_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_validate_action_discrete_preconditions_hold_in_state_when_predicates_are_valid_returns_true(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in test_predicates[:-1]}
    )
    assert information_gain_learner_with_predicates._validate_action_discrete_preconditions_hold_in_state(test_predicates)


def test_validate_action_discrete_preconditions_hold_in_state_when_no_positive_sample_was_observed_returns_by_default_false(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    # Since no positive observation was encountered cannot know if the preconditions hold or not.
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(columns=[p.untyped_representation for p in test_predicates])
    assert not information_gain_learner_with_predicates._validate_action_discrete_preconditions_hold_in_state(test_predicates[:-1])


def test_validate_action_discrete_preconditions_hold_in_state_when_predicates_are_are_partial_from_required_returns_false(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [random.randint(0, 1) for _ in range(10)] for p in test_predicates}
    )
    assert not information_gain_learner_with_predicates._validate_action_discrete_preconditions_hold_in_state(test_predicates[:-1])


def test_in_hull_captures_that_point_is_in_1d_hull_when_given_only_one_test_point(informative_states_learner_no_predicates: InformationStatesLearner):
    hull_df = pd.DataFrame({"(x)": [0, 1],})
    points_to_test = pd.DataFrame({"(x)": [0.5],})
    assert informative_states_learner_no_predicates._in_hull(points_to_test, hull_df)


def test_visited_previously_failed_execution_when_predicates_match_negative_samples_but_numeric_values_not_match(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    not_observed_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(18 + index)
        new_numeric_sample[function_name] = new_func

    information_gain_learner_with_predicates.add_negative_sample(new_numeric_sample, new_discrete_sample)
    assert not information_gain_learner_with_predicates._visited_previously_failed_execution(
        new_numeric_sample=not_observed_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_visited_previously_failed_execution_when_numeric_functions_match_negative_samples_but_predicates_do_not_match(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4 + index)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    not_observed_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

        elif index % 2 == 1:
            not_observed_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.add_negative_sample(new_numeric_sample, new_discrete_sample)
    assert not information_gain_learner_with_predicates._visited_previously_failed_execution(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=not_observed_discrete_sample
    )


def test_visited_previously_failed_execution_when_both_predicates_and_numeric_functions_match_return_true(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
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
    assert information_gain_learner_with_predicates._visited_previously_failed_execution(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_non_informative_safe_returns_true_when_the_point_is_in_the_positive_samples(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_sample[function_name] = new_func

    assert informative_states_learner_no_predicates._is_applicable_and_non_informative(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_false_when_the_point_is_not_in_the_ch(informative_states_learner_no_predicates: InformationStatesLearner):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1.5)
        new_sample[function_name] = new_func

    assert not informative_states_learner_no_predicates._is_applicable_and_non_informative(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_false_when_the_point_is_not_in_range_when_only_one_function_exists(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 1],})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1.5)
        new_sample[function_name] = new_func

    assert not informative_states_learner_no_predicates._is_applicable_and_non_informative(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_true_when_the_point_is_in_range_when_only_one_function_exists(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 1],})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    for function_name in positive_samples_df.columns:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1)
        new_sample[function_name] = new_func

    assert informative_states_learner_no_predicates._is_applicable_and_non_informative(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_safe_returns_true_when_the_predicates_in_the_state_are_superset_of_existing_predicates(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    positive_samples_df = pd.DataFrame({"(x)": [0, 1, 0, 1], "(y)": [0, 0, 1, 1],})
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )
    assert information_gain_learner_with_predicates._is_applicable_and_non_informative(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_non_informative_safe_returns_false_when_predicates_are_contained_in_state_but_the_numeric_part_is_not_safe(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(2.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    positive_samples_df = pd.DataFrame({"(x)": [0, 1, 0, 1], "(y)": [0, 0, 1, 1],})
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.lifted_predicates = [p.untyped_representation for p in new_discrete_sample]
    assert not information_gain_learner_with_predicates._is_applicable_and_non_informative(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=test_predicates
    )


def test_is_non_informative_safe_returns_false_when_some_predicates_are_missing_but_the_ch_is_correct(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_numeric_sample = {}
    for index, function_name in enumerate(["(x)", "(y)"]):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_numeric_sample[function_name] = new_func

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    positive_samples_df = pd.DataFrame({"(x)": [0, 1, 0, 1], "(y)": [0, 0, 1, 1],})
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1 for _ in range(10)] for p in test_predicates}
    )
    assert not information_gain_learner_with_predicates._is_applicable_and_non_informative(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_non_informative_safe_returns_action_informative_when_not_observed_before(
    depot_numeric_information_gain_learner: InformationStatesLearner, depot_observation: Observation
):
    initial_state = depot_observation.components[0].previous_state
    state_predicates = []
    for predicate_set in initial_state.state_predicates.values():
        state_predicates.extend(predicate_set)

    assert not depot_numeric_information_gain_learner._is_applicable_and_non_informative(
        new_numeric_sample=initial_state.state_fluents, new_propositional_sample=state_predicates
    )


def test_is_non_informative_safe_returns_false_when_the_relevant_fluents_is_empty_and_the_predicates_are_all_in_the_preconditions_and_the_effects_are_not_determined_perfectly(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
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

    assert not information_gain_learner_with_predicates._is_applicable_and_non_informative(
        new_numeric_sample=new_sample, new_propositional_sample=[p for p in test_predicates if p in new_discrete_sample], relevant_numeric_features=[]
    )


def test_is_non_informative_safe_returns_true_when_the_relevant_fluents_is_empty_and_the_predicates_are_all_in_the_preconditions_and_the_effects_are_determined_perfectly(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func
    information_gain_learner_with_predicates._effects_learned_perfectly = True
    assert information_gain_learner_with_predicates._is_applicable_and_non_informative(
        new_numeric_sample=new_sample, new_propositional_sample=[p for p in test_predicates if p in new_discrete_sample], relevant_numeric_features=[]
    )


def test_is_non_informative_unsafe_returns_true_when_a_negative_point_is_in_the_ch_created_from_the_combined_model(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    negative_samples_df = pd.DataFrame({"(x)": [0.5], "(y)": [1.5],})

    informative_states_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert informative_states_learner_no_predicates._is_definitely_not_applicable(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_non_informative_unsafe_returns_true_when_a_negative_point_is_in_the_ch_created_from_the_combined_model_and_the_discrete_model_is_correct(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    negative_discrete_data = {p.untyped_representation: [1 if p in new_discrete_sample else 0] for p in test_predicates}

    negative_samples_df = pd.DataFrame({"(x)": [0.5], "(y)": [1.5], **negative_discrete_data})
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )
    information_gain_learner_with_predicates.negative_combined_sample_df = negative_samples_df
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_non_informative_unsafe_returns_true_when_a_non_of_the_negative_samples_match_the_discrete_part_of_the_model(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.

    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    negative_discrete_data = {p.untyped_representation: [int(index % 2 == 0)] for index, p in enumerate(new_discrete_sample)}

    negative_samples_df = pd.DataFrame({"(x)": [0.5], "(y)": [1.5], **negative_discrete_data})
    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )

    information_gain_learner_with_predicates.numeric_negative_samples = negative_samples_df
    information_gain_learner_with_predicates.numeric_positive_samples = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_non_informative_unsafe_should_return_false_when_the_relevant_fluents_is_empty_and_the_state_predicates_do_not_appear_in_any_negative_sample(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_discrete_sample = []
    negative_samples = {p.untyped_representation: [] for p in test_predicates}
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())
            negative_samples[predicate.untyped_representation].append(0)
        else:
            negative_samples[predicate.untyped_representation].append(1)

    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    negative_samples[new_func.name] = [0.5]
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func
    negative_samples[new_func.name] = [1.5]
    information_gain_learner_with_predicates.negative_combined_sample_df = DataFrame(data=negative_samples)

    assert not information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample,
        new_propositional_sample=[p for p in test_predicates if p not in new_discrete_sample],
        relevant_numeric_features=[],
    )


def test_is_non_informative_unsafe_should_return_true_when_the_relevant_fluents_is_empty_and_the_state_predicates_appear_in_a_negative_sample_and_the_numeric_values_match_an_observed_state(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )

    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func
    information_gain_learner_with_predicates.add_negative_sample(new_sample, new_discrete_sample)
    assert information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample, new_propositional_sample=new_discrete_sample, relevant_numeric_features=[]
    )


def test_is_non_informative_unsafe_should_return_true_when_the_relevant_fluents_is_empty_and_the_state_predicates_appear_in_a_negative_sample_and_the_numeric_values_do_not_match_an_observed_state(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_discrete_sample = []
    negative_samples = {p.untyped_representation: [] for p in test_predicates}
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())
            negative_samples[predicate.untyped_representation].append(1)
        else:
            negative_samples[predicate.untyped_representation].append(0)

    negative_samples["(x)"] = [0.5]
    negative_samples["(y)"] = [2.2]

    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )
    information_gain_learner_with_predicates.negative_combined_sample_df = DataFrame(data=negative_samples)

    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample,
        new_propositional_sample=[p for p in test_predicates if p not in new_discrete_sample],
        relevant_numeric_features=[],
    )


def test_is_non_informative_unsafe_returns_false_when_the_relevant_fluents_is_empty_and_some_predicates_are_in_the_state_since_cannot_determine_if_not_informative(
    information_gain_learner_with_predicates: InformationStatesLearner, test_predicates: List[Predicate]
):
    new_discrete_sample = []
    for index, predicate in enumerate(test_predicates):
        if index % 2 != 0:
            new_discrete_sample.append(predicate.copy())

    information_gain_learner_with_predicates.positive_discrete_sample_df = DataFrame(
        data={p.untyped_representation: [1] for p in new_discrete_sample}
    )
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not information_gain_learner_with_predicates._is_definitely_not_applicable(
        new_numeric_sample=new_sample, new_propositional_sample=random.choices(new_discrete_sample, k=2), relevant_numeric_features=[]
    )


def test_is_non_informative_unsafe_returns_false_when_no_negative_point_is_in_the_ch_created_from_the_combined_model(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    negative_samples_df = pd.DataFrame({"(x)": [2.5, 3.5], "(y)": [4.5, 5.5],})

    informative_states_learner_no_predicates.negative_samples_df = negative_samples_df
    informative_states_learner_no_predicates.positive_samples_df = positive_samples_df
    new_sample = {}
    new_func = PDDLFunction(name="(x)", signature={})
    new_func.set_value(0.5)
    new_sample[new_func.name] = new_func
    new_func = PDDLFunction(name="(y)", signature={})
    new_func.set_value(2.0)
    new_sample[new_func.name] = new_func

    assert not informative_states_learner_no_predicates._is_definitely_not_applicable(new_numeric_sample=new_sample, new_propositional_sample=[])


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_negative(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [1, 2], "(y)": [1, 2], "(z)": [1, 2], "(w)": [1, 2]})
    negative_samples_df = pd.DataFrame({"(x)": [3, 4], "(y)": [3, 4], "(z)": [3, 4], "(w)": [3, 4]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(4)
        new_sample[function_name] = new_func

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_if_the_point_was_already_observed_positive(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [1, 2], "(y)": [1, 2], "(z)": [1, 2], "(w)": [1, 2]})
    negative_samples_df = pd.DataFrame({"(x)": [3, 4], "(y)": [3, 4], "(z)": [3, 4], "(w)": [3, 4]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for function_name in TEST_FUNCTION_NAMES:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(1)
        new_sample[function_name] = new_func

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_when_not_enough_points_to_create_convex_hull_verifies_that_the_point_was_not_observed_in_the_negative_and_positive_samples(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [1, 2], "(y)": [1, 2], "(z)": [1, 2], "(w)": [1, 2]})
    negative_samples_df = pd.DataFrame({"(x)": [3, 4], "(y)": [3, 4], "(z)": [3, 4], "(w)": [3, 4]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(index)
        new_sample[function_name] = new_func

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert informative


def test_is_sample_informative_when_can_create_convex_hull_and_point_is_in_ch_returns_that_the_point_is_not_informative(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    negative_samples_df = pd.DataFrame({"(x)": [3, 4], "(y)": [3, 4],})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}
    for function_name in ["(x)", "(y)"]:
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(0.5)
        new_sample[function_name] = new_func

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_returns_zero_when_new_point_combined_with_positive_points_creates_a_hull_that_includes_a_negative_sample(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    negative_samples_df = pd.DataFrame({"(x)": [0.5, 4], "(y)": [1.5, 4],})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.negative_combined_sample_df = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(2.0)
    new_sample["(y)"] = y_function

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_returns_value_greater_than_zero_when_new_point_should_be_informative(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1]})  # The convex hull is a square.
    negative_samples_df = pd.DataFrame({"(x)": [0.5], "(y)": [1.5],})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
    new_sample = {}

    x_function = PDDLFunction(name="(x)", signature={})
    x_function.set_value(-0.5)
    new_sample["(x)"] = x_function
    y_function = PDDLFunction(name="(y)", signature={})
    y_function.set_value(-0.5)
    new_sample["(y)"] = y_function

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert informative


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_the_convex_hull_only_on_the_relevant_part_when_consts_are_equal(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 1, 0, 1], "(z)": [0, 0, 0, 0], "(w)": [1, 1, 1, 1]})
    negative_samples_df = pd.DataFrame({"(x)": [2.5], "(y)": [1.5], "(z)": [1.0], "(w)": [1.1]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
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

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_min_max_values_only_on_the_relevant_part_when_consts_are_equal(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 0, 2, 2], "(z)": [0, 0, 0, 0], "(w)": [1, 1, 1, 1]})
    negative_samples_df = pd.DataFrame({"(x)": [2.5], "(y)": [1.5], "(z)": [1.0], "(w)": [1.1]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
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

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert not informative


def test_is_sample_informative_when_there_are_constant_features_in_the_dataset_checks_min_max_values_only_on_the_relevant_part_when_consts_are_equal_and_value_out_of_range(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 0, 2, 2], "(z)": [0, 0, 0, 0], "(w)": [1, 1, 1, 1]})
    negative_samples_df = pd.DataFrame({"(x)": [2.5], "(y)": [1.5], "(z)": [1.0], "(w)": [1.1]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
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

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [])
    assert informative


def test_is_sample_informative_when_using_relevant_features_with_a_single_features_only_examines_only_that_feature_for_the_informative_calculation(
    informative_states_learner_no_predicates: InformationStatesLearner,
):
    positive_samples_df = pd.DataFrame({"(x)": [0, 0, 1, 1], "(y)": [0, 0, 2, 2], "(z)": [0, 0, 0, 0], "(w)": [1, 1, 1, 1]})
    negative_samples_df = pd.DataFrame({"(x)": [2.5], "(y)": [1.5], "(z)": [1.0], "(w)": [1.1]})
    informative_states_learner_no_predicates.numeric_positive_samples = positive_samples_df
    informative_states_learner_no_predicates.numeric_negative_samples = negative_samples_df
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

    informative, _ = informative_states_learner_no_predicates.is_sample_informative(new_sample, [], relevant_numeric_features=["(z)"])
    assert not informative

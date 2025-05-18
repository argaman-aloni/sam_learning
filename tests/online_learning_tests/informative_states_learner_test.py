"""Module tests for the numeric information gaining process."""
import random
from typing import List, Set, Dict

import pandas as pd
import pytest
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Predicate, Domain, Observation

from sam_learning.core import VocabularyCreator
from sam_learning.core.online_learning import IncrementalConvexHullLearner, InformationStatesLearner, OnlineDiscreteModelLearner
from sam_learning.core.online_learning.informative_states_learner import LABEL_COLUMN

TEST_ACTION_NAME = "drive"
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
    return IncrementalConvexHullLearner(TEST_ACTION_NAME, domain_functions=parameter_bound_function_vocabulary, polynom_degree=0)


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
        new_numeric_sample[func.untyped_representation] = new_func

    combined_df = informative_states_learner_no_predicates._create_combined_sample_data(new_numeric_sample, set())
    assert len(combined_df) == 1


def test_create_combined_sample_when_domain_does_not_have_functions_and_only_predicates_data_creates_a_dataframe_with_the_correct_columns(
    informative_states_learner_only_discrete: InformationStatesLearner, lifted_depot_vocabulary: Set[Predicate]
):
    combined_df = informative_states_learner_only_discrete._create_combined_sample_data({}, lifted_depot_vocabulary)
    assert len(combined_df) == 1


def test_create_combined_sample_when_domain_contains_numeric_and_discrete_parts_creates_a_dataframe_with_the_correct_columns(
    depot_informative_states_learner: InformationStatesLearner,
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    lifted_depot_vocabulary: Set[Predicate],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    combined_df = depot_informative_states_learner._create_combined_sample_data(new_numeric_sample, new_discrete_sample)
    assert len(combined_df) == 1


def test_add_new_sample_adds_new_sample_to_the_existing_dataframe_when_no_predicates_are_in_the_domain(
    informative_states_learner_no_predicates: InformationStatesLearner, parameter_bound_function_vocabulary: Dict[str, PDDLFunction]
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample, is_successful=True)
    assert len(informative_states_learner_no_predicates.combined_data) == 1
    assert len(informative_states_learner_no_predicates.numeric_data) == 1
    assert len(informative_states_learner_no_predicates.numeric_data.columns) == 4


def test_add_new_sample_when_sample_is_successful_does_not_add_to_negative_sample(
    informative_states_learner_no_predicates: InformationStatesLearner, parameter_bound_function_vocabulary: Dict[str, PDDLFunction]
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample, is_successful=True)
    assert (informative_states_learner_no_predicates.combined_data[LABEL_COLUMN] == True).all()
    assert (informative_states_learner_no_predicates.numeric_data[LABEL_COLUMN] == True).all()


def test_add_new_sample_when_domain_contains_numeric_and_discrete_parts_adds_discrete_data_to_combined_data(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    depot_informative_states_learner.add_new_sample(new_numeric_sample, new_discrete_sample, is_successful=True)
    assert len(depot_informative_states_learner.combined_data) == 1
    assert {predicate.untyped_representation for predicate in lifted_depot_vocabulary}.issubset(
        depot_informative_states_learner.combined_data.columns.tolist()
    )


def test_visited_previously_failed_execution_when_predicates_match_negative_samples_but_numeric_values_not_match_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    not_observed_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(18 + index)
        new_numeric_sample[function_name] = new_func

    depot_informative_states_learner.add_new_sample(new_numeric_sample, new_discrete_sample, is_successful=False)
    assert not depot_informative_states_learner._visited_previously_failed_execution(
        new_numeric_sample=not_observed_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_visited_previously_failed_execution_when_numeric_functions_match_negative_samples_but_predicates_do_not_match_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    not_observed_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 == 0:
            not_observed_discrete_sample.add(predicate.copy())

    depot_informative_states_learner.add_new_sample(new_numeric_sample, new_discrete_sample, is_successful=False)
    assert not depot_informative_states_learner._visited_previously_failed_execution(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=not_observed_discrete_sample
    )


def test_visited_previously_failed_execution_when_both_predicates_and_numeric_functions_match_returns_true(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    new_discrete_sample = set()
    for index, predicate in enumerate(lifted_depot_vocabulary):
        if index % 2 != 0:
            new_discrete_sample.add(predicate.copy())

    depot_informative_states_learner.add_new_sample(new_numeric_sample, new_discrete_sample, is_successful=False)
    assert depot_informative_states_learner._visited_previously_failed_execution(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=new_discrete_sample
    )


def test_is_state_not_applicable_in_numeric_model_when_no_negative_samples_exist_in_dataset_returns_false(
    informative_states_learner_no_predicates: InformationStatesLearner,
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    not_observed_numeric_sample = {}
    for index, function_name in enumerate(TEST_FUNCTION_NAMES):
        new_func = PDDLFunction(name=function_name, signature={})
        new_func.set_value(18 + index)
        new_numeric_sample[function_name] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=new_numeric_sample, is_successful=True)
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    assert not informative_states_learner_no_predicates._is_state_not_applicable_in_numeric_model(not_observed_numeric_sample)


def test_is_state_not_applicable_in_numeric_model_when_single_negative_sample_exists_and_new_sample_does_not_match_observed_one_returns_false(
    informative_states_learner_no_predicates: InformationStatesLearner, parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    not_observed_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(18 + index)
        not_observed_numeric_sample[func.untyped_representation] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=new_numeric_sample, is_successful=False)
    assert not informative_states_learner_no_predicates._is_state_not_applicable_in_numeric_model(not_observed_numeric_sample)


def test_is_state_not_applicable_in_numeric_model_when_negative_sample_exist_and_new_observation_with_existing_convex_hull_includes_negative_sample_returns_true(
    informative_states_learner_no_predicates: InformationStatesLearner,
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
):
    function_to_positive_values_map = {
        "(load_limit ?x)": [0, 0, 1, 1],
        "(current_load ?x)": [0, 1, 0, 1],
        "(fuel-cost )": [0] * 4,
    }
    function_to_negative_values_map = {
        "(load_limit ?x)": 0.5,
        "(current_load ?x)": 1.5,
        "(fuel-cost )": 0,
    }

    for index in range(4):
        state_dict = {}
        for func_name in function_to_positive_values_map:
            new_func = PDDLFunction(
                name=parameter_bound_function_vocabulary[func_name].name, signature=parameter_bound_function_vocabulary[func_name].signature
            )
            new_func.set_value(function_to_positive_values_map[func_name][index])
            state_dict[func_name] = new_func

        incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in state_dict.values()})
        informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=state_dict, is_successful=True)

    negative_state_dict = {}
    for func_name, value in function_to_negative_values_map.items():
        new_func = PDDLFunction(name=func_name, signature=parameter_bound_function_vocabulary[func_name].signature)
        new_func.set_value(value)
        negative_state_dict[func_name] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=negative_state_dict, is_successful=False)

    new_invalid_sample = {}
    new_func = PDDLFunction(name="load_limit", signature=parameter_bound_function_vocabulary["(load_limit ?x)"].signature)
    new_func.set_value(0.5)
    new_invalid_sample[new_func.untyped_representation] = new_func
    new_func = PDDLFunction(name="current_load", signature=parameter_bound_function_vocabulary["(current_load ?x)"].signature)
    new_func.set_value(2.0)
    new_invalid_sample[new_func.untyped_representation] = new_func
    new_func = PDDLFunction(name="fuel-cost", signature=parameter_bound_function_vocabulary["(fuel-cost )"].signature)
    new_func.set_value(0.0)
    new_invalid_sample[new_func.untyped_representation] = new_func

    assert informative_states_learner_no_predicates._is_state_not_applicable_in_numeric_model(new_numeric_sample=new_invalid_sample)


def test_is_state_not_applicable_in_numeric_model_when_negative_sample_exist_and_new_observation_with_existing_convex_hull_does_not_include_negative_sample_returns_false(
    informative_states_learner_no_predicates: InformationStatesLearner,
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
):
    function_to_positive_values_map = {
        "(load_limit ?x)": [0, 0, 1, 1],
        "(current_load ?x)": [0, 1, 0, 1],
        "(fuel-cost )": [0] * 4,
    }
    function_to_negative_values_map = {
        "(load_limit ?x)": 0.5,
        "(current_load ?x)": 1.5,
        "(fuel-cost )": 0,
    }

    for index in range(4):
        state_dict = {}
        for func_name in function_to_positive_values_map:
            new_func = PDDLFunction(
                name=parameter_bound_function_vocabulary[func_name].name, signature=parameter_bound_function_vocabulary[func_name].signature
            )
            new_func.set_value(function_to_positive_values_map[func_name][index])
            state_dict[func_name] = new_func

        incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in state_dict.values()})
        informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=state_dict, is_successful=True)

    negative_state_dict = {}
    for func_name, value in function_to_negative_values_map.items():
        new_func = PDDLFunction(name=func_name, signature=parameter_bound_function_vocabulary[func_name].signature)
        new_func.set_value(value)
        negative_state_dict[func_name] = new_func

    informative_states_learner_no_predicates.add_new_sample(new_numeric_sample=negative_state_dict, is_successful=False)

    new_invalid_sample = {}
    new_func = PDDLFunction(name="load_limit", signature=parameter_bound_function_vocabulary["(load_limit ?x)"].signature)
    new_func.set_value(0.5)
    new_invalid_sample[new_func.untyped_representation] = new_func
    new_func = PDDLFunction(name="current_load", signature=parameter_bound_function_vocabulary["(current_load ?x)"].signature)
    new_func.set_value(1.25)
    new_invalid_sample[new_func.untyped_representation] = new_func
    new_func = PDDLFunction(name="fuel-cost", signature=parameter_bound_function_vocabulary["(fuel-cost )"].signature)
    new_func.set_value(0.0)
    new_invalid_sample[new_func.untyped_representation] = new_func

    assert not informative_states_learner_no_predicates._is_state_not_applicable_in_numeric_model(new_numeric_sample=new_invalid_sample)


def test_is_sample_informative_when_no_observation_were_observed_yet_returns_true(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    assert (
        depot_informative_states_learner.is_sample_informative(
            new_numeric_sample=new_numeric_sample, new_propositional_sample=lifted_depot_vocabulary
        )
        is True
    )


def test_is_informative_when_new_observation_already_observed_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
    online_discrete_model_learner: OnlineDiscreteModelLearner,
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    post_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 == 0}

    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=pre_state_predicates, post_state_predicates=post_state_predicates, is_transition_successful=True
    )
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=pre_state_predicates, is_successful=True
    )
    assert (
        depot_informative_states_learner.is_sample_informative(new_numeric_sample=new_numeric_sample, new_propositional_sample=pre_state_predicates)
        is False
    )


def test_is_informative_when_propositional_part_of_state_is_superset_of_the_observed_data_and_point_inside_the_convex_hull_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
    online_discrete_model_learner: OnlineDiscreteModelLearner,
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    post_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 == 0}

    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=pre_state_predicates, post_state_predicates=post_state_predicates, is_transition_successful=True
    )

    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=pre_state_predicates, is_successful=True
    )
    assert (
        depot_informative_states_learner.is_sample_informative(
            new_numeric_sample=new_numeric_sample, new_propositional_sample=lifted_depot_vocabulary
        )
        is False
    )


def test_is_informative_when_propositional_part_contains_only_what_cannot_be_preconditions_and_numeric_model_is_safe_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
    online_discrete_model_learner: OnlineDiscreteModelLearner,
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    post_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 == 0}

    # adding positive example.
    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=pre_state_predicates, post_state_predicates=post_state_predicates, is_transition_successful=True
    )
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=pre_state_predicates, is_successful=True
    )

    # There are only 2 predicates in the pre-state predicates, so we need to remove one of them to create a negative example.
    negative_pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    negative_pre_state_predicates.pop()

    # adding negative example.
    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=negative_pre_state_predicates, post_state_predicates=negative_pre_state_predicates, is_transition_successful=False
    )
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=negative_pre_state_predicates, is_successful=False
    )

    assert (
        depot_informative_states_learner.is_sample_informative(new_numeric_sample=new_numeric_sample, new_propositional_sample=post_state_predicates)
        is False
    )


def test_is_informative_when_propositional_part_does_not_contain_unit_clause_must_be_preconditions_and_numeric_model_is_safe_returns_false(
    depot_informative_states_learner: InformationStatesLearner,
    lifted_depot_vocabulary: Set[Predicate],
    parameter_bound_function_vocabulary: Dict[str, PDDLFunction],
    incremental_convex_hull_learner: IncrementalConvexHullLearner,
    online_discrete_model_learner: OnlineDiscreteModelLearner,
):
    new_numeric_sample = {}
    for index, func in enumerate(parameter_bound_function_vocabulary.values()):
        new_func = PDDLFunction(name=func.name, signature=func.signature)
        new_func.set_value(4 + index)
        new_numeric_sample[func.untyped_representation] = new_func

    pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    post_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 == 0}

    # adding positive example.
    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=pre_state_predicates, post_state_predicates=post_state_predicates, is_transition_successful=True
    )
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=pre_state_predicates, is_successful=True
    )

    # There are only 2 predicates in the pre-state predicates, so we need to remove one of them to create a negative example.
    negative_pre_state_predicates = {predicate.copy() for index, predicate in enumerate(lifted_depot_vocabulary) if index % 2 != 0}
    negative_pre_state_predicates.pop()

    # adding negative example.
    online_discrete_model_learner.add_transition_data(
        pre_state_predicates=negative_pre_state_predicates, post_state_predicates=negative_pre_state_predicates, is_transition_successful=False
    )
    incremental_convex_hull_learner.add_new_point(point={func.untyped_representation: func.value for func in new_numeric_sample.values()})
    depot_informative_states_learner.add_new_sample(
        new_numeric_sample=new_numeric_sample, new_propositional_sample=negative_pre_state_predicates, is_successful=False
    )

    assert (
        depot_informative_states_learner.is_sample_informative(
            new_numeric_sample=new_numeric_sample, new_propositional_sample=post_state_predicates.union(negative_pre_state_predicates)
        )
        is False
    )

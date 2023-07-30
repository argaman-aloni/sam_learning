"""Module test for the online_nsam module."""
from pddl_plus_parser.models import Domain, Observation
from pytest import fixture, fail

from sam_learning.learners import OnlineNSAMLearner


@fixture()
def depot_online_nsam(depot_domain: Domain) -> OnlineNSAMLearner:
    return OnlineNSAMLearner(partial_domain=depot_domain)


def test_init_online_learning_creates_an_information_gain_learner_for_each_action(
        depot_online_nsam: OnlineNSAMLearner, depot_domain: Domain):
    depot_online_nsam.init_online_learning()
    assert len(depot_online_nsam.ig_learner) == len(depot_domain.actions)


def test_calculate_state_information_gain_returns_value_greater_than_zero_when_action_is_observed_for_the_first_time(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    assert depot_online_nsam.calculate_state_information_gain(state=tested_state, action=tested_action) > 0


def test_execute_action_when_action_is_successful_adds_action_to_positive_samples_in_information_gain_learner(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)
    assert len(depot_online_nsam.ig_learner[tested_action.name].positive_samples_df) == 1


def test_execute_action_when_action_is_successful_removes_redundant_functions_from_the_possible_functions_in_the_dataframes(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    previous_num_lifted_functions = len(depot_online_nsam.ig_learner[tested_action.name].lifted_functions)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)

    assert len(depot_online_nsam.ig_learner[tested_action.name].lifted_functions) <= previous_num_lifted_functions


def test_execute_action_when_action_is_successful_adds_the_action_to_observed_actions_and_learn_partial_model_of_the_action(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)

    assert tested_action.name in depot_online_nsam.observed_actions
    # checking that the drive action was correctly learned
    assert len(depot_online_nsam.partial_domain.actions[tested_action.name].preconditions.root.operands) > 1
    assert len(depot_online_nsam.partial_domain.actions[tested_action.name].discrete_effects) == 2


def test_execute_action_when_action_is_not_successful_adds_action_to_negative_samples_and_removes_redundant_functions_as_well(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    previous_num_lifted_functions = len(depot_online_nsam.ig_learner[tested_action.name].lifted_functions)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_next_state, next_state=tested_next_state)

    assert len(depot_online_nsam.ig_learner[tested_action.name].negative_samples_df) == 1
    assert len(depot_online_nsam.ig_learner[tested_action.name].lifted_functions) <= previous_num_lifted_functions


def test_execute_action_when_action_is_not_successful_does_not_add_action_to_observed_actions(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_next_state, next_state=tested_next_state)

    assert tested_action.name not in depot_online_nsam.observed_actions


def test_calculate_state_information_gain_when_action_is_observed_twice_returns_zero_in_the_second_calculation(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state

    assert depot_online_nsam.calculate_state_information_gain(state=tested_previous_state, action=tested_action) > 0
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)

    assert depot_online_nsam.calculate_state_information_gain(state=tested_previous_state, action=tested_action) == 0


def test_consecutive_execution_of_informative_actions_creates_small_convex_hulls_and_does_not_fail(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    try:
        for component in depot_observation.components:
            tested_previous_state = component.previous_state
            tested_action = component.grounded_action_call
            tested_next_state = component.next_state
            if depot_online_nsam.calculate_state_information_gain(state=tested_previous_state,
                                                                  action=tested_action) > 0:
                depot_online_nsam.execute_action(
                    action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)

    except Exception as e:
        fail()


def test_consecutive_execution_of_informative_actions_creates_a_usable_model(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    for component in depot_observation.components:
        tested_previous_state = component.previous_state
        tested_action = component.grounded_action_call
        tested_next_state = component.next_state
        if depot_online_nsam.calculate_state_information_gain(state=tested_previous_state,
                                                              action=tested_action) > 0:
            depot_online_nsam.execute_action(
                action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)
    domain = depot_online_nsam.create_safe_model()
    print(domain.to_pddl())

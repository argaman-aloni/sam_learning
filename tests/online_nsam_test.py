"""Module test for the online_nsam module."""
from pddl_plus_parser.models import Domain, Observation, ActionCall
from pytest import fixture, fail

from sam_learning.core import PriorityQueue
from sam_learning.learners import OnlineNSAMLearner


@fixture()
def depot_online_nsam(depot_domain: Domain) -> OnlineNSAMLearner:
    return OnlineNSAMLearner(partial_domain=depot_domain)


def test_init_online_learning_creates_an_information_gain_learner_for_each_action(
        depot_online_nsam: OnlineNSAMLearner, depot_domain: Domain):
    depot_online_nsam.init_online_learning()
    assert len(depot_online_nsam.ig_learner) == len(depot_domain.actions)


def test_select_next_action_to_execute_when_failure_rate_is_large_selects_from_executable_actions(
        depot_online_nsam: OnlineNSAMLearner, depot_domain: Domain, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    depot_online_nsam._state_failure_rate = 1000
    depot_online_nsam._state_applicable_actions = PriorityQueue()
    applicable_action = depot_observation.components[0].grounded_action_call
    depot_online_nsam._state_applicable_actions.insert(item=applicable_action, priority=1.0, selection_probability=1.0)
    frontier = PriorityQueue()
    not_applicable_action = depot_observation.components[1].grounded_action_call
    frontier.insert(item=not_applicable_action, priority=1.0, selection_probability=1.0)
    next_action = depot_online_nsam._select_next_action_to_execute(frontier_actions=frontier)
    assert next_action == applicable_action


def test_select_next_action_to_execute_when_failure_rate_is_zero_will_select_from_the_frontier(
        depot_online_nsam: OnlineNSAMLearner, depot_domain: Domain, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    depot_online_nsam._state_failure_rate = 0
    depot_online_nsam._state_applicable_actions = PriorityQueue()
    applicable_action = depot_observation.components[0].grounded_action_call
    depot_online_nsam._state_applicable_actions.insert(item=applicable_action, priority=1.0, selection_probability=1.0)
    frontier = PriorityQueue()
    not_applicable_action = depot_observation.components[1].grounded_action_call
    frontier.insert(item=not_applicable_action, priority=1.0, selection_probability=1.0)
    next_action = depot_online_nsam._select_next_action_to_execute(frontier_actions=frontier)
    assert next_action == not_applicable_action


def test_calculate_state_information_gain_returns_value_greater_than_zero_when_action_is_observed_for_the_first_time(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    tested_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_state, action=tested_action) > 0


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

    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) > 0
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)

    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) == 0


def test_consecutive_execution_of_informative_actions_creates_small_convex_hulls_and_does_not_fail(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    depot_online_nsam.init_online_learning()
    try:
        for component in depot_observation.components:
            tested_previous_state = component.previous_state
            tested_action = component.grounded_action_call
            tested_next_state = component.next_state
            if depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
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
        if depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) > 0:
            depot_online_nsam.execute_action(
                action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state)
    domain = depot_online_nsam.create_safe_model()
    print(domain.to_pddl())


def test_create_all_grounded_actions_create_all_possible_grounded_action_assignments_for_all_actions(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation, depot_domain: Domain):
    grounded_actions = depot_online_nsam.create_all_grounded_actions(
        observed_objects=depot_observation.grounded_objects)
    assert len({action.name for action in grounded_actions}) == len(depot_domain.actions)


def test_calculate_valid_neighbors_returns_a_set_of_actions_containing_the_action_that_was_actually_executed_on_the_state(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    grounded_actions = depot_online_nsam.create_all_grounded_actions(
        observed_objects=depot_observation.grounded_objects)
    initial_state = depot_observation.components[0].previous_state
    observation_action = depot_observation.components[0].grounded_action_call

    depot_online_nsam.init_online_learning()
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    queue_items = set()
    while len(valid_neighbors) > 0:
        action = valid_neighbors.get_item()
        assert isinstance(action, ActionCall)
        queue_items.add(str(action))

    assert str(observation_action) in queue_items


def test_calculate_valid_neighbors_returns_a_set_with_less_actions_when_action_already_executed_in_state_and_the_action_is_not_applicable(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    grounded_actions = depot_online_nsam.create_all_grounded_actions(
        observed_objects=depot_observation.grounded_objects)
    initial_state = depot_observation.components[0].previous_state
    observation_action = depot_observation.components[0].grounded_action_call

    depot_online_nsam.init_online_learning()
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    num_neighbors = 0
    while len(valid_neighbors) > 0:
        num_neighbors += 1
        valid_neighbors.get_item()

    depot_online_nsam.execute_action(observation_action, previous_state=initial_state, next_state=initial_state)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    new_num_neighbors = 0
    while len(valid_neighbors) > 0:
        new_num_neighbors += 1
        valid_neighbors.get_item()

    assert new_num_neighbors == num_neighbors - 2


def test_calculate_valid_neighbors_returns_a_set_with_less_actions_when_action_already_executed_in_state_and_the_action_is_applicable(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    grounded_actions = depot_online_nsam.create_all_grounded_actions(
        observed_objects=depot_observation.grounded_objects)
    initial_state = depot_observation.components[0].previous_state
    observation_action = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state

    depot_online_nsam.init_online_learning()
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    num_neighbors = 0
    while len(valid_neighbors) > 0:
        num_neighbors += 1
        valid_neighbors.get_item()

    # executed action '(drive truck0 depot0 distributor0)'
    depot_online_nsam.execute_action(observation_action, previous_state=initial_state, next_state=next_state)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    new_num_neighbors = 0
    while len(valid_neighbors) > 0:
        new_num_neighbors += 1
        valid_neighbors.get_item()

    non_informative_actions = ['(drive truck0 depot0 distributor0)', '(drive truck0 depot0 distributor1)',
                               '(drive truck0 distributor0 depot0)', '(drive truck0 distributor1 depot0)',
                               '(drive truck1 distributor0 depot0)', '(drive truck1 distributor1 depot0)']
    assert new_num_neighbors == num_neighbors - len(non_informative_actions)

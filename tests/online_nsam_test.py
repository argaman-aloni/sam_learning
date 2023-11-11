"""Module test for the online_nsam module."""
from pddl_plus_parser.models import Domain, Observation, ActionCall, ObservedComponent, PDDLObject, State
from pytest import fixture, fail
from typing import Dict

from sam_learning.core import PriorityQueue, InformationGainLearner
from sam_learning.learners import OnlineNSAMLearner
from tests.consts import sync_snapshot


@fixture()
def depot_online_nsam(depot_domain: Domain) -> OnlineNSAMLearner:
    return OnlineNSAMLearner(partial_domain=depot_domain)


@fixture()
def minecraft_large_map_online_nsam(minecraft_large_domain: Domain) -> OnlineNSAMLearner:
    return OnlineNSAMLearner(partial_domain=minecraft_large_domain)


def init_information_gain_dataframes(
        online_nsam: OnlineNSAMLearner, state: State, observed_objects: Dict[str, PDDLObject],
        action: ActionCall) -> None:
    """Initializes the information gain data frames for the given action.

    :param online_nsam: the online NSAM learner.
    :param state: the state to initialize the data frames for.
    :param observed_objects: the observed objects in the state.
    :param action: the action to initialize the data frames for.
    """
    grounded_state_propositions = online_nsam.triplet_snapshot.create_propositional_state_snapshot(
        state, action, observed_objects)
    lifted_predicates = online_nsam.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
    grounded_state_functions = online_nsam.triplet_snapshot.create_numeric_state_snapshot(state, action,
                                                                                          observed_objects)
    lifted_functions = online_nsam.function_matcher.match_state_functions(action, grounded_state_functions)
    online_nsam.ig_learner[action.name].init_dataframes(
        valid_lifted_functions=list([func for func in lifted_functions.keys()]),
        lifted_predicates=[pred.untyped_representation for pred in lifted_predicates])


def test_extract_objects_from_state_extract_all_objects_from_state(
        minecraft_large_map_online_nsam: OnlineNSAMLearner, minecraft_large_trajectory: Observation):
    objects = minecraft_large_map_online_nsam._extract_objects_from_state(
        state=minecraft_large_trajectory.components[0].previous_state)

    valid_regular_cells = [f"cell{i}" for i in range(36) if i != 16]
    assert "crafting_table" in objects
    assert all([cell in objects for cell in valid_regular_cells])
    assert len(objects) == 36


def test_add_action_execution_to_db_adds_correctly_the_execution_data_of_a_successful_action_in_a_state(
        minecraft_large_map_online_nsam: OnlineNSAMLearner, minecraft_large_trajectory: Observation):
    action = minecraft_large_trajectory.components[0].grounded_action_call
    prev_state = minecraft_large_trajectory.components[0].previous_state
    lifted_functions, lifted_predicates = minecraft_large_map_online_nsam._get_lifted_bounded_state(action, prev_state)
    assert all([len(val) for val in minecraft_large_map_online_nsam._state_action_execution_db.values()]) == 0
    minecraft_large_map_online_nsam._add_action_execution_to_db(
        str(minecraft_large_map_online_nsam.partial_domain.actions["tp_to"]), lifted_predicates, lifted_functions,
        execution_result=1)
    assert all([len(val) for val in minecraft_large_map_online_nsam._state_action_execution_db.values()]) == 1


def test_select_next_action_to_execute_when_failure_rate_is_large_selects_from_executable_actions(
        depot_online_nsam: OnlineNSAMLearner, depot_domain: Domain, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    depot_online_nsam._action_failure_rate = 1000
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
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    depot_online_nsam._action_failure_rate = 0
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
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)

    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    tested_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_state, action=tested_action) > 0


def test_execute_action_when_action_is_successful_adds_action_to_positive_samples_in_information_gain_learner(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state, reward=1)
    assert len(depot_online_nsam.ig_learner[tested_action.name].numeric_positive_samples) == 1
    assert len(depot_online_nsam.ig_learner[tested_action.name].positive_discrete_sample_df) == 1


def test_execute_action_when_action_is_successful_adds_the_action_to_observed_actions_and_learn_partial_model_of_the_action(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state, reward=1)

    assert tested_action.name in depot_online_nsam.observed_actions
    # checking that the drive action was correctly learned
    assert len(depot_online_nsam.partial_domain.actions[tested_action.name].preconditions.root.operands) > 1
    assert len(depot_online_nsam.partial_domain.actions[tested_action.name].discrete_effects) == 2


def test_execute_action_when_action_is_not_successful_adds_action_to_negative_samples_and_removes_redundant_functions_as_well(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_next_state, next_state=tested_next_state, reward=-1)

    assert len(depot_online_nsam.ig_learner[tested_action.name].numeric_negative_samples) == 1
    assert len(depot_online_nsam.ig_learner[tested_action.name].negative_combined_sample_df) == 1


def test_execute_action_when_action_is_not_successful_does_not_add_action_to_observed_actions(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_next_state, next_state=tested_next_state, reward=-1)

    assert tested_action.name not in depot_online_nsam.observed_actions


def test_calculate_state_information_gain_when_action_is_observed_twice_returns_zero_in_the_second_calculation(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    tested_previous_state = depot_observation.components[0].previous_state
    tested_action = depot_observation.components[0].grounded_action_call
    tested_next_state = depot_observation.components[0].next_state
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) > 0
    depot_online_nsam.execute_action(
        action_to_execute=tested_action, previous_state=tested_previous_state, next_state=tested_next_state, reward=1)

    assert depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) == 0


def test_consecutive_execution_of_informative_actions_creates_small_convex_hulls_and_does_not_fail(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    try:
        for component in depot_observation.components:
            tested_previous_state = component.previous_state
            tested_action = component.grounded_action_call
            tested_next_state = component.next_state
            if depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                         action=tested_action) > 0:
                depot_online_nsam.execute_action(
                    action_to_execute=tested_action,
                    previous_state=tested_previous_state, next_state=tested_next_state, reward=1)

    except Exception as e:
        fail()


def test_consecutive_execution_of_informative_actions_creates_a_usable_model(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    depot_online_nsam.create_all_grounded_actions(observed_objects=depot_observation.grounded_objects)
    for component in depot_observation.components:
        tested_previous_state = component.previous_state
        tested_action = component.grounded_action_call
        tested_next_state = component.next_state
        if depot_online_nsam.calculate_state_action_information_gain(state=tested_previous_state,
                                                                     action=tested_action) > 0:
            depot_online_nsam.execute_action(
                action_to_execute=tested_action, previous_state=tested_previous_state,
                next_state=tested_next_state, reward=1)
    depot_online_nsam._create_safe_action_model()
    domain = depot_online_nsam.partial_domain
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

    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
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

    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    num_neighbors = valid_neighbors.__len__()

    depot_online_nsam.execute_action(observation_action, previous_state=initial_state, next_state=initial_state,
                                     reward=-1)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    new_num_neighbors = valid_neighbors.__len__()
    assert new_num_neighbors < num_neighbors


def test_calculate_valid_neighbors_returns_a_set_with_less_actions_when_action_already_executed_in_state_and_the_action_is_applicable(
        depot_online_nsam: OnlineNSAMLearner, depot_observation: Observation):
    grounded_actions = depot_online_nsam.create_all_grounded_actions(
        observed_objects=depot_observation.grounded_objects)
    initial_state = depot_observation.components[0].previous_state
    observation_action = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state

    init_information_gain_dataframes(
        depot_online_nsam, depot_observation.components[0].previous_state,
        depot_observation.grounded_objects, depot_observation.components[0].grounded_action_call)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    num_neighbors = valid_neighbors.__len__()

    # executed action '(drive truck0 depot0 distributor0)'
    depot_online_nsam.execute_action(observation_action, previous_state=initial_state, next_state=next_state, reward=1)
    valid_neighbors = depot_online_nsam.calculate_valid_neighbors(grounded_actions=grounded_actions,
                                                                  current_state=initial_state)
    new_num_neighbors = valid_neighbors.__len__()
    assert new_num_neighbors < num_neighbors

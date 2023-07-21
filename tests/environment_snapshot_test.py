from pddl_plus_parser.models import Domain, Observation
from pytest import fixture

from sam_learning.core import EnvironmentSnapshot


@fixture()
def elevators_environment_snapshot(elevators_domain: Domain):
    return EnvironmentSnapshot(partial_domain=elevators_domain)


def test_create_state_discrete_snapshot_creates_a_snapshot_with_negative_predicates_set_that_is_not_empty(
        elevators_environment_snapshot: EnvironmentSnapshot, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    predicates = elevators_environment_snapshot._create_state_discrete_snapshot(
        state=initial_state, relevant_objects=observed_objects)
    negative_predicates = [p for p in predicates if not p.is_positive]
    assert len(negative_predicates) > 0


def test_create_state_discrete_snapshot_creates_a_snapshot_with_negative_and_positive_predicates(
        elevators_environment_snapshot: EnvironmentSnapshot, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    predicates = elevators_environment_snapshot._create_state_discrete_snapshot(
        state=initial_state, relevant_objects=observed_objects)
    positive_predicates_str = set([p.untyped_representation for p in predicates if p.is_positive])
    negative_predicates_str = set([p.untyped_representation for p in predicates if not p.is_positive])
    assert positive_predicates_str.intersection(negative_predicates_str) == set()


def test_create_state_discrete_snapshot_creates_a_snapshot_with_positive_predicates_covering_all_state_predicates(
        elevators_environment_snapshot: EnvironmentSnapshot, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    predicates = elevators_environment_snapshot._create_state_discrete_snapshot(
        state=initial_state, relevant_objects=observed_objects)
    positive_predicates = [p for p in predicates if p.is_positive]
    state_predicates = set()
    for predicates in initial_state.state_predicates.values():
        state_predicates.update(predicates)

    positive_predicates_str = set([p.untyped_representation for p in positive_predicates])
    state_predicates_str = set([p.untyped_representation for p in state_predicates])
    assert len(positive_predicates_str.intersection(state_predicates_str)) == len(state_predicates_str)
    assert len(positive_predicates_str.intersection(state_predicates_str)) == len(positive_predicates_str)


def test_create_snapshot_when_should_not_include_all_objects_creates_lower_number_of_predicates_than_when_with_all_objects(
        elevators_environment_snapshot: EnvironmentSnapshot, elevators_observation: Observation, elevators_domain: Domain):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    current_action = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    elevators_environment_snapshot.create_triplet_snapshot(
        previous_state=previous_state, next_state=next_state, current_action=current_action,
        observation_objects=observed_objects)
    num_predicates_without_all_objects = len(elevators_environment_snapshot.previous_state_predicates)

    all_types = [pddl_type for pddl_type in elevators_domain.types if pddl_type != "object"]
    elevators_environment_snapshot.create_triplet_snapshot(
        previous_state=previous_state, next_state=next_state, current_action=current_action,
        observation_objects=observed_objects, specific_types=all_types)
    num_predicates_with_all_objects = len(elevators_environment_snapshot.previous_state_predicates)

    assert num_predicates_without_all_objects < num_predicates_with_all_objects

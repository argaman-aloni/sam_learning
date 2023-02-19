"""module tests for the SAM learning algorithm"""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, ActionCall, Problem, Observation, \
    ObservedComponent
from pytest import fixture

from sam_learning.learners import SAMLearner
from tests.consts import ELEVATORS_DOMAIN_PATH, ELEVATORS_PROBLEM_PATH, ELEVATORS_TRAJECTORY_PATH


@fixture()
def elevators_domain() -> Domain:
    domain_parser = DomainParser(ELEVATORS_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def elevators_problem(elevators_domain: Domain) -> Problem:
    return ProblemParser(problem_path=ELEVATORS_PROBLEM_PATH, domain=elevators_domain).parse_problem()


@fixture()
def elevators_observation(elevators_domain: Domain, elevators_problem: Problem) -> Observation:
    return TrajectoryParser(elevators_domain, elevators_problem).parse_trajectory(ELEVATORS_TRAJECTORY_PATH)


@fixture()
def sam_learning(elevators_domain: Domain) -> SAMLearner:
    return SAMLearner(elevators_domain)


def test_create_complete_world_state_creates_the_representation_of_the_world_with_negative_predicates_that_are_not_empty(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    _, negative_predicates = sam_learning._create_complete_world_state(
        relevant_objects=observed_objects,
        state=initial_state)
    assert len(negative_predicates) > 0


def test_create_complete_world_state_creates_the_representation_of_the_world_with_negative_and_positive_predicates(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    positive_predicates, negative_predicates = sam_learning._create_complete_world_state(
        relevant_objects=observed_objects,
        state=initial_state)
    positive_predicates_str = set([p.untyped_representation for p in positive_predicates])
    negative_predicates_str = set([p.untyped_representation for p in negative_predicates])
    assert positive_predicates_str.intersection(negative_predicates_str) == set()


def test_create_complete_world_state_creates_the_representation_of_the_world_with_positive_predicates_covering_all_state_predicates(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    initial_state = observation_component.previous_state
    observed_objects = elevators_observation.grounded_objects
    positive_predicates, _ = sam_learning._create_complete_world_state(
        relevant_objects=observed_objects,
        state=initial_state)
    state_predicates = set()
    for predicates in initial_state.state_predicates.values():
        state_predicates.update(predicates)

    positive_predicates_str = set([p.untyped_representation for p in positive_predicates])
    state_predicates_str = set([p.untyped_representation for p in state_predicates])
    assert len(positive_predicates_str.intersection(state_predicates_str)) == len(state_predicates_str)
    assert len(positive_predicates_str.intersection(state_predicates_str)) == len(positive_predicates_str)


def test_add_new_action_with_single_trajectory_component_adds_action_data_to_learned_domain(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    sam_learning._create_fully_observable_triplet_predicates(
        current_action=test_action_call,
        previous_state=previous_state,
        next_state=next_state)
    sam_learning.add_new_action(grounded_action=test_action_call,
                                previous_state=previous_state,
                                next_state=next_state)

    added_action_name = "move-down-slow"
    assert added_action_name in sam_learning.partial_domain.actions
    learned_action_data = sam_learning.partial_domain.actions[added_action_name]
    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_deduce_initial_inequality_preconditions_deduce_that_all_objects_with_same_type_should_not_be_equal(
        sam_learning: SAMLearner):
    sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = sam_learning.partial_domain.actions[example_action_name]
    assert action.inequality_preconditions == {("?f1", "?f2")}


def test_verify_parameter_duplication_removes_inequality_if_found_action_with_duplicated_items_in_observation(
        sam_learning: SAMLearner):
    sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = sam_learning.partial_domain.actions[example_action_name]
    assert action.inequality_preconditions == {("?f1", "?f2")}
    duplicated_action_call = ActionCall(name=example_action_name, grounded_parameters=["slow-lift", "c1", "c1"])
    sam_learning._verify_parameter_duplication(duplicated_action_call)
    assert len(action.inequality_preconditions) == 0


def test_handle_action_effects_returns_delete_effects_with_predicates_with_is_positive_false_in_the_delete_effects_and_is_positive_true_in_the_add_effects(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    add_effects, delete_effects = sam_learning._handle_action_effects(test_action_call, previous_state, next_state)
    assert len(add_effects) > 0
    assert len(delete_effects) > 0
    for add_effects_predicate in add_effects:
        assert add_effects_predicate.is_positive

    for delete_effects_predicate in delete_effects:
        assert not delete_effects_predicate.is_positive


def test_handle_action_effects_does_not_create_intersecting_sets_of_effects(sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    add_effects, delete_effects = sam_learning._handle_action_effects(test_action_call, previous_state, next_state)
    add_effects_str = set([p.untyped_representation for p in add_effects])
    delete_effects_str = set([p.untyped_representation for p in delete_effects])
    assert not add_effects_str.intersection(delete_effects_str)

def test_add_new_action_preconditions_adds_both_negative_and_positive_preconditions_to_the_action(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    sam_learning._create_fully_observable_triplet_predicates(
        current_action=test_action_call,
        previous_state=previous_state,
        next_state=next_state)
    sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = sam_learning.partial_domain.actions[test_action_call.name]
    assert len(learned_action_data.positive_preconditions) > 0
    assert len(learned_action_data.negative_preconditions) > 0
    print(learned_action_data.to_pddl())


def test_add_new_action_preconditions_adds_correct_positive_preconditions_to_action(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    sam_learning._create_fully_observable_triplet_predicates(
        current_action=test_action_call,
        previous_state=previous_state,
        next_state=next_state)
    sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = sam_learning.partial_domain.actions[test_action_call.name]
    positive_conditions = {p.untyped_representation for p in learned_action_data.positive_preconditions}
    expected_conditions = {"(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"}
    assert expected_conditions.issubset(positive_conditions)


def test_add_new_action_preconditions_do_not_adds_intersecting_positive_and_negative_preconditions(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    sam_learning.current_trajectory_objects = observed_objects
    sam_learning._create_fully_observable_triplet_predicates(
        current_action=test_action_call,
        previous_state=previous_state,
        next_state=next_state)
    sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = sam_learning.partial_domain.actions[test_action_call.name]
    positive_conditions = {p.untyped_representation for p in learned_action_data.positive_preconditions}
    negative_conditions = {p.untyped_representation for p in learned_action_data.negative_preconditions}
    assert not positive_conditions.intersection(negative_conditions)

def test_update_action_with_two_trajectory_component_updates_action_data_correctly(
        sam_learning: SAMLearner, elevators_observation: Observation):
    first_observation_component = elevators_observation.components[0]
    second_observation_component = elevators_observation.components[4]

    sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    first_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n16"])
    second_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow1-0", "n9", "n8"])
    sam_learning._create_fully_observable_triplet_predicates(
        current_action=first_action_call,
        previous_state=first_observation_component.previous_state,
        next_state=first_observation_component.next_state)

    sam_learning.add_new_action(grounded_action=first_action_call,
                                previous_state=first_observation_component.previous_state,
                                next_state=first_observation_component.next_state)

    print(second_observation_component.previous_state.serialize())

    sam_learning._create_fully_observable_triplet_predicates(
        current_action=second_action_call,
        previous_state=second_observation_component.previous_state,
        next_state=second_observation_component.next_state)

    sam_learning.update_action(grounded_action=second_action_call,
                               previous_state=second_observation_component.previous_state,
                               next_state=second_observation_component.next_state)
    added_action_name = "move-down-slow"

    assert added_action_name in sam_learning.partial_domain.actions
    learned_action_data = sam_learning.partial_domain.actions[added_action_name]
    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_handle_single_trajectory_component_not_allowing_actions_with_duplicated_parameters(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    test_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n17"])
    component = ObservedComponent(observation_component.previous_state, test_action_call,
                                  observation_component.next_state)
    sam_learning.handle_single_trajectory_component(component)

    added_action_name = "move-down-slow"
    learned_action_data = sam_learning.partial_domain.actions[added_action_name]
    assert len(learned_action_data.positive_preconditions) == 0
    assert len(learned_action_data.add_effects) == 0
    assert len(learned_action_data.delete_effects) == 0


def test_handle_single_trajectory_component_learns_preconditions_and_effects_when_given_a_non_duplicate_component(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    sam_learning.handle_single_trajectory_component(observation_component)

    added_action_name = "move-down-slow"
    learned_action_data = sam_learning.partial_domain.actions[added_action_name]

    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_learn_action_model_returns_learned_model(sam_learning: SAMLearner, elevators_observation: Observation):
    learned_model, learning_report = sam_learning.learn_action_model([elevators_observation])
    print(learning_report)
    print(learned_model.to_pddl())

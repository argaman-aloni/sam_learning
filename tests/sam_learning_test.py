"""module tests for the SAM learning algorithm"""

from pddl_plus_parser.models import ActionCall, Observation, \
    ObservedComponent

from sam_learning.learners import SAMLearner
from tests.consts import sync_snapshot


def test_add_new_action_preconditions_adds_both_negative_and_positive_preconditions_to_the_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    assert len(learned_action_data.positive_preconditions) > 0
    assert len(learned_action_data.negative_preconditions) > 0
    print(learned_action_data.to_pddl())


def test_add_new_action_preconditions_adds_correct_positive_preconditions_to_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    positive_conditions = {p.untyped_representation for p in learned_action_data.positive_preconditions}
    expected_conditions = {"(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"}
    assert expected_conditions.issubset(positive_conditions)


def test_add_new_action_preconditions_do_not_adds_intersecting_positive_and_negative_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    positive_conditions = {p.untyped_representation for p in learned_action_data.positive_preconditions}
    negative_conditions = {p.untyped_representation for p in learned_action_data.negative_preconditions}
    assert not positive_conditions.intersection(negative_conditions)


def test_update_action_preconditions_reduces_the_number_of_positive_predicates_but_does_not_remove_actual_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)
    second_component = elevators_observation.components[4]
    sync_snapshot(elevators_sam_learning, second_component, elevators_observation.grounded_objects)
    second_test_action_call = second_component.grounded_action_call
    elevators_sam_learning._update_action_preconditions(grounded_action=second_test_action_call)
    positive_preconditions = elevators_sam_learning.partial_domain.actions["move-down-slow"].positive_preconditions
    preconditions_str = set([p.untyped_representation for p in positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    preconditions_str = set([p.untyped_representation for p in
                             elevators_sam_learning.partial_domain.actions["move-down-slow"].negative_preconditions])
    assert all([p.startswith("(not") for p in preconditions_str])


def test_update_action_preconditions_reduces_the_number_of_negative_predicates_but_does_not_remove_actual_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)

    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)
    second_component = elevators_observation.components[4]
    sync_snapshot(elevators_sam_learning, second_component, elevators_observation.grounded_objects)
    second_test_action_call = second_component.grounded_action_call
    elevators_sam_learning._update_action_preconditions(grounded_action=second_test_action_call)
    negative_preconditions = elevators_sam_learning.partial_domain.actions["move-down-slow"].negative_preconditions
    preconditions_str = set([p.untyped_representation for p in negative_preconditions])
    assert "(not (lift-at ?lift ?f2))" in preconditions_str
    assert "(not (above ?f1 ?f2))" in preconditions_str


def test_add_new_action_with_single_trajectory_component_adds_correct_preconditions_to_learned_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)

    added_action_name = "move-down-slow"
    assert added_action_name in elevators_sam_learning.partial_domain.actions
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]
    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])

    preconditions_str = set([p.untyped_representation for p in learned_action_data.negative_preconditions])
    assert all([p.startswith("(not") for p in preconditions_str])


def test_add_new_action_with_single_trajectory_component_adds_correct_add_and_del_effects_to_the_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)

    added_action_name = "move-down-slow"
    assert added_action_name in elevators_sam_learning.partial_domain.actions
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_add_new_action_maintains_the_correct_add_and_delete_effects(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning.add_new_action(test_action_call, previous_state, next_state)
    learned_action = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    add_effects = {eff.untyped_representation for eff in learned_action.add_effects}
    delete_effects = {eff.untyped_representation for eff in learned_action.delete_effects}
    assert add_effects == {"(lift-at ?lift ?f2)"}
    assert delete_effects == {"(not (lift-at ?lift ?f1))"}


def test_update_action_does_not_remove_effects_from_the_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)
    second_component = elevators_observation.components[4]
    sync_snapshot(elevators_sam_learning, second_component, elevators_observation.grounded_objects)
    second_previous_state = second_component.previous_state
    second_next_state = second_component.next_state
    second_test_action_call = second_component.grounded_action_call
    elevators_sam_learning.update_action(grounded_action=second_test_action_call, previous_state=second_previous_state,
                                         next_state=second_next_state)
    learned_action = elevators_sam_learning.partial_domain.actions[second_test_action_call.name]
    add_effects = {eff.untyped_representation for eff in learned_action.add_effects}
    delete_effects = {eff.untyped_representation for eff in learned_action.delete_effects}
    assert add_effects == {"(lift-at ?lift ?f2)"}
    assert delete_effects == {"(not (lift-at ?lift ?f1))"}


def test_update_action_with_two_trajectory_component_updates_action_data_correctly(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    first_observation_component = elevators_observation.components[0]
    second_observation_component = elevators_observation.components[4]

    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    first_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n16"])
    second_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow1-0", "n9", "n8"])
    sync_snapshot(elevators_sam_learning, first_observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning.add_new_action(grounded_action=first_action_call,
                                          previous_state=first_observation_component.previous_state,
                                          next_state=first_observation_component.next_state)

    sync_snapshot(elevators_sam_learning, second_observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning.update_action(grounded_action=second_action_call,
                                         previous_state=second_observation_component.previous_state,
                                         next_state=second_observation_component.next_state)
    added_action_name = "move-down-slow"

    assert added_action_name in elevators_sam_learning.partial_domain.actions
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]
    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_deduce_initial_inequality_preconditions_deduce_that_all_objects_with_same_type_should_not_be_equal(
        elevators_sam_learning: SAMLearner):
    elevators_sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = elevators_sam_learning.partial_domain.actions[example_action_name]
    assert action.inequality_preconditions == {("?f1", "?f2")}


def test_verify_parameter_duplication_removes_inequality_if_found_action_with_duplicated_items_in_observation(
        elevators_sam_learning: SAMLearner):
    elevators_sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = elevators_sam_learning.partial_domain.actions[example_action_name]
    assert action.inequality_preconditions == {("?f1", "?f2")}
    duplicated_action_call = ActionCall(name=example_action_name, grounded_parameters=["slow-lift", "c1", "c1"])
    elevators_sam_learning._verify_parameter_duplication(duplicated_action_call)
    assert len(action.inequality_preconditions) == 0


def test_handle_action_effects_returns_delete_effects_with_predicates_with_is_positive_false_in_the_delete_effects_and_is_positive_true_in_the_add_effects(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    elevators_sam_learning.current_trajectory_objects = observed_objects
    add_effects, delete_effects = elevators_sam_learning._handle_action_effects(test_action_call, previous_state,
                                                                                next_state)
    assert len(add_effects) > 0
    assert len(delete_effects) > 0
    for add_effects_predicate in add_effects:
        assert add_effects_predicate.is_positive

    for delete_effects_predicate in delete_effects:
        assert not delete_effects_predicate.is_positive


def test_handle_action_effects_returns_correct_add_and_delete_effects_predicates_after_action_is_executed(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    elevators_sam_learning.current_trajectory_objects = observed_objects
    add_effects, delete_effects = elevators_sam_learning._handle_action_effects(test_action_call, previous_state,
                                                                                next_state)
    assert len(add_effects) == len(delete_effects) == 1
    assert add_effects[0].untyped_representation == "(lift-at ?lift ?f2)"
    assert delete_effects[0].untyped_representation == "(not (lift-at ?lift ?f1))"


def test_handle_action_effects_does_not_create_intersecting_sets_of_effects(elevators_sam_learning: SAMLearner,
                                                                            elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    observed_objects = elevators_observation.grounded_objects
    elevators_sam_learning.current_trajectory_objects = observed_objects
    add_effects, delete_effects = elevators_sam_learning._handle_action_effects(test_action_call, previous_state,
                                                                                next_state)
    add_effects_str = set([p.untyped_representation for p in add_effects])
    delete_effects_str = set([p.untyped_representation for p in delete_effects])
    assert not add_effects_str.intersection(delete_effects_str)


def test_handle_single_trajectory_component_not_allowing_actions_with_duplicated_parameters(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    test_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n17"])
    component = ObservedComponent(observation_component.previous_state, test_action_call,
                                  observation_component.next_state)
    elevators_sam_learning.handle_single_trajectory_component(component)

    added_action_name = "move-down-slow"
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]
    assert len(learned_action_data.positive_preconditions) == 0
    assert len(learned_action_data.add_effects) == 0
    assert len(learned_action_data.delete_effects) == 0


def test_handle_single_trajectory_component_learns_preconditions_and_effects_when_given_a_non_duplicate_component(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    added_action_name = "move-down-slow"
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]

    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_handle_single_trajectory_component_maintains_the_correct_preconditions_invariants_after_multiple_executions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    observation_component = elevators_observation.components[4]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    learned_action_data = elevators_sam_learning.partial_domain.actions[observation_component.grounded_action_call.name]
    preconditions_str = set([p.untyped_representation for p in learned_action_data.positive_preconditions])
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])

    preconditions_str = set([p.untyped_representation for p in learned_action_data.negative_preconditions])
    assert preconditions_str.issuperset(["(not (lift-at ?lift ?f2))", "(not (above ?f1 ?f2))"])
    print(learned_action_data.to_pddl())


def test_handle_single_trajectory_component_maintains_the_correct_effect_invariants_after_multiple_executions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    observation_component = elevators_observation.components[4]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    learned_action_data = elevators_sam_learning.partial_domain.actions[observation_component.grounded_action_call.name]
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]


def test_learn_action_model_does_not_obstruct_the_correctness_of_the_actions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    learned_model, _ = elevators_sam_learning.learn_action_model([elevators_observation])

    learned_action_data = learned_model.actions[observation_component.grounded_action_call.name]
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(not (lift-at ?lift ?f1))"]
    print(learned_action_data.to_pddl())


def test_learn_action_model_returns_learned_model(elevators_sam_learning: SAMLearner,
                                                  elevators_observation: Observation):
    learned_model, learning_report = elevators_sam_learning.learn_action_model([elevators_observation])
    print(learning_report)
    print(learned_model.to_pddl())

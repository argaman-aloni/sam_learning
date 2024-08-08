"""module tests for the SAM learning algorithm"""

from pddl_plus_parser.models import ActionCall, Observation, \
    ObservedComponent, Predicate

from sam_learning.learners import SAMLearner
from tests.consts import sync_snapshot, extract_preconditions_predicates


def test_add_new_action_preconditions_adds_both_negative_and_positive_preconditions_to_the_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    positive_preconditions = [precond for precond in discrete_preconditions if precond.is_positive]
    negative_preconditions = [precond for precond in discrete_preconditions if not precond.is_positive]
    assert len(positive_preconditions) > 0
    assert len(negative_preconditions) > 0
    print(learned_action_data.to_pddl())


def test_add_new_action_preconditions_adds_correct_positive_preconditions_to_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    positive_preconditions = [precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive]
    expected_conditions = {"(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"}
    assert expected_conditions.issubset(positive_preconditions)


def test_add_new_action_preconditions_do_not_adds_intersecting_positive_and_negative_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning._add_new_action_preconditions(grounded_action=test_action_call)
    learned_action_data = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    positive_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive}
    negative_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive}
    assert not positive_preconditions.intersection(negative_preconditions)


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
    discrete_preconditions = extract_preconditions_predicates(
        elevators_sam_learning.partial_domain.actions["move-down-slow"].preconditions)

    positive_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive}
    negative_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive}
    assert positive_preconditions.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert all([p.startswith("(not") for p in negative_preconditions])


def test_update_action_preconditions_does_not_add_preconditions_not_observed_previously_when_first_added_the_action(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    test_partial_previous_state = previous_state.copy()
    test_partial_previous_state.state_predicates["(reachable-floor ?lift ?floor)"] = set()
    first_component = ObservedComponent(previous_state=test_partial_previous_state,
                                        next_state=next_state, call=test_action_call)
    sync_snapshot(elevators_sam_learning, first_component, elevators_observation.grounded_objects)

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=test_partial_previous_state,
                                          next_state=next_state)
    print(str(elevators_sam_learning.partial_domain.actions["move-down-slow"].preconditions))

    second_component = ObservedComponent(previous_state=previous_state, next_state=next_state, call=test_action_call)
    sync_snapshot(elevators_sam_learning, second_component, elevators_observation.grounded_objects)
    elevators_sam_learning._update_action_preconditions(test_action_call)
    discrete_preconditions = extract_preconditions_predicates(
        elevators_sam_learning.partial_domain.actions["move-down-slow"].preconditions)
    assert "(reachable-floor ?lift ?f2)" not in [precond.untyped_representation for precond in discrete_preconditions]


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
    discrete_preconditions = extract_preconditions_predicates(
        elevators_sam_learning.partial_domain.actions["move-down-slow"].preconditions)
    negative_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive}
    assert "(not (lift-at ?lift ?f2))" in negative_preconditions
    assert "(not (above ?f1 ?f2))" in negative_preconditions


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
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    positive_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive}
    negative_preconditions = {precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive}
    assert positive_preconditions.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert all([p.startswith("(not") for p in negative_preconditions])


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
    assert {p.untyped_representation for p in learned_action_data.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_add_new_action_maintains_the_correct_add_and_delete_effects(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    elevators_sam_learning.add_new_action(test_action_call, previous_state, next_state)
    elevators_sam_learning.add_new_action(test_action_call, previous_state, next_state)
    learned_action = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    assert {p.untyped_representation for p in learned_action.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_add_action_does_not_add_the_same_predicates_twice_to_the_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)
    learned_action = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action.preconditions)
    positive_preconditions = [precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive]
    negative_preconditions = [precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive]
    assert len(positive_preconditions) == len(set(positive_preconditions))
    assert len(negative_preconditions) == len(set(negative_preconditions))


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
    assert {p.untyped_representation for p in learned_action.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_update_action_does_not_add_the_same_predicates_twice_to_the_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    sync_snapshot(elevators_sam_learning, observation_component, elevators_observation.grounded_objects)
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    elevators_sam_learning.add_new_action(grounded_action=test_action_call,
                                          previous_state=previous_state,
                                          next_state=next_state)
    elevators_sam_learning.update_action(grounded_action=test_action_call, previous_state=previous_state,
                                         next_state=next_state)
    learned_action = elevators_sam_learning.partial_domain.actions[test_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action.preconditions)
    positive_preconditions = [precond.untyped_representation for precond in discrete_preconditions if
                              precond.is_positive]
    negative_preconditions = [precond.untyped_representation for precond in discrete_preconditions if
                              not precond.is_positive]
    assert len(positive_preconditions) == len(set(positive_preconditions))
    assert len(negative_preconditions) == len(set(negative_preconditions))


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
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    preconditions_str = {p.untyped_representation for p in discrete_preconditions if p.is_positive}
    assert preconditions_str.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert {p.untyped_representation for p in learned_action_data.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_deduce_initial_inequality_preconditions_deduce_that_all_objects_with_same_type_should_not_be_equal(
        elevators_sam_learning: SAMLearner):
    elevators_sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = elevators_sam_learning.partial_domain.actions[example_action_name]
    assert action.preconditions.root.inequality_preconditions == {("?f1", "?f2")}


def test_verify_parameter_duplication_removes_inequality_if_found_action_with_duplicated_items_in_observation(
        elevators_sam_learning: SAMLearner):
    elevators_sam_learning.deduce_initial_inequality_preconditions()
    example_action_name = "move-up-slow"
    action = elevators_sam_learning.partial_domain.actions[example_action_name]
    assert action.preconditions.root.inequality_preconditions == {("?f1", "?f2")}
    duplicated_action_call = ActionCall(name=example_action_name, grounded_parameters=["slow-lift", "c1", "c1"])
    elevators_sam_learning._verify_parameter_duplication(duplicated_action_call)
    assert len(action.preconditions.root.inequality_preconditions) == 0


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
    assert len(learned_action_data.preconditions.root.operands) == 0
    assert len(learned_action_data.discrete_effects) == 0


def test_handle_single_trajectory_component_learns_preconditions_and_effects_when_given_a_non_duplicate_component(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    added_action_name = "move-down-slow"
    learned_action_data = elevators_sam_learning.partial_domain.actions[added_action_name]

    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    discrete_preconditions_str = {p.untyped_representation for p in discrete_preconditions}
    assert discrete_preconditions_str.issuperset(
        ["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])
    assert {p.untyped_representation for p in learned_action_data.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_handle_single_trajectory_component_maintains_the_correct_preconditions_invariants_after_multiple_executions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    observation_component = elevators_observation.components[4]
    elevators_sam_learning.current_trajectory_objects = elevators_observation.grounded_objects
    elevators_sam_learning.handle_single_trajectory_component(observation_component)

    learned_action_data = elevators_sam_learning.partial_domain.actions[observation_component.grounded_action_call.name]
    discrete_preconditions = extract_preconditions_predicates(learned_action_data.preconditions)
    positive_preconditions = {p.untyped_representation for p in discrete_preconditions if p.is_positive}
    negative_preconditions = {p.untyped_representation for p in discrete_preconditions if not p.is_positive}
    assert positive_preconditions.issuperset(["(lift-at ?lift ?f1)", "(above ?f2 ?f1)", "(reachable-floor ?lift ?f2)"])

    assert negative_preconditions.issuperset(["(not (lift-at ?lift ?f2))", "(not (above ?f1 ?f2))"])
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
    assert {p.untyped_representation for p in learned_action_data.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}


def test_learn_action_model_does_not_obstruct_the_correctness_of_the_actions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    learned_model, _ = elevators_sam_learning.learn_action_model([elevators_observation])

    learned_action_data = learned_model.actions[observation_component.grounded_action_call.name]
    assert {p.untyped_representation for p in learned_action_data.discrete_effects} == \
           {"(lift-at ?lift ?f2)", "(not (lift-at ?lift ?f1))"}
    print(learned_action_data.to_pddl())


def test_learn_action_model_returns_learned_model(elevators_sam_learning: SAMLearner,
                                                  elevators_observation: Observation):
    learned_model, learning_report = elevators_sam_learning.learn_action_model([elevators_observation])
    print(learning_report)
    print(learned_model.to_pddl())


def test_learn_action_model_with_hard_policy_deletes_negative_preconditions(
        elevators_sam_learning_hard_policy: SAMLearner, elevators_observation: Observation):
    learned_model_ignore, _ = elevators_sam_learning_hard_policy.learn_action_model(
        [elevators_observation])

    for action in learned_model_ignore.actions.values():
        for pre in action.preconditions.root.operands:
            if isinstance(pre, Predicate):
                assert pre.is_positive


def test_learn_action_model_with_hard_negative_precondition_policy_keep_positive_preconditions(
        elevators_sam_learning: SAMLearner, elevators_observation: Observation,
        elevators_sam_learning_hard_policy: SAMLearner):
    learned_model, _ = elevators_sam_learning.learn_action_model([elevators_observation])
    learned_model_ignore, _ = elevators_sam_learning_hard_policy.learn_action_model([elevators_observation])

    for action, action_ignored in zip(learned_model.actions.values(), learned_model_ignore.actions.values()):
        preconds = {prec for prec in action.preconditions.root.operands if isinstance(prec,Predicate)}
        preconds_ignore = {prec for prec in action_ignored.preconditions.root.operands if isinstance(prec,Predicate)}

        difference_ignore_from_classic = preconds_ignore.difference(preconds)
        difference_classic_from_ignore = preconds.difference(preconds_ignore)

        assert len(difference_ignore_from_classic) == 0

        for pre in difference_classic_from_ignore:
            if isinstance(pre, Predicate):
                assert not pre.is_positive


def test_learn_action_model_with_hard_policy_delete_effect_has_positive_precondition(
        elevators_sam_learning_hard_policy: SAMLearner, elevators_observation: Observation):
    learned_model, _ = elevators_sam_learning_hard_policy.learn_action_model(
        [elevators_observation])

    for action in learned_model.actions.values():
        predicates = [pre.untyped_representation for pre in action.preconditions.root.operands if
                      pre.is_positive and isinstance(pre, Predicate)]

        del_effects = [eff for eff in action.discrete_effects if not eff.is_positive and isinstance(eff, Predicate)]

        for del_eff in del_effects:
            del_eff.is_positive = not del_eff.is_positive

        flipped_del_effects = {del_eff.untyped_representation for del_eff in del_effects}

        assert len(del_effects) == len(flipped_del_effects.intersection(predicates))


def test_learn_action_model_with_soft_policy_delete_preconditions_has_add_effect(
    elevators_sam_learning: SAMLearner, elevators_sam_learning_soft_policy: SAMLearner,
        elevators_observation: Observation):
    learned_model, _ = elevators_sam_learning.learn_action_model([elevators_observation])
    learned_model_soft, _ = elevators_sam_learning_soft_policy.learn_action_model(
        [elevators_observation])

    for action, action_ignored in zip(learned_model.actions.values(), learned_model_soft.actions.values()):
        preconds = {prec for prec in action.preconditions.root.operands if isinstance(prec,Predicate)}
        preconds_ignore = {prec for prec in action_ignored.preconditions.root.operands if isinstance(prec,Predicate)}

        difference_ignore_from_classic = preconds_ignore.difference(preconds)
        difference_classic_from_ignore = preconds.difference(preconds_ignore)
        add_effects = [effect.untyped_representation for effect in action.discrete_effects if effect.is_positive]

        assert len(difference_ignore_from_classic) == 0

        for pre in difference_classic_from_ignore:
            if isinstance(pre, Predicate):
                pre_positive_copy = pre.copy()
                pre_positive_copy.is_positive = True
                assert (not pre.is_positive) and (pre_positive_copy.untyped_representation in add_effects)

"""Module test for Universally Conditional SAM."""

import pytest
from pddl_plus_parser.models import Domain, Observation, ActionCall, UniversalPrecondition, Predicate, UniversalEffect
from pytest import fixture

from sam_learning.core import DependencySet
from sam_learning.learners import UniversallyConditionalSAM
from sam_learning.learners.universaly_conditional_sam import create_additional_parameter_name, find_unique_objects_by_type
from tests.consts import sync_snapshot


@fixture()
def nurikabe_conditional_sam(nurikabe_domain: Domain) -> UniversallyConditionalSAM:
    universals_map = {}
    for action in nurikabe_domain.actions:
        universals_map[action] = [*nurikabe_domain.types]
    return UniversallyConditionalSAM(nurikabe_domain, max_antecedents_size=1, universals_map=universals_map)


@fixture()
def satellite_conditional_sam(satellite_numeric_domain: Domain) -> UniversallyConditionalSAM:
    universals_map = {}
    for action in satellite_numeric_domain.actions:
        universals_map[action] = [*satellite_numeric_domain.types]
    return UniversallyConditionalSAM(satellite_numeric_domain, max_antecedents_size=1, universals_map=universals_map)


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name(
    nurikabe_conditional_sam: UniversallyConditionalSAM,
):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, learner_domain.types["cell"])
    assert parameter_name == "?c"


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name_with_index(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain
):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, nurikabe_domain.types["group"])
    assert parameter_name == "?g1"


def test_find_unique_objects_by_type_returns_correct_objects(nurikabe_observation: Observation):
    observation_objects = nurikabe_observation.grounded_objects
    unique_objects = find_unique_objects_by_type(observation_objects)
    assert sum(len(objects) for objects in unique_objects.values()) == len(observation_objects)


def test_initialize_universal_dependencies_adds_additional_parameter_for_each_newly_created_type(
    satellite_conditional_sam: UniversallyConditionalSAM, satellite_adl_domain: Domain, satellite_adl_observation: Observation
):
    grounded_action = ActionCall(name="switch_off", grounded_parameters=["instrument7", "satellite2"])
    satellite_conditional_sam.current_trajectory_objects = satellite_adl_observation.grounded_objects
    sync_snapshot(
        satellite_conditional_sam,
        satellite_adl_observation.components[0],
        satellite_adl_observation.grounded_objects,
        should_include_all_objects=True,
    )
    satellite_conditional_sam._initialize_universal_dependencies(grounded_action)
    for pddl_type in satellite_adl_domain.types:
        if pddl_type == "object":
            continue

        assert pddl_type in satellite_conditional_sam.additional_parameters[grounded_action.name]


def test_initialize_universal_dependencies_adds_additional_parameter_for_each_newly_created_type_for_every_action(
    satellite_conditional_sam: UniversallyConditionalSAM, satellite_adl_domain: Domain, satellite_adl_observation: Observation
):
    sync_snapshot(
        satellite_conditional_sam,
        satellite_adl_observation.components[0],
        satellite_adl_observation.grounded_objects,
        should_include_all_objects=True,
    )
    for action_name in satellite_conditional_sam.partial_domain.actions:
        grounded_action = ActionCall(name=action_name, grounded_parameters=[])
        satellite_conditional_sam._initialize_universal_dependencies(grounded_action)
        for pddl_type in satellite_adl_domain.types:
            if pddl_type == "object":
                continue

            assert pddl_type in satellite_conditional_sam.additional_parameters[grounded_action.name]

        print(satellite_conditional_sam.additional_parameters[grounded_action.name])


def test_initialize_universal_dependencies_adds_possible_dependencies_for_action_in_dependency_set(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    assert len(nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].possible_antecedents) > 0


def test_initialize_universal_dependencies_creates_dependency_set_for_each_type_of_quantified_object(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    assert len(nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]) == 3


def test_initialize_universal_dependencies_initialize_dependencies_objects_with_new_data(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation
):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_antecedents) > 0


def test_initialize_universal_dependencies_adds_the_new_additional_parameters_for_the_action(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation
):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.additional_parameters[grounded_action.name]) > 0


def test_initialize_universal_dependencies_not_add_dependencies_if_additional_param_does_not_appear(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation
):
    grounded_action = ActionCall(name="start-painting", grounded_parameters=["pos-1-2", "g3", "n1", "n2"])
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert "(not (next n0 ?n2))" not in nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["num"].possible_antecedents


def test_update_observed_effects_does_not_add_universal_effect_if_not_observed_in_post_state(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation
):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )

    nurikabe_conditional_sam._initialize_actions_dependencies(grounded_action)
    nurikabe_conditional_sam._update_observed_effects(
        grounded_action, nurikabe_observation.components[0].previous_state, nurikabe_observation.components[0].next_state
    )
    for possible_effects in nurikabe_conditional_sam.observed_universal_effects[grounded_action.name].values():
        assert len(possible_effects) == 0


def test_find_literals_existing_in_state_correctly_selects_literals_with_the_additional_parameter(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    extra_parameter_name = "?c"
    predicates_in_state = nurikabe_conditional_sam._find_literals_existing_in_state(
        grounded_action=grounded_action,
        grounded_predicates=nurikabe_conditional_sam.triplet_snapshot.previous_state_predicates,
        extra_grounded_object="pos-0-2",
        extra_lifted_object=extra_parameter_name,
    )

    expected_set = {"(connected ?to ?c)", "(connected ?c ?to)"}
    assert predicates_in_state.issuperset(expected_set)


def test_find_literals_existing_in_state_does_not_choose_literals_that_might_match_the_action_parameter_without_the_added_variable(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    extra_parameter_name = "?c"
    predicates_in_state = nurikabe_conditional_sam._find_literals_existing_in_state(
        grounded_action=grounded_action,
        grounded_predicates=nurikabe_conditional_sam.triplet_snapshot.previous_state_predicates,
        extra_grounded_object="pos-0-2",
        extra_lifted_object=extra_parameter_name,
    )

    assert not predicates_in_state.issubset({"(not (painted ?to))", "(moving)", "(connected ?from ?to)"})


def test_remove_existing_previous_state_quantified_dependencies_removes_correct_predicates_from_literals_that_are_effects_only(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    not_dependencies = {"(moving )", "(not (painted ?to))", "(connected ?from ?to)"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].possible_antecedents[
            "(painted ?c)"
        ]


def test_remove_existing_previous_state_quantified_dependencies_tries_to_remove_from_a_literal_that_does_not_contain_a_matching_parameter_will_will_fail(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = ActionCall(name="start-painting", grounded_parameters=["pos-1-2", "g3", "n1", "n2"])
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    try:
        nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    except KeyError:
        pytest.fail("KeyError was raised")


def test_remove_existing_previous_state_quantified_dependencies_shrinks_dependencies_from_previous_iteration(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    previous_dependencies_size = 16
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    assert any(
        len(dependencies) < previous_dependencies_size
        for dependencies in nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].possible_antecedents.values()
    )


def test_remove_non_existing_previous_state_quantified_dependencies_does_not_raise_error(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    try:
        nurikabe_conditional_sam._remove_non_existing_previous_state_quantified_dependencies(grounded_action, previous_state, next_state)

    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_construct_universal_effects_from_dependency_set_will_not_add_conditional_effect_if_the_literal_not_observed_as_an_effect(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}
    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    test_action = nurikabe_conditional_sam.partial_domain.actions["move"]
    nurikabe_conditional_sam.observed_universal_effects[test_action.name]["cell"] = {"(blocked ?c)"}
    nurikabe_conditional_sam.additional_parameters[test_action.name]["cell"] = "?c"
    nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects.add(UniversalEffect("?c", nurikabe_domain.types["cell"]))
    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(test_action, dependency_set, "cell", "(not (available ?c))")
    universal_effect = [
        effect
        for effect in nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects
        if effect.quantified_type.name == "cell"
    ][0]
    assert len(universal_effect.conditional_effects) == 0


def test_construct_universal_effects_from_dependency_set_when_the_literal_is_safe_returns_the_effect_with_the_correct_result(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}
    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    dependency_set.possible_disjunctive_antecedents = {"(blocked ?c)": [], "(not (available ?c))": []}
    test_action = nurikabe_conditional_sam.partial_domain.actions["move"]
    nurikabe_conditional_sam.observed_universal_effects[test_action.name]["cell"] = {"(blocked ?c)"}
    nurikabe_conditional_sam.additional_parameters[test_action.name]["cell"] = "?c"
    nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects.add(UniversalEffect("?c", nurikabe_domain.types["cell"]))
    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(test_action, dependency_set, "cell", "(blocked ?c)")
    universal_effect = [
        effect
        for effect in nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects
        if effect.quantified_type.name == "cell"
    ][0]
    print(str(universal_effect))

    effect = universal_effect.conditional_effects.pop()
    assert len(effect.discrete_effects) == 1
    assert [eff.untyped_representation for eff in effect.discrete_effects] == ["(blocked ?c)"]


def test_construct_universal_effects_from_dependency_set_constructs_with_more_than_one_conditional_effect_when_called_with_two_safe_effects(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    tested_type = "cell"
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}
    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    dependency_set.possible_disjunctive_antecedents = {"(blocked ?c)": [], "(not (available ?c))": []}
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]
    nurikabe_conditional_sam.observed_universal_effects[test_action.name]["cell"] = {"(blocked ?c)", "(not (available ?c))"}
    nurikabe_conditional_sam.additional_parameters[test_action.name]["cell"] = "?c"
    nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects.add(UniversalEffect("?c", nurikabe_domain.types["cell"]))

    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(test_action, dependency_set, tested_type, "(not (available ?c))")
    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(test_action, dependency_set, tested_type, "(blocked ?c)")

    universal_effect = [
        effect
        for effect in nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects
        if effect.quantified_type.name == "cell"
    ][0]

    assert len(universal_effect.conditional_effects) == 2
    for effect in test_action.universal_effects:
        print(str(effect))


def test_construct_restrictive_universal_preconditions_creates_correct_restrictive_preconditions_for_the_action(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    tested_type = "cell"
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}

    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    dependency_set.possible_disjunctive_antecedents = {"(blocked ?c)": [], "(not (available ?c))": []}
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]
    nurikabe_conditional_sam._construct_restrictive_preconditions(test_action, dependency_set, "(blocked ?c)", tested_type)

    # "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c)))))"
    assert len(test_action.preconditions.root.operands) == 1
    universal_precondition = test_action.preconditions.root.operands.pop()
    assert isinstance(universal_precondition, UniversalPrecondition)
    assert universal_precondition.quantified_parameter == "?c"
    assert universal_precondition.quantified_type.name == "cell"
    assert len(universal_precondition.operands) == 1
    first_layer_condition = universal_precondition.operands.pop()
    assert first_layer_condition.binary_operator == "or"
    for cond in first_layer_condition.operands:
        if isinstance(cond, Predicate):
            assert cond.untyped_representation in ["(blocked ?c)", "(not (connected ?to ?c))"]


def test_construct_restrictive_universal_preconditions_creates_correct_restrictive_preconditions_for_the_action_when_literal_is_effect(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    tested_type = "cell"
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects

    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}
    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    dependency_set.possible_disjunctive_antecedents = {"(blocked ?c)": [], "(not (available ?c))": []}
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]

    nurikabe_conditional_sam.observed_universal_effects[test_action.name][tested_type].add("(blocked ?c)")
    nurikabe_conditional_sam._construct_restrictive_preconditions(test_action, dependency_set, "(blocked ?c)", tested_type)
    # "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c))) (and (connected ?to ?c))))"
    assert len(test_action.preconditions.root.operands) == 1
    universal_precondition = test_action.preconditions.root.operands.pop()
    assert isinstance(universal_precondition, UniversalPrecondition)
    assert universal_precondition.quantified_parameter == "?c"
    assert universal_precondition.quantified_type.name == "cell"
    assert len(universal_precondition.operands) == 1
    first_layer_condition = universal_precondition.operands.pop()
    assert first_layer_condition.binary_operator == "or"
    for cond in first_layer_condition.operands:
        if isinstance(cond, Predicate):
            assert cond.untyped_representation in ["(blocked ?c)", "(not (connected ?to ?c))", "(connected ?to ?c)"]


def test_construct_restrictive_universal_effect_constructs_correct_restrictive_universal_effect(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain
):
    sync_snapshot(
        nurikabe_conditional_sam, nurikabe_observation.components[0], nurikabe_observation.grounded_objects, should_include_all_objects=True
    )
    tested_type = "cell"
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])

    extended_signature = {**nurikabe_domain.actions[grounded_action.name].signature, "?c": nurikabe_domain.types["cell"]}
    dependency_set = DependencySet(max_size_antecedents=1, action_signature=extended_signature, domain_constants=nurikabe_domain.constants)
    dependency_set.possible_antecedents = {"(blocked ?c)": [{"(connected ?to ?c)"}], "(not (available ?c))": [{"(available ?c)"}]}
    dependency_set.possible_disjunctive_antecedents = {"(blocked ?c)": [], "(not (available ?c))": []}
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]

    nurikabe_conditional_sam.quantified_antecedents[grounded_action.name][tested_type] = dependency_set
    nurikabe_conditional_sam.observed_universal_effects[test_action.name][tested_type].add("(blocked ?c)")
    nurikabe_conditional_sam._construct_restrictive_universal_effect(test_action, tested_type, "(blocked ?c)")

    non_empty_effects = [eff for eff in test_action.universal_effects if len(eff.conditional_effects) > 0]
    assert len(non_empty_effects) == 1
    universal_effect = non_empty_effects[0]
    assert universal_effect.quantified_parameter == "?c"
    assert universal_effect.quantified_type.name == "cell"
    assert len(universal_effect.conditional_effects) == 1
    conditional_effect = universal_effect.conditional_effects.pop()
    assert [eff.untyped_representation for eff in conditional_effect.discrete_effects] == ["(blocked ?c)"]


def test_handle_single_trajectory_component_updates_correct_observed_effects_for_conditional_effects_and_universal_effects(
    nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation
):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = nurikabe_conditional_sam.observed_effects[first_component.grounded_action_call.name]
    assert actual_effects == expected_effects
    for observed_effects in nurikabe_conditional_sam.observed_universal_effects[first_component.grounded_action_call.name].values():
        assert len(observed_effects) == 0


def test_learn_action_model_learns_restrictive_action_mode(nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    learned_model, _ = nurikabe_conditional_sam.learn_action_model([nurikabe_observation])
    print(learned_model.to_pddl())

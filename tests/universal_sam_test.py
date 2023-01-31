"""Module test for Universally Conditional SAM."""

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation, ActionCall
from pytest import fixture

from sam_learning.core import DependencySet
from sam_learning.learners import UniversallyConditionalSAM
from sam_learning.learners.universaly_conditional_sam import create_additional_parameter_name, \
    find_unique_objects_by_type
from tests.consts import NURIKABE_PROBLEM_PATH, NURIKABE_TRAJECTORY_PATH, ADL_SATELLITE_DOMAIN_PATH, \
    ADL_SATELLITE_PROBLEM_PATH, \
    ADL_SATELLITE_TRAJECTORY_PATH, NURIKABE_DOMAIN_PATH


@fixture()
def nurikabe_domain() -> Domain:
    return DomainParser(NURIKABE_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def nurikabe_problem(nurikabe_domain: Domain) -> Problem:
    return ProblemParser(problem_path=NURIKABE_PROBLEM_PATH, domain=nurikabe_domain).parse_problem()


@fixture()
def nurikabe_observation(nurikabe_domain: Domain, nurikabe_problem: Problem) -> Observation:
    return TrajectoryParser(nurikabe_domain, nurikabe_problem).parse_trajectory(NURIKABE_TRAJECTORY_PATH)


@fixture()
def satellite_domain() -> Domain:
    return DomainParser(ADL_SATELLITE_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def satellite_problem(satellite_domain: Domain) -> Problem:
    return ProblemParser(problem_path=ADL_SATELLITE_PROBLEM_PATH, domain=satellite_domain).parse_problem()


@fixture()
def satellite_observation(satellite_domain: Domain, satellite_problem: Problem) -> Observation:
    return TrajectoryParser(satellite_domain, satellite_problem).parse_trajectory(ADL_SATELLITE_TRAJECTORY_PATH)


@fixture()
def nurikabe_conditional_sam(nurikabe_domain: Domain) -> UniversallyConditionalSAM:
    return UniversallyConditionalSAM(nurikabe_domain, max_antecedents_size=1)


@fixture()
def satellite_conditional_sam(satellite_domain: Domain) -> UniversallyConditionalSAM:
    return UniversallyConditionalSAM(satellite_domain, max_antecedents_size=1)


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name(
        nurikabe_conditional_sam: UniversallyConditionalSAM):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, learner_domain.types["cell"])
    assert parameter_name == '?c'


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name_with_index(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, nurikabe_domain.types["group"])
    assert parameter_name == '?g1'


def test_find_unique_objects_by_type_returns_correct_objects(nurikabe_observation: Observation):
    observation_objects = nurikabe_observation.grounded_objects
    unique_objects = find_unique_objects_by_type(observation_objects)
    assert sum(len(objects) for objects in unique_objects.values()) == len(observation_objects)


def test_initialize_universal_dependencies_adds_additional_parameter_for_each_newly_created_type(
        satellite_conditional_sam: UniversallyConditionalSAM, satellite_domain: Domain,
        satellite_observation: Observation):
    grounded_action = ActionCall(name="switch_off", grounded_parameters=["instrument7", "satellite2"])
    satellite_conditional_sam.current_trajectory_objects = satellite_observation.grounded_objects
    satellite_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=satellite_observation.components[0].previous_state,
        next_state=satellite_observation.components[0].next_state)
    satellite_conditional_sam._initialize_universal_dependencies(grounded_action)

    for pddl_type in satellite_domain.types:
        if pddl_type == "object":
            continue

        assert pddl_type in satellite_conditional_sam.additional_parameters[grounded_action.name]


def test_initialize_universal_dependencies_adds_additional_parameter_for_each_newly_created_type_for_every_action(
        satellite_conditional_sam: UniversallyConditionalSAM, satellite_domain: Domain,
        satellite_observation: Observation):
    for action_name in satellite_conditional_sam.partial_domain.actions:
        grounded_action = ActionCall(name=action_name, grounded_parameters=[])
        satellite_conditional_sam.current_trajectory_objects = satellite_observation.grounded_objects
        satellite_conditional_sam._create_fully_observable_triplet_predicates(
            current_action=grounded_action,
            previous_state=satellite_observation.components[0].previous_state,
            next_state=satellite_observation.components[0].next_state)
        satellite_conditional_sam._initialize_universal_dependencies(grounded_action)
        for pddl_type in satellite_domain.types:
            if pddl_type == "object":
                continue

            assert pddl_type in satellite_conditional_sam.additional_parameters[grounded_action.name]
        print(satellite_conditional_sam.additional_parameters[grounded_action.name])


def test_initialize_universal_dependencies_adds_possible_dependencies_for_action_in_dependency_set(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain,
        nurikabe_observation: Observation):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].dependencies) > 0


def test_initialize_universal_dependencies_creates_dependency_set_for_each_type_of_quantified_object(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain,
        nurikabe_observation: Observation):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]) == 3


def test_initialize_universal_dependencies_initialize_dependencies_objects_with_new_data(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain,
        nurikabe_observation: Observation):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_antecedents) > 0


def test_initialize_universal_dependencies_adds_the_new_additional_parameters_for_the_action(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain,
        nurikabe_observation: Observation):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.additional_parameters[grounded_action.name]) > 0


def test_initialize_universal_dependencies_not_add_dependencies_if_additional_param_does_not_appear(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_domain: Domain,
        nurikabe_observation: Observation):
    grounded_action = ActionCall(name="start-painting", grounded_parameters=["pos-1-2", "g3", "n1", "n2"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert "(not (next n0 ?n2))" not in nurikabe_conditional_sam.quantified_antecedents[grounded_action.name][
        "num"].dependencies


def test_update_observed_effects_does_not_add_universal_effect_if_not_observed_in_post_state(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)

    nurikabe_conditional_sam._initialize_actions_dependencies(grounded_action)
    nurikabe_conditional_sam._update_observed_effects(grounded_action,
                                                      nurikabe_observation.components[0].previous_state,
                                                      nurikabe_observation.components[0].next_state)
    assert len(nurikabe_conditional_sam.observed_universal_effects[grounded_action.name]) == 0


def test_find_literals_existing_in_state_correctly_selects_literals_with_the_additional_parameter(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)
    extra_parameter_name = "?c"
    predicates_in_state = nurikabe_conditional_sam._find_literals_existing_in_state(
        grounded_action=grounded_action,
        positive_predicates=nurikabe_conditional_sam.previous_state_positive_predicates,
        negative_predicates=nurikabe_conditional_sam.previous_state_negative_predicates,
        extra_grounded_object="pos-0-2",
        extra_lifted_object=extra_parameter_name)

    expected_set = {"(connected ?to ?c)", "(connected ?c ?to)"}
    assert predicates_in_state.issuperset(expected_set)


def test_find_literals_existing_in_state_does_not_choose_literals_that_might_match_the_action_parameter_without_the_added_variable(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)

    extra_parameter_name = "?c"
    predicates_in_state = nurikabe_conditional_sam._find_literals_existing_in_state(
        grounded_action=grounded_action,
        positive_predicates=nurikabe_conditional_sam.previous_state_positive_predicates,
        negative_predicates=nurikabe_conditional_sam.previous_state_negative_predicates,
        extra_grounded_object="pos-0-2",
        extra_lifted_object=extra_parameter_name)

    assert not predicates_in_state.issubset({"(not (painted ?to))", "(moving)", "(connected ?from ?to)"})


def test_remove_existing_previous_state_quantified_dependencies_removes_correct_predicates_from_literals_that_are_effects_only(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    not_dependencies = {"(moving )", "(not (painted ?to))", "(connected ?from ?to)"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in \
               nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].dependencies[
                   "(painted ?c)"]


def test_remove_existing_previous_state_quantified_dependencies_tries_to_remove_from_a_literal_that_does_not_contain_a_matching_parameter_not_will_fail(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = ActionCall(name="start-painting", grounded_parameters=["pos-1-2", "g3", "n1", "n2"])
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    try:
        nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    except KeyError:
        pytest.fail("KeyError was raised")


def test_remove_existing_previous_state_quantified_dependencies_shrinks_dependencies_from_previous_iteration(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    previous_dependencies_size = 16
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    assert any(len(dependencies) < previous_dependencies_size for dependencies in
               nurikabe_conditional_sam.quantified_antecedents[grounded_action.name]["cell"].dependencies.values())


def test_remove_non_existing_previous_state_quantified_dependencies_does_not_raise_error(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    try:
        nurikabe_conditional_sam._remove_non_existing_previous_state_quantified_dependencies(
            grounded_action, previous_state, next_state)

    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_construct_universal_effects_from_dependency_set_constructs_correct_conditional_effect(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation,
        nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)

    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.dependencies = {
        "(blocked ?c)": [{"(connected ?to ?c)"}],
        "(not (available ?c))": [{"(available ?c)"}]
    }
    test_action = nurikabe_conditional_sam.partial_domain.actions["move"]
    ground_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    nurikabe_conditional_sam._initialize_universal_dependencies(ground_action)
    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(
        test_action, dependency_set, "cell", "(blocked ?c)")
    universal_effect = \
        [effect for effect in nurikabe_conditional_sam.partial_domain.actions[test_action.name].universal_effects
         if effect.quantified_type.name == "cell"][0]
    print(universal_effect)

    effect = universal_effect.conditional_effects.pop()
    assert len(effect.positive_conditions) == 1


def test_construct_universal_effects_from_dependency_set_constructs_correct_universal_effect(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    tested_type = "cell"
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)

    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.dependencies = {
        "(blocked ?c)": [{"(connected ?to ?c)"}],
        "(not (available ?c))": [{"(available ?c)"}]
    }
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]
    nurikabe_conditional_sam._construct_universal_effects_from_dependency_set(
        test_action, dependency_set, tested_type, "(not (available ?c))")
    assert any(len(effect.conditional_effects) > 0 for effect in test_action.universal_effects)
    for effect in test_action.universal_effects:
        print(str(effect))


def test_construct_restrictive_universal_preconditions_creates_correct_restrictive_preconditions_for_the_action(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    tested_type = "cell"
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)

    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.dependencies = {
        "(blocked ?c)": [{"(connected ?to ?c)"}],
        "(not (available ?c))": [{"(available ?c)"}]
    }
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]
    nurikabe_conditional_sam._construct_restrictive_preconditions(
        test_action, dependency_set, "(blocked ?c)", tested_type)

    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c)))))"]


def test_construct_restrictive_universal_preconditions_creates_correct_restrictive_preconditions_for_the_action_when_literal_is_effect(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    tested_type = "cell"
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)

    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.dependencies = {
        "(blocked ?c)": [{"(connected ?to ?c)"}],
        "(not (available ?c))": [{"(available ?c)"}]
    }
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]

    nurikabe_conditional_sam.observed_universal_effects[test_action.name][tested_type].add("(blocked ?c)")
    nurikabe_conditional_sam._construct_restrictive_preconditions(
        test_action, dependency_set, "(blocked ?c)", tested_type)

    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c))) (and (connected ?to ?c))))"]


def test_construct_restrictive_universal_effect_constructs_correct_restrictive_universal_effect(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    tested_type = "cell"
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = ActionCall(name="move-painting", grounded_parameters=["pos-0-0", "pos-0-1", "g1", "n1", "n2"])
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)

    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.dependencies = {
        "(blocked ?c)": [{"(connected ?to ?c)"}],
        "(not (available ?c))": [{"(available ?c)"}]
    }
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    test_action = nurikabe_conditional_sam.partial_domain.actions["move-painting"]

    nurikabe_conditional_sam.quantified_antecedents[grounded_action.name][tested_type] = dependency_set
    nurikabe_conditional_sam.observed_universal_effects[test_action.name][tested_type].add("(blocked ?c)")
    nurikabe_conditional_sam._construct_restrictive_universal_effect(
        test_action, tested_type, "(blocked ?c)")

    for universal_effect in test_action.universal_effects:
        print(universal_effect)


def test_handle_single_trajectory_component_updates_correct_observed_effects_for_conditional_effects_and_universal_effects(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = nurikabe_conditional_sam.observed_effects[first_component.grounded_action_call.name]
    assert actual_effects == expected_effects
    for observed_effects in nurikabe_conditional_sam.observed_universal_effects[
        first_component.grounded_action_call.name].values():
        assert len(observed_effects) == 0


def test_verify_and_construct_safe_conditional_effects_does_not_change_observed_effects(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    test_action = nurikabe_conditional_sam.partial_domain.actions[first_component.grounded_action_call.name]
    nurikabe_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = nurikabe_conditional_sam.observed_effects[first_component.grounded_action_call.name]


def test_learn_action_model_learns_restrictive_action_mode(
        nurikabe_conditional_sam: UniversallyConditionalSAM, nurikabe_observation: Observation):
    learned_model, _ = nurikabe_conditional_sam.learn_action_model([nurikabe_observation])
    print(learned_model.to_pddl())

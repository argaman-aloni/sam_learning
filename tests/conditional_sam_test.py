"""Module test for Conditional SAM."""
from typing import Set

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation, GroundedPredicate, ActionCall, PDDLObject
from pytest import fixture

from sam_learning.core import DependencySet
from sam_learning.learners import ConditionalSAM
from sam_learning.learners.conditional_sam import extract_predicate_data, create_additional_parameter_name, \
    find_unique_objects_by_type
from tests.consts import SPIDER_DOMAIN_PATH, SPIDER_PROBLEM_PATH, SPIDER_TRAJECTORY_PATH, NURIKABE_DOMAIN_PATH, \
    NURIKABE_PROBLEM_PATH, NURIKABE_TRAJECTORY_PATH, ADL_SATELLITE_DOMAIN_PATH, ADL_SATELLITE_PROBLEM_PATH, \
    ADL_SATELLITE_TRAJECTORY_PATH


@fixture()
def spider_domain() -> Domain:
    return DomainParser(SPIDER_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def spider_problem(spider_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SPIDER_PROBLEM_PATH, domain=spider_domain).parse_problem()


@fixture()
def spider_observation(spider_domain: Domain, spider_problem: Problem) -> Observation:
    return TrajectoryParser(spider_domain, spider_problem).parse_trajectory(SPIDER_TRAJECTORY_PATH)


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
def conditional_sam(spider_domain: Domain) -> ConditionalSAM:
    return ConditionalSAM(spider_domain, max_antecedents_size=1)


@fixture()
def nurikabe_conditional_sam(nurikabe_domain: Domain) -> ConditionalSAM:
    return ConditionalSAM(nurikabe_domain, max_antecedents_size=1)


@fixture()
def satellite_conditional_sam(satellite_domain: Domain) -> ConditionalSAM:
    return ConditionalSAM(satellite_domain, max_antecedents_size=1)


@fixture()
def positive_initial_state_predicates(spider_observation: Observation) -> Set[GroundedPredicate]:
    initial_state = spider_observation.components[0].previous_state
    initial_state_predicates = set()
    for predicate in initial_state.state_predicates.values():
        initial_state_predicates.update(predicate)
    return initial_state_predicates


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name(
        nurikabe_conditional_sam: ConditionalSAM):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    additional_object = PDDLObject(name="c4", type=learner_domain.types["cell"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, additional_object)
    assert parameter_name == '?c'


def test_create_additional_parameter_name_creates_a_parameter_name_based_on_the_type_and_action_name_with_index(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain):
    learner_domain = nurikabe_conditional_sam.partial_domain
    action_call = ActionCall(name="move-painting", grounded_parameters=["pos-5-0", "pos-4-0", "g0", "n1", "n0"])
    parameter_name = create_additional_parameter_name(learner_domain, action_call, nurikabe_domain.types["group"])
    assert parameter_name == '?g1'


def test_find_unique_objects_by_type_returns_correct_objects(spider_observation: Observation):
    observation_objects = spider_observation.grounded_objects
    unique_objects = find_unique_objects_by_type(observation_objects)
    assert sum(len(objects) for objects in unique_objects.values()) == len(observation_objects)


def test_initialize_actions_dependencies_adds_correct_dependencies(conditional_sam: ConditionalSAM,
                                                                   spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    assert conditional_sam.dependency_set[grounded_action.name] is not None


def test_initialize_universal_dependencies_adds_additional_parameter_for_each_newly_created_type(
        satellite_conditional_sam: ConditionalSAM, satellite_domain: Domain, satellite_observation: Observation):
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
        satellite_conditional_sam: ConditionalSAM, satellite_domain: Domain, satellite_observation: Observation):
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
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name]["cell"].dependencies) > 0


def test_initialize_universal_dependencies_creates_dependency_set_for_each_type_of_quantified_object(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation):
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=nurikabe_observation.components[0].previous_state,
        next_state=nurikabe_observation.components[0].next_state)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name]) == 3


def test_initialize_universal_dependencies_initialize_dependencies_objects_with_new_data(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.quantified_dependency_set) > 0


def test_initialize_universal_dependencies_adds_the_new_additional_parameters_for_the_action(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation):
    grounded_action = ActionCall(name="move", grounded_parameters=["pos-0-0", "pos-0-1"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert len(nurikabe_conditional_sam.additional_parameters[grounded_action.name]) > 0


def test_initialize_universal_dependencies_not_add_dependencies_if_additional_param_does_not_appear(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_domain: Domain, nurikabe_observation: Observation):
    grounded_action = ActionCall(name="start-painting", grounded_parameters=["pos-1-2", "g3", "n1", "n2"])
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)

    assert "(not (next n0 ?n2))" not in nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name][
        "num"].dependencies


def test_update_observed_effects_adds_the_observed_effects_to_the_correct_set_for_conditional_effects(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam._update_observed_effects(grounded_action, spider_observation.components[0].previous_state,
                                             spider_observation.components[0].next_state)
    assert len(conditional_sam.observed_effects[grounded_action.name]) > 0
    print(conditional_sam.observed_effects[grounded_action.name])


def test_update_observed_effects_does_not_add_universal_effect_if_not_observed_in_post_state(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam._update_observed_effects(grounded_action, spider_observation.components[0].previous_state,
                                             spider_observation.components[0].next_state)
    assert len(conditional_sam.observed_universal_effects[grounded_action.name]) == 0


def test_find_literals_not_in_state_correctly_sets_the_literals_that_do_not_appear_in_the_state(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    predicates_not_in_state = conditional_sam._find_literals_not_in_state(
        grounded_action=grounded_action,
        positive_predicates=conditional_sam.previous_state_positive_predicates,
        negative_predicates=conditional_sam.previous_state_negative_predicates)

    negative_preconditions = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                              "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                              "(currently-dealing )"}

    assert negative_preconditions.issubset(predicates_not_in_state)


def test_find_literals_existing_in_state_correctly_sets_the_literals_that_do_appear_in_the_state(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    predicates_not_in_state = conditional_sam._find_literals_existing_in_state(
        grounded_action=grounded_action,
        positive_predicates=conditional_sam.previous_state_positive_predicates,
        negative_predicates=conditional_sam.previous_state_negative_predicates)

    negated_negative_preconditions = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                                      "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                                      "(currently-dealing )"}

    assert not negated_negative_preconditions.issubset(predicates_not_in_state)


def test_find_literals_existing_in_state_correctly_selects_literals_with_the_additional_parameter(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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
    assert predicates_in_state.issuperset(expected_set) and predicates_in_state.issubset(expected_set)


def test_find_literals_existing_in_state_does_not_choose_literals_that_might_match_the_action_parameter_without_the_added_variable(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
    previous_state = nurikabe_observation.components[0].previous_state
    grounded_action = nurikabe_observation.components[0].grounded_action_call
    next_state = nurikabe_observation.components[0].next_state
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    nurikabe_conditional_sam._initialize_universal_dependencies(grounded_action)
    nurikabe_conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    not_dependencies = {"(moving )", "(not (painted ?to))", "(robot-pos ?from)"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in \
               nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name]["cell"].dependencies[
                   "(painted ?c)"]


def test_remove_existing_previous_state_quantified_dependencies_tries_to_remove_from_a_literal_that_does_not_contain_a_matching_parameter_not_will_fail(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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


def test_remove_existing_previous_state_quantified_dependencies_does_not_get_key_error(
        conditional_sam: ConditionalSAM, spider_observation: Observation, spider_domain: Domain):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = ActionCall(name="finish-collecting-deck",
                                 grounded_parameters=["card-d0-s3-v4", "card-d0-s0-v4", "pile-0"])
    next_state = spider_observation.components[0].next_state
    conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state, should_ignore_action=True)
    conditional_sam._initialize_universal_dependencies(grounded_action)
    try:
        conditional_sam._remove_existing_previous_state_quantified_dependencies(grounded_action)
    except KeyError:
        pytest.fail("KeyError was raised")


def test_remove_existing_previous_state_quantified_dependencies_shrinks_dependencies_from_previous_iteration(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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
               nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name]["cell"].dependencies.values())


def test_remove_non_existing_previous_state_dependencies_removes_correct_predicates_from_literals_that_are_effects_only(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = spider_observation.components[0].grounded_action_call
    next_state = spider_observation.components[0].next_state
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam._update_observed_effects(grounded_action, previous_state, next_state)
    conditional_sam._remove_non_existing_previous_state_dependencies(grounded_action, previous_state, next_state)
    not_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                        "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                        "(currently-dealing )"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in conditional_sam.dependency_set[grounded_action.name].dependencies[
            "(currently-dealing )"]


def test_remove_non_existing_previous_state_quantified_dependencies_does_not_raise_error(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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


def test_remove_existing_previous_state_dependencies_removes_correct_predicates_from_literals_not_in_effects(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = spider_observation.components[0].grounded_action_call
    next_state = spider_observation.components[0].next_state
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam._update_observed_effects(grounded_action, previous_state, next_state)
    conditional_sam._remove_existing_previous_state_dependencies(grounded_action)
    positive_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                             "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                             "(currently-dealing )"}

    negative_dependencies = {f"(not {dep})" for dep in
                             positive_dependencies.difference({"(currently-collecting-deck )"})}

    tested_literal = "(not (currently-collecting-deck ))"
    for dependency in positive_dependencies:
        assert {dependency} in conditional_sam.dependency_set[grounded_action.name].dependencies[tested_literal]

    for dependency in negative_dependencies:
        assert {dependency} in conditional_sam.dependency_set[grounded_action.name].dependencies[
            tested_literal]


def test_update_effects_data_updates_the_relevant_effects_and_removes_irrelevant_literals_from_dependency_set(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = spider_observation.components[0].grounded_action_call
    next_state = spider_observation.components[0].next_state
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    initialized_add_effects = conditional_sam.partial_domain.actions[grounded_action.name].add_effects
    initialized_delete_effects = conditional_sam.partial_domain.actions[grounded_action.name].delete_effects
    conditional_sam._update_effects_data(grounded_action, previous_state, next_state)
    positive_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                             "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                             "(currently-dealing )"}

    negative_dependencies = {f"(not {dep})" for dep in
                             positive_dependencies.difference({"(currently-collecting-deck )"})}

    tested_literal = "(not (currently-collecting-deck ))"
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects) <= len(initialized_add_effects)
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects) <= len(
        initialized_delete_effects)

    for dependency in positive_dependencies:
        assert {dependency} in conditional_sam.dependency_set[grounded_action.name].dependencies[tested_literal]
        assert {dependency} not in conditional_sam.dependency_set[grounded_action.name].dependencies[
            "(currently-dealing )"]

    for dependency in negative_dependencies:
        assert {dependency} in conditional_sam.dependency_set[grounded_action.name].dependencies[tested_literal]


def test_add_new_action_updates_action_negative_preconditions(conditional_sam: ConditionalSAM,
                                                              spider_observation: Observation,
                                                              spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam.add_new_action(grounded_action,
                                   spider_observation.components[0].previous_state,
                                   spider_observation.components[0].next_state)

    added_action = conditional_sam.partial_domain.actions[grounded_action.name]
    negative_preconditions = {f"(not {precondition.untyped_representation})" for precondition in
                              added_action.negative_preconditions}
    assert negative_preconditions.issuperset({"(not (currently-updating-movable ))",
                                              "(not (currently-updating-unmovable ))",
                                              "(not (currently-updating-part-of-tableau ))",
                                              "(not (currently-collecting-deck ))",
                                              "(not (currently-dealing ))"})


def test_add_new_action_updates_action_positive_preconditions(conditional_sam: ConditionalSAM,
                                                              spider_observation: Observation,
                                                              spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam.add_new_action(grounded_action,
                                   spider_observation.components[0].previous_state,
                                   spider_observation.components[0].next_state)

    added_action = conditional_sam.partial_domain.actions[grounded_action.name]
    positive_preconditions = {precondition.untyped_representation for precondition in
                              added_action.positive_preconditions}
    assert len(positive_preconditions) == 0


def test_add_new_action_updates_action_effects(conditional_sam: ConditionalSAM,
                                               spider_observation: Observation,
                                               spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    initialized_add_effects = conditional_sam.partial_domain.actions[grounded_action.name].add_effects
    initialized_delete_effects = conditional_sam.partial_domain.actions[grounded_action.name].delete_effects
    conditional_sam.add_new_action(grounded_action,
                                   spider_observation.components[0].previous_state,
                                   spider_observation.components[0].next_state)

    added_action = conditional_sam.partial_domain.actions[grounded_action.name]
    assert len(added_action.add_effects) <= len(initialized_add_effects)
    assert len(added_action.delete_effects) <= len(initialized_delete_effects)


def test_update_action_updates_preconditions(conditional_sam: ConditionalSAM,
                                             spider_observation: Observation,
                                             spider_domain: Domain):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam.add_new_action(grounded_action,
                                   spider_observation.components[0].previous_state,
                                   spider_observation.components[0].next_state)
    conditional_sam.update_action(grounded_action,
                                  spider_observation.components[0].previous_state,
                                  spider_observation.components[0].next_state)

    added_action = conditional_sam.partial_domain.actions[grounded_action.name]
    negative_preconditions = {f"(not {precondition.untyped_representation})" for precondition in
                              added_action.negative_preconditions}
    assert negative_preconditions.issuperset({"(not (currently-updating-movable ))",
                                              "(not (currently-updating-unmovable ))",
                                              "(not (currently-updating-part-of-tableau ))",
                                              "(not (currently-collecting-deck ))",
                                              "(not (currently-dealing ))"})


def test_extract_predicate_data_returns_correct_predicate_when_predicate_contains_no_parameters(
        conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(currently-updating-movable )"
    learner_action = conditional_sam.partial_domain.actions["start-dealing"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "currently-updating-movable"
    assert len(result_predicate.signature) == 0


def test_extract_predicate_data_returns_correct_predicate_predicate_contains_parameters(
        conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(to-deal ?c ?totableau ?fromdeal ?from)"
    learner_action = conditional_sam.partial_domain.actions["deal-card"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "to-deal"
    assert len(result_predicate.signature) == 4
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["?totableau"].name == "tableau"
    assert result_predicate.signature["?fromdeal"].name == "deal"
    assert result_predicate.signature["?from"].name == "cardposition"


def test_extract_predicate_data_returns_correct_predicate_predicate_contains_constants(
        conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(on ?c discard)"
    learner_action = conditional_sam.partial_domain.actions["collect-card"]
    result_predicate = extract_predicate_data(learner_action, test_predicate, spider_domain.constants)
    assert result_predicate.name == "on"
    assert len(result_predicate.signature) == 2
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["discard"].name == "cardposition"


def test_extract_predicate_data_returns_correct_predicate_with_additional_type(
        conditional_sam: ConditionalSAM, spider_domain: Domain):
    test_predicate = "(can-be-placed-on ?c ?c1)"
    learner_action = conditional_sam.partial_domain.actions["collect-card"]
    result_predicate = extract_predicate_data(
        learner_action, test_predicate, spider_domain.constants,
        additional_parameter="?c1", additional_parameter_type=spider_domain.types["card"])
    assert result_predicate.name == "can-be-placed-on"
    assert len(result_predicate.signature) == 2
    assert result_predicate.signature["?c"].name == "card"
    assert result_predicate.signature["?c1"].name == "card"


def test_construct_conditional_effects_from_dependency_set_constructs_correct_conditional_effect(
        conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.dependencies = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = conditional_sam.partial_domain.actions["deal-card"]

    conditional_effect = conditional_sam._construct_conditional_effect_data(test_action, dependecy_set,
                                                                            "(make-unmovable ?to)")
    assert len(conditional_effect.negative_conditions) == 1
    assert conditional_effect.negative_conditions.pop().untyped_representation == "(can-continue-group ?c ?to)"
    assert len(conditional_effect.positive_conditions) == 0
    add_effect = conditional_effect.add_effects.pop().untyped_representation
    assert add_effect == "(make-unmovable ?to)"


def test_construct_restrictive_preconditions_constructs_correct_restrictive_precondition_string_as_required(
        conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.dependencies = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = conditional_sam.partial_domain.actions["deal-card"]

    conditional_sam._construct_restrictive_preconditions(test_action, dependecy_set, "(make-unmovable ?to)")
    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == ["(or (make-unmovable ?to) (and (can-continue-group ?c ?to)))"]


def test_construct_restrictive_preconditions_constructs_correct_restrictive_precondition_string_as_required_when_is_effect(
        conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.dependencies = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = conditional_sam.partial_domain.actions["deal-card"]

    conditional_sam.observed_effects[test_action.name].add("(make-unmovable ?to)")
    conditional_sam._construct_restrictive_preconditions(test_action, dependecy_set, "(make-unmovable ?to)")
    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(or (make-unmovable ?to) (and (can-continue-group ?c ?to)) (and (not (can-continue-group ?c ?to))))"]


def test_construct_restrictive_conditional_effects_constructs_the_correct_conditional_effect_in_the_action(
        conditional_sam: ConditionalSAM):
    dependecy_set = DependencySet(max_size_antecedents=1)
    dependecy_set.dependencies = {
        "(currently-updating-unmovable )": [{"(not (can-continue-group ?c ?to))"}],
        "(make-unmovable ?to)": [{"(not (can-continue-group ?c ?to))"}]
    }
    test_action = conditional_sam.partial_domain.actions["deal-card"]

    conditional_sam.observed_effects[test_action.name].add("(make-unmovable ?to)")
    conditional_sam._construct_restrictive_conditional_effects(test_action, dependecy_set, "(make-unmovable ?to)")
    conditional_effect = test_action.conditional_effects.pop()
    print(str(conditional_effect))
    assert str(conditional_effect) == "(when (and (not (can-continue-group ?c ?to))) (and (make-unmovable ?to)))"


def test_construct_universal_effects_from_dependency_set_constructs_correct_conditional_effect(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation, nurikabe_domain: Domain):
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


def test_remove_preconditions_from_effects_removes_action_preconditions_from_dependencies(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    conditional_sam.handle_single_trajectory_component(spider_observation.components[0])
    conditional_sam._remove_preconditions_from_dependency_set(
        conditional_sam.partial_domain.actions["start-dealing"])
    assert "(not (currently-updating-movable ))" not in conditional_sam.dependency_set["start-dealing"].dependencies
    assert "(not (currently-updating-unmovable ))" not in conditional_sam.dependency_set["start-dealing"].dependencies
    assert "(not (currently-updating-part-of-tableau ))" not in conditional_sam.dependency_set[
        "start-dealing"].dependencies
    assert "(not (currently-collecting-deck ))" not in conditional_sam.dependency_set["start-dealing"].dependencies
    assert "(not (currently-dealing ))" not in conditional_sam.dependency_set["start-dealing"].dependencies


def test_handle_single_trajectory_component_learns_correct_information(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    conditional_sam.current_trajectory_objects = spider_observation.grounded_objects
    conditional_sam.handle_single_trajectory_component(spider_observation.components[0])
    conditional_sam._remove_preconditions_from_dependency_set(
        conditional_sam.partial_domain.actions["start-dealing"])
    pddl_action = conditional_sam.partial_domain.actions["start-dealing"].to_pddl()
    assert "(not (currently-updating-unmovable ))" in pddl_action
    assert "(not (currently-updating-movable ))" in pddl_action
    assert "(not (currently-collecting-deck ))" in pddl_action
    assert "(not (currently-updating-part-of-tableau ))" in pddl_action
    assert "(not (currently-dealing ))" in pddl_action


def test_construct_universal_effects_from_dependency_set_constructs_correct_universal_effect(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
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
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
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
    nurikabe_conditional_sam._construct_restrictive_universal_preconditions(
        test_action, dependency_set, tested_type, "(blocked ?c)")

    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c)))))"]


def test_construct_restrictive_universal_preconditions_creates_correct_restrictive_preconditions_for_the_action_when_literal_is_effect(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
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
    nurikabe_conditional_sam._construct_restrictive_universal_preconditions(
        test_action, dependency_set, tested_type, "(blocked ?c)")

    print(test_action.manual_preconditions)
    assert test_action.manual_preconditions == [
        "(forall (?c - cell) (or (blocked ?c) (and (not (connected ?to ?c))) (and (connected ?to ?c))))"]


def test_construct_restrictive_universal_effect_constructs_correct_restrictive_universal_effect(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
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

    nurikabe_conditional_sam.quantified_dependency_set[grounded_action.name][tested_type] = dependency_set
    nurikabe_conditional_sam.observed_universal_effects[test_action.name][tested_type].add("(blocked ?c)")
    nurikabe_conditional_sam._construct_restrictive_universal_effect(
        test_action, tested_type, "(blocked ?c)")

    for universal_effect in test_action.universal_effects:
        print(universal_effect)


def test_handle_single_trajectory_component_updates_correct_observed_effects_for_conditional_effects_and_universal_effects(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = nurikabe_conditional_sam.observed_effects[first_component.grounded_action_call.name]
    assert actual_effects == expected_effects
    for observed_effects in nurikabe_conditional_sam.observed_universal_effects[
        first_component.grounded_action_call.name].values():
        assert len(observed_effects) == 0


def test_verify_and_construct_safe_conditional_effects_does_not_change_observed_effects(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    test_action = nurikabe_conditional_sam.partial_domain.actions[first_component.grounded_action_call.name]
    nurikabe_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    expected_effects = {"(robot-pos ?to)", "(not (robot-pos ?from))"}
    actual_effects = nurikabe_conditional_sam.observed_effects[first_component.grounded_action_call.name]


def test_verify_and_construct_safe_conditional_effects_constructs_correct_safe_conditional_effects_with_observed_effects_only(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
    first_component = nurikabe_observation.components[0]
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam.handle_single_trajectory_component(first_component)
    test_action = nurikabe_conditional_sam.partial_domain.actions[first_component.grounded_action_call.name]
    nurikabe_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    assert [effect.untyped_representation for effect in test_action.add_effects] == ["(robot-pos ?to)"]
    assert [effect.untyped_representation for effect in test_action.delete_effects] == ["(robot-pos ?from)"]
    assert len(test_action.conditional_effects) == 0


def test_verify_and_construct_safe_conditional_effects_creates_conditional_effect_when_action_is_unsafe(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
    fourth_component = nurikabe_observation.components[4]
    nurikabe_conditional_sam.current_trajectory_objects = nurikabe_observation.grounded_objects
    nurikabe_conditional_sam.handle_single_trajectory_component(fourth_component)
    test_action = nurikabe_conditional_sam.partial_domain.actions[fourth_component.grounded_action_call.name]
    nurikabe_conditional_sam._verify_and_construct_safe_conditional_effects(test_action)

    assert len(test_action.conditional_effects) > 0


def test_learn_action_model_learns_restrictive_action_mode(
        nurikabe_conditional_sam: ConditionalSAM, nurikabe_observation: Observation):
    learned_model, _ = nurikabe_conditional_sam.learn_action_model([nurikabe_observation])
    print(learned_model.to_pddl())
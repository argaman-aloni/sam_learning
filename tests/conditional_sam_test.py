"""Module test for Conditional SAM."""
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation, GroundedPredicate
from pytest import fixture
from typing import Set

from sam_learning.learners import ConditionalSAM
from tests.consts import SPIDER_DOMAIN_PATH, SPIDER_PROBLEM_PATH, SPIDER_TRAJECTORY_PATH


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
def conditional_sam(spider_domain: Domain) -> ConditionalSAM:
    return ConditionalSAM(spider_domain, max_antecedents_size=2)


@fixture()
def positive_initial_state_predicates(spider_observation: Observation) -> Set[GroundedPredicate]:
    initial_state = spider_observation.components[0].previous_state
    initial_state_predicates = set()
    for predicate in initial_state.state_predicates.values():
        initial_state_predicates.update(predicate)
    return initial_state_predicates


def test_merge_positive_and_negative_predicates_creates_correct_set_with_combined_positive_and_negative_predicates(
        conditional_sam: ConditionalSAM, positive_initial_state_predicates: Set[GroundedPredicate]):
    output_predicates = conditional_sam._merge_positive_and_negative_predicates(positive_initial_state_predicates,
                                                                                set())
    assert len(output_predicates) == len(positive_initial_state_predicates)


def test_initialize_actions_dependencies_sets_correct_effects(conditional_sam: ConditionalSAM,
                                                              spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects) > 0
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects) > 0
    print(len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects))
    print(len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects))


def test_initialize_actions_dependencies_adds_correct_dependencies(conditional_sam: ConditionalSAM,
                                                                   spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    assert conditional_sam.dependency_set[grounded_action.name] is not None


def test_update_action_effects_sets_correct_effects(conditional_sam: ConditionalSAM, spider_observation: Observation):
    grounded_action = spider_observation.components[0].grounded_action_call
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action,
        previous_state=spider_observation.components[0].previous_state,
        next_state=spider_observation.components[0].next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    initialized_add_effects = conditional_sam.partial_domain.actions[grounded_action.name].add_effects
    initialized_delete_effects = conditional_sam.partial_domain.actions[grounded_action.name].delete_effects
    conditional_sam._update_action_effects(grounded_action)
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects) <= len(initialized_add_effects)
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects) <= len(
        initialized_delete_effects)


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

    predicates_not_in_state = conditional_sam._find_literals_not_in_state(
        grounded_action=grounded_action,
        positive_predicates=conditional_sam.previous_state_positive_predicates,
        negative_predicates=conditional_sam.previous_state_negative_predicates)

    negative_preconditions = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                              "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                              "(currently-dealing )"}

    assert not negative_preconditions.issubset(predicates_not_in_state)


def test_remove_non_existing_previous_state_dependencies_removes_correct_predicates_from_literals_that_are_effects_only(
        conditional_sam: ConditionalSAM, spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = spider_observation.components[0].grounded_action_call
    next_state = spider_observation.components[0].next_state
    conditional_sam._create_fully_observable_triplet_predicates(
        current_action=grounded_action, previous_state=previous_state, next_state=next_state)

    conditional_sam._initialize_actions_dependencies(grounded_action)
    conditional_sam._update_action_effects(grounded_action)
    conditional_sam._remove_non_existing_previous_state_dependencies(grounded_action, previous_state, next_state)
    not_dependencies = {"(currently-updating-movable )", "(currently-updating-unmovable )",
                        "(currently-updating-part-of-tableau )", "(currently-collecting-deck )",
                        "(currently-dealing )"}

    for not_dependency in not_dependencies:
        assert {not_dependency} not in conditional_sam.dependency_set[grounded_action.name].dependencies[
            "(currently-dealing )"]

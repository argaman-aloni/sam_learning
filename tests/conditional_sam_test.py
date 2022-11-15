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


def test_create_possible_predicates_creates_correct_set_with_combined_positive_and_negative_predicates(
        conditional_sam: ConditionalSAM, positive_initial_state_predicates: Set[GroundedPredicate]):
    output_predicates = conditional_sam._create_possible_predicates(positive_initial_state_predicates, set())
    assert len(output_predicates) == len(positive_initial_state_predicates)


def test_initialize_action_effects_sets_correct_effects(conditional_sam: ConditionalSAM,
                                                        spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    grounded_action = spider_observation.components[0].grounded_action_call
    observed_objects = spider_observation.grounded_objects
    positive_state_predicates, negative_state_predicates = conditional_sam._create_complete_world_state(
        observed_objects, previous_state)
    conditional_sam._initialize_action_effects(grounded_action, positive_state_predicates, negative_state_predicates)
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects) > 0
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects) > 0
    print(len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects))
    print(len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects))


def test_update_action_effects_sets_correct_effects(conditional_sam: ConditionalSAM, spider_observation: Observation):
    previous_state = spider_observation.components[0].previous_state
    next_state = spider_observation.components[0].next_state
    grounded_action = spider_observation.components[0].grounded_action_call
    observed_objects = spider_observation.grounded_objects
    positive_previous_state_predicates, negative_previous_state_predicates = \
        conditional_sam._create_complete_world_state(observed_objects, previous_state)
    positive_next_state_predicates, negative_next_state_predicates = conditional_sam._create_complete_world_state(
        observed_objects, next_state)
    conditional_sam._initialize_action_effects(grounded_action, positive_previous_state_predicates,
                                               negative_previous_state_predicates)
    conditional_sam._update_action_effects(grounded_action,
                                           list(positive_next_state_predicates),
                                           list(negative_next_state_predicates))
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].add_effects) > 0
    assert len(conditional_sam.partial_domain.actions[grounded_action.name].delete_effects) == 0

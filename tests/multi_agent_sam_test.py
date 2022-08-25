"""Module test for the multi agent action model learning."""
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, MultiAgentObservation, ActionCall, MultiAgentComponent
from pytest import fixture

from sam_learning.learners import MultiAgentSAM
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH, WOODWORKING_COMBINED_PROBLEM_PATH, \
    WOODWORKING_COMBINED_TRAJECTORY_PATH

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]


@fixture()
def combined_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def combined_problem(combined_domain: Domain) -> Problem:
    return ProblemParser(problem_path=WOODWORKING_COMBINED_PROBLEM_PATH, domain=combined_domain).parse_problem()


@fixture()
def multi_agent_observation(combined_domain: Domain, combined_problem: Problem) -> MultiAgentObservation:
    return TrajectoryParser(combined_domain, combined_problem).parse_trajectory(
        WOODWORKING_COMBINED_TRAJECTORY_PATH, executing_agents=WOODWORKING_AGENT_NAMES)


@fixture()
def ma_sam(combined_domain: Domain) -> MultiAgentSAM:
    return MultiAgentSAM(combined_domain)


@fixture()
def do_plane_observation_component(multi_agent_observation: MultiAgentObservation) -> MultiAgentComponent:
    return multi_agent_observation.components[1]


@fixture()
def do_plane_first_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer0", "p2", "verysmooth", "natural", "varnished"])


@fixture()
def do_plane_second_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer0", "p1", "rough", "natural", "untreated"])


def test_add_new_single_agent_action_adds_new_predicates_and_learns_the_preconditions_correctly(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation, do_plane_first_action_call: ActionCall):
    component = multi_agent_observation.components[1]
    # (do-plane planer0 p2 verysmooth natural varnished)
    previous_state = component.previous_state
    next_state = component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2

    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)
    assert "do-plane" in ma_sam.partial_domain.actions
    learned_action = ma_sam.partial_domain.actions["do-plane"]
    positive_preconditions = [p.untyped_representation for p in learned_action.positive_preconditions]
    assert {"(treatment ?x ?oldtreatment)", "(surface-condition ?x ?oldsurface)", "(available ?x)",
            "(colour ?x ?oldcolour)"}.issubset(positive_preconditions)


def test_add_new_single_agent_action_adds_new_predicates_and_learns_the_add_effects_correctly(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation, do_plane_first_action_call: ActionCall):
    component = multi_agent_observation.components[1]
    # (do-plane planer0 p2 verysmooth natural varnished)
    previous_state = component.previous_state
    next_state = component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2

    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)
    assert "do-plane" in ma_sam.partial_domain.actions
    add_effects = [p.untyped_representation for p in ma_sam.might_be_add_effects["do-plane"]]
    assert {"(surface-condition ?x smooth)", "(colour ?x natural)", "(treatment ?x untreated)"}.issuperset(add_effects)


def test_add_new_single_agent_action_adds_new_predicates_and_learns_the_delete_effects_correctly(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation,
        do_plane_first_action_call: ActionCall):
    component = multi_agent_observation.components[1]
    previous_state = component.previous_state
    next_state = component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2

    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)
    assert "do-plane" in ma_sam.partial_domain.actions
    delete_effects = [p.untyped_representation for p in ma_sam.might_be_delete_effects["do-plane"]]
    assert {"(treatment ?x ?oldtreatment)", "(surface-condition ?x ?oldsurface)", "(colour ?x ?oldcolour)"}.issuperset(
        delete_effects)


def test_update_single_agent_action_updates_action_with_correct_preconditions(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation,
        do_plane_first_action_call: ActionCall, do_plane_observation_component: MultiAgentComponent,
        do_plane_second_action_call: ActionCall):
    previous_state = do_plane_observation_component.previous_state
    next_state = do_plane_observation_component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2
    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)

    component = multi_agent_observation.components[2]
    previous_state = component.previous_state
    next_state = component.next_state
    ma_sam.update_single_agent_action(do_plane_second_action_call, previous_state, next_state, num_actions,
                                      observed_objects)
    learned_action = ma_sam.partial_domain.actions["do-plane"]
    positive_preconditions = [p.untyped_representation for p in learned_action.positive_preconditions]
    assert {"(treatment ?x ?oldtreatment)", "(surface-condition ?x ?oldsurface)", "(available ?x)",
            "(colour ?x ?oldcolour)"}.issubset(positive_preconditions)


def test_update_single_agent_action_updates_action_with_correct_add_effects(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation,
        do_plane_first_action_call: ActionCall, do_plane_observation_component: MultiAgentComponent,
        do_plane_second_action_call: ActionCall):
    previous_state = do_plane_observation_component.previous_state
    next_state = do_plane_observation_component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2
    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)

    component = multi_agent_observation.components[2]
    previous_state = component.previous_state
    next_state = component.next_state
    ma_sam.update_single_agent_action(do_plane_second_action_call, previous_state, next_state, num_actions,
                                      observed_objects)
    add_effects = [p.untyped_representation for p in ma_sam.might_be_add_effects["do-plane"]]
    assert {"(surface-condition ?x smooth)", "(colour ?x natural)", "(treatment ?x untreated)"}.issuperset(add_effects)


def test_update_single_agent_action_updates_action_with_correct_delete_effects(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation,
        do_plane_first_action_call: ActionCall, do_plane_observation_component: MultiAgentComponent,
        do_plane_second_action_call: ActionCall):
    previous_state = do_plane_observation_component.previous_state
    next_state = do_plane_observation_component.next_state
    observed_objects = multi_agent_observation.grounded_objects
    num_actions = 2
    ma_sam.add_new_single_agent_action(do_plane_first_action_call, previous_state, next_state, observed_objects,
                                       num_actions)

    component = multi_agent_observation.components[2]
    previous_state = component.previous_state
    next_state = component.next_state
    ma_sam.update_single_agent_action(do_plane_second_action_call, previous_state, next_state, num_actions,
                                      observed_objects)
    delete_effects = [p.untyped_representation for p in ma_sam.might_be_delete_effects["do-plane"]]
    assert {"(treatment ?x ?oldtreatment)", "(surface-condition ?x ?oldsurface)", "(colour ?x ?oldcolour)"}.issuperset(
        delete_effects)


def test_learn_action_model_returns_learned_model(
        ma_sam: MultiAgentSAM, multi_agent_observation: MultiAgentObservation):
    learned_model, learning_report = ma_sam.learn_combined_action_model([multi_agent_observation],
                                                                        agent_names=WOODWORKING_AGENT_NAMES)
    print(learning_report)
    print(learned_model.to_pddl())

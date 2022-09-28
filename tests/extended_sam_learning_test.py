"""module tests for the SAM learning algorithm"""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, ActionCall, Problem, Observation, \
    GroundedPredicate
from pytest import fixture

from sam_learning.learners import ExtendedSAM
from tests.consts import WOODWORKING_DOMAIN_PATH, WOODWORKING_PROBLEM_PATH, WOODWORKING_TRAJECTORY_PATH


@fixture()
def woodworking_domain() -> Domain:
    domain_parser = DomainParser(WOODWORKING_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def woodworking_complete_domain() -> Domain:
    domain_parser = DomainParser(WOODWORKING_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def woodworking_problem(woodworking_domain: Domain) -> Problem:
    return ProblemParser(problem_path=WOODWORKING_PROBLEM_PATH, domain=woodworking_domain).parse_problem()


@fixture()
def woodworking_observation(woodworking_domain: Domain, woodworking_problem: Problem) -> Observation:
    return TrajectoryParser(woodworking_domain, woodworking_problem).parse_trajectory(WOODWORKING_TRAJECTORY_PATH)


@fixture()
def esam_learning(woodworking_domain: Domain) -> ExtendedSAM:
    return ExtendedSAM(woodworking_domain)


def test_extract_maybe_delete_effects_when_no_matches_does_not_change_delete_effect(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_del_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"}),
                                 GroundedPredicate(name="surface-condition",
                                                   signature=woodworking_domain.predicates[
                                                       "surface-condition"].signature,
                                                   object_mapping={"?obj": "o1", "?surface": "rough"})}
    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o2", "red", "smooth"])
    must_be_delete_effects = []
    esam_learning._extract_maybe_delete_effects(test_action_call, test_grounded_del_effects, must_be_delete_effects)
    assert len(must_be_delete_effects) == 0


def test_extract_maybe_delete_effects_when_only_one_possible_match_add_to_must_be_delete_effects(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_del_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"})}

    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o1", "red", "smooth"])
    must_be_delete_effects = []
    esam_learning._extract_maybe_delete_effects(test_action_call, test_grounded_del_effects, must_be_delete_effects)
    assert len(must_be_delete_effects) == 1


def test_extract_maybe_delete_effects_when_more_than_one_possible_match_add_to_maybe_delete_effects(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_del_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"})}

    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o1", "natural", "smooth"])
    must_be_delete_effects = []
    esam_learning._extract_maybe_delete_effects(test_action_call, test_grounded_del_effects, must_be_delete_effects)
    assert len(must_be_delete_effects) == 0
    assert len(esam_learning.possible_delete_effects["do-immersion-varnish"]) == 2


def test_extract_maybe_add_effects_when_no_matches_does_not_change_delete_effect(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_edd_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"}),
                                 GroundedPredicate(name="surface-condition",
                                                   signature=woodworking_domain.predicates[
                                                       "surface-condition"].signature,
                                                   object_mapping={"?obj": "o1", "?surface": "rough"})}
    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o2", "red", "smooth"])
    must_be_add_effects = []
    esam_learning._extract_maybe_add_effects(test_action_call, test_grounded_edd_effects, must_be_add_effects)
    assert len(must_be_add_effects) == 0


def test_extract_maybe_add_effects_when_only_one_possible_match_add_to_must_be_delete_effects(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_add_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"})}

    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o1", "red", "smooth"])
    must_be_add_effects = []
    esam_learning._extract_maybe_add_effects(test_action_call, test_grounded_add_effects, must_be_add_effects)
    assert len(must_be_add_effects) == 1


def test_extract_maybe_add_effects_when_more_than_one_possible_match_add_to_maybe_delete_effects(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_add_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"})}

    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o1", "natural", "smooth"])
    must_be_add_effects = []
    esam_learning._extract_maybe_add_effects(test_action_call, test_grounded_add_effects, must_be_add_effects)
    assert len(must_be_add_effects) == 0
    assert len(esam_learning.possible_add_effects["do-immersion-varnish"]) == 2


def test_remove_impossible_effects_removes_an_effect_once_it_has_not_been_observed_in_a_state(
        esam_learning: ExtendedSAM, woodworking_domain: Domain):
    test_grounded_del_effects = {GroundedPredicate(name="colour",
                                                   signature=woodworking_domain.predicates["colour"].signature,
                                                   object_mapping={"?obj": "o1", "?colour": "natural"})}

    test_action_call = ActionCall(name="do-immersion-varnish",
                                  grounded_parameters=["immersion-varnisher0", "o1", "natural", "smooth"])
    must_be_delete_effects = []
    esam_learning._extract_maybe_delete_effects(test_action_call, test_grounded_del_effects, must_be_delete_effects)
    assert len(esam_learning.possible_delete_effects["do-immersion-varnish"]) == 2
    test_new_action_call = ActionCall(name="do-immersion-varnish",
                                      grounded_parameters=["immersion-varnisher0", "o1", "red", "smooth"])
    state_predicates = {"(colour ?obj ?colour)": {
        GroundedPredicate(name="colour",
                          signature=woodworking_domain.predicates[
                              "colour"].signature,
                          object_mapping={"?obj": "o1", "?colour": "red"})}}
    esam_learning._remove_impossible_effects(test_new_action_call, state_predicates)
    assert len(esam_learning.possible_delete_effects["do-immersion-varnish"]) == 1
    effect = esam_learning.possible_delete_effects["do-immersion-varnish"].pop()
    assert effect.untyped_representation == "(colour ?x natural)"


def test_handle_action_effects_learns_correct_options_for_must_be_effects(esam_learning: ExtendedSAM,
                                                                          woodworking_complete_domain: Domain,
                                                                          woodworking_observation: Observation):
    previous_state = woodworking_observation.components[0].previous_state
    next_state = woodworking_observation.components[0].next_state
    action_call = woodworking_observation.components[0].grounded_action_call
    must_be_add_effects, must_be_delete_effects = esam_learning._handle_action_effects(
        action_call, previous_state, next_state)
    domain_action = woodworking_complete_domain.actions[action_call.name]
    assert domain_action.add_effects.issuperset(must_be_add_effects)
    assert domain_action.delete_effects.issuperset(must_be_delete_effects)


def test_learn_action_model_returns_learned_model(esam_learning: ExtendedSAM, woodworking_observation: Observation):
    learned_model, learning_report = esam_learning.learn_action_model([woodworking_observation])
    print(learning_report)
    print(learned_model.to_pddl())

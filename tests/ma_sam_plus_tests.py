"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent, \
    GroundedPredicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MASAMPlus
from tests.consts import sync_ma_snapshot

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam_plus(woodworking_ma_combined_domain: Domain) -> MASAMPlus:
    return MASAMPlus(woodworking_ma_combined_domain)


@fixture()
def rovers_ma_sam(ma_rovers_domain) -> MASAMPlus:
    return MASAMPlus(ma_rovers_domain)


@fixture()
def do_plane_observation_component(multi_agent_observation: MultiAgentObservation) -> MultiAgentComponent:
    return multi_agent_observation.components[1]


@fixture()
def do_plane_first_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer0", "p2", "verysmooth", "natural", "varnished"])


@fixture()
def do_plane_second_action_call() -> ActionCall:
    return ActionCall("do-plane", ["planer1", "p2", "verysmooth", "natural", "untreated"])


@fixture()
def communicate_image_data_action_call() -> ActionCall:
    return ActionCall("communicate_image_data", ["rover0", "lander0", "objective4", "colour", "waypoint0", "waypoint1"])


@fixture()
def woodworking_literals_cnf(woodworking_ma_combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in woodworking_ma_combined_domain.actions.keys()]
    return LiteralCNF(action_names)


@fixture()
def rovers_literals_cnf(ma_rovers_domain: Domain) -> LiteralCNF:
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    return LiteralCNF(action_names)


def test_play(woodworking_ma_sam_plus:MASAMPlus, multi_agent_observation: MultiAgentObservation):
    learned_model, _ = woodworking_ma_sam_plus.learn_combined_action_model([multi_agent_observation])

def test_construct_safe_actions_returns_empty_list_if_no_action_is_safe(
        woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf
    possible_effects = [("do-grind", "(surface-condition ?m ?oldcolour)"),
                        ("do-plane", "(surface-condition ?m ?colour)")]
    woodworking_literals_cnf.add_possible_effect(possible_effects)
    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_ma_sam_plus.observed_actions.append("do-plane")
    woodworking_ma_sam_plus.observed_actions.append("do-immersion-varnish")
    woodworking_ma_sam_plus.observed_actions.append("do-grind")

    woodworking_ma_sam_plus.construct_safe_actions()
    woodworking_ma_sam_plus.construct_macro_actions()
    assert "do-plane" not in woodworking_ma_sam_plus.safe_actions
    assert "do-grind" not in woodworking_ma_sam_plus.safe_actions
    assert "do-immersion-varnish" in woodworking_ma_sam_plus.safe_actions

def test_learn_action_model_with_colliding_actions_returns_that_actions_are_unsafe(
        rovers_ma_sam: MASAMPlus, ma_rovers_observation):
    learned_domain, learning_report = rovers_ma_sam.learn_combined_action_model([ma_rovers_observation])
    assert learning_report["navigate"] == "NOT SAFE"
    assert learning_report["communicate_rock_data"] == "NOT SAFE"
    assert learning_report["navigate_communicate_rock_data"] == "OK"
    print(learned_domain.to_pddl())
"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent, \
    GroundedPredicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MASAMPlus, MacroActionParser
from tests.consts import sync_ma_snapshot

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam_plus(woodworking_ma_combined_domain: Domain) -> MASAMPlus:
    return MASAMPlus(woodworking_ma_combined_domain)


@fixture()
def rovers_ma_sam(ma_rovers_domain) -> MASAMPlus:
    return MASAMPlus(ma_rovers_domain,
                     ["rover0", "rover1", "rover2", "rover3", "rover4",
                      "rover5", "rover6", "rover7", "rover8", "rover9"])

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


def test_generate_macro_action_name(
        rovers_ma_sam: MASAMPlus,
        woodworking_ma_combined_domain: Domain
):
    actions = ["way-forward", "way-backward", "way"]
    macro_name = MacroActionParser.generate_macro_action_name(actions)
    assert macro_name == "way-forward-way-backward-way"


def test_generate_macro_mappings_without_grouping(
        rovers_ma_sam: MASAMPlus,
        woodworking_ma_combined_domain: Domain):
    actions = {rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]}
    binding = MacroActionParser.generate_macro_mappings([], actions)

    assert (len(set(binding.keys())) == len(rovers_ma_sam.partial_domain.actions["drop"].parameter_names) +
            len(rovers_ma_sam.partial_domain.actions["sample_rock"].parameter_names) == len(set(binding.values())))


def test_generate_macro_mappings_with_grouping(
        rovers_ma_sam: MASAMPlus,
        woodworking_ma_combined_domain: Domain):
    actions = {rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]}
    grouping = [{("drop", "?x"), ("sample_rock", "?x")}]
    binding = MacroActionParser.generate_macro_mappings(grouping, actions)
    duplicates = 1
    assert (len(set(binding.keys())) == len(rovers_ma_sam.partial_domain.actions["drop"].parameter_names) +
            len(rovers_ma_sam.partial_domain.actions["sample_rock"].parameter_names) ==
            len(set(binding.values())) + duplicates)


def test_generate_macro_action_signature(
        rovers_ma_sam: MASAMPlus,
        woodworking_ma_combined_domain: Domain):

    actions = {rovers_ma_sam.partial_domain.actions["drop"],
               rovers_ma_sam.partial_domain.actions["sample_rock"]}

    bindings = {('drop', '?x'): '?x0', ('drop', '?y'): '?y0',
                ('sample_rock', '?p'): '?p1', ('sample_rock', '?s'): '?s1', ('sample_rock', '?x'): '?x1'}

    signature = MacroActionParser.generate_macro_action_signature(actions, bindings)
    action_list = [rovers_ma_sam.partial_domain.actions["drop"],
                   rovers_ma_sam.partial_domain.actions["sample_rock"]]

    for i, action in enumerate(action_list):
        for parameter in action.parameter_names:
            assert action.signature[parameter] == signature[parameter + str(i)]

    assert len(set(signature)) == len(set(bindings.values()))

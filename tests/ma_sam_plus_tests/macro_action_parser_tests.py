"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent, GroundedPredicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MASAMPlus
from utilities import MacroActionParser
from tests.consts import sync_ma_snapshot

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0", "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam_plus(woodworking_ma_combined_domain: Domain) -> MASAMPlus:
    return MASAMPlus(woodworking_ma_combined_domain)


@fixture()
def rovers_ma_sam(ma_rovers_domain) -> MASAMPlus:
    return MASAMPlus(ma_rovers_domain, ["rover0", "rover1", "rover2", "rover3", "rover4", "rover5", "rover6", "rover7", "rover8", "rover9"])


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


def test_generate_macro_action_name_returns_action_name_ordered_according_to_list():
    actions = ["way-forward", "way-backward", "way"]
    macro_name = MacroActionParser.generate_macro_action_name(actions, [])
    assert macro_name == "way-forward-way-backward-way"


def test_generate_macro_mappings_without_grouping_returns_mapping_where_no_parameters_share_macro_parameter(rovers_ma_sam: MASAMPlus):
    actions = {rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]}
    binding = MacroActionParser.generate_macro_mappings([], actions)
    drop_params_len = len(rovers_ma_sam.partial_domain.actions["drop"].parameter_names)
    sample_rock_params_len = len(rovers_ma_sam.partial_domain.actions["sample_rock"].parameter_names)

    assert len(set(binding.keys())) == len(set(binding.values()))
    assert (drop_params_len + sample_rock_params_len) == len(set(binding.values()))


def test_generate_macro_mappings_with_grouping_returns_mapping_where_parameters_share_macro_parameter(rovers_ma_sam: MASAMPlus,):
    actions = {rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]}
    grouping = [{("drop", "?x"), ("sample_rock", "?x")}]
    binding = MacroActionParser.generate_macro_mappings(grouping, actions)
    shared_parameters = 1

    drop_params_len = len(rovers_ma_sam.partial_domain.actions["drop"].parameter_names)
    sample_rock_params_len = len(rovers_ma_sam.partial_domain.actions["sample_rock"].parameter_names)

    assert len(set(binding.keys())) == (len(set(binding.values())) + shared_parameters)
    assert (drop_params_len + sample_rock_params_len) == len(set(binding.keys()))


def test_generate_macro_action_signature_returns_macro_signature_with_correct_types(rovers_ma_sam: MASAMPlus, woodworking_ma_combined_domain: Domain):

    actions = {rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]}

    bindings = {
        ("drop", "?x"): "?x0",
        ("drop", "?y"): "?y0",
        ("sample_rock", "?p"): "?p1",
        ("sample_rock", "?s"): "?s1",
        ("sample_rock", "?x"): "?x1",
    }

    signature = MacroActionParser.generate_macro_action_signature(actions, bindings)
    action_list = [rovers_ma_sam.partial_domain.actions["drop"], rovers_ma_sam.partial_domain.actions["sample_rock"]]

    for i, action in enumerate(action_list):
        for parameter in action.parameter_names:
            assert action.signature[parameter] == signature[parameter + str(i)]

    assert len(set(signature)) == len(set(bindings.values()))


def test_extract_actions_from_macro_action_from_line_with_macro_returns_several_micro_actions_lines(
    rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation
):
    rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation])
    macro_name = list(rovers_ma_sam_plus.mapping.keys())[0]
    action_line = f"({macro_name} a b c d e f h)"
    new_action_lines = MacroActionParser.extract_actions_from_macro_action(action_line, rovers_ma_sam_plus.mapping)
    print(new_action_lines)
    print(rovers_ma_sam_plus.mapping)
    assert {"(communicate_rock_data a b c d e)", "(navigate f c h)"} == new_action_lines


def test_extract_actions_from_macro_actions_from_line_without_macro_returns_same_line(rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation):
    rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation])
    action_line = f"(navigate a b c d e f h)"
    new_action_lines = MacroActionParser.extract_actions_from_macro_action(action_line, rovers_ma_sam_plus.mapping)
    print(new_action_lines)
    assert {action_line} == new_action_lines

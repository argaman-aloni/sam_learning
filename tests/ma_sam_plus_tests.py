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


def test_learn_action_model_with_colliding_actions_returns_model_with_macro_actions(
        rovers_ma_sam: MASAMPlus, ma_rovers_observation):
    learned_domain, learning_report = rovers_ma_sam.learn_combined_action_model([ma_rovers_observation])
    print(learned_domain.to_pddl())


def test_extract_relevant_lmas_with_colliding_actions_returns_correct_lmas(
     woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-plane", "(surface-condition ?m ?colour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?colour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])

    lmas = woodworking_ma_sam_plus.extract_relevant_lmas()
    lmas_names = list(map(lambda x: sorted(list(map(lambda u: u.name, x))), lmas))
    assert len(lmas_names) == 4
    assert sorted(["do-immersion-varnish", "do-grind"]) in lmas_names
    assert sorted(["do-grind", "do-plane"]) in lmas_names
    assert sorted(["do-plane", "do-immersion-varnish"]) in lmas_names
    assert sorted(["do-grind", "do-plane", "do-immersion-varnish"]) in lmas_names


def test_extract_relevant_binding_with_colliding_actions_returns_correct_binding(
     woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?newcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?oldcolour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?oldcolour)")])

    lma = {woodworking_ma_sam_plus.partial_domain.actions["do-immersion-varnish"],
           woodworking_ma_sam_plus.partial_domain.actions["do-grind"],
           woodworking_ma_sam_plus.partial_domain.actions["do-plane"]}

    amount_of_params = (len(woodworking_ma_sam_plus.partial_domain.actions["do-immersion-varnish"].parameter_names)
                        + len(woodworking_ma_sam_plus.partial_domain.actions["do-grind"].parameter_names) +
                        len(woodworking_ma_sam_plus.partial_domain.actions["do-plane"].parameter_names))

    bindings = woodworking_ma_sam_plus.generate_possible_bindings(lma)
    binds = bindings[0]
    param_names_set = set(binds.values())

    amount_of_duplicates = 4

    assert len(param_names_set) == len(binds.values()) - amount_of_duplicates
    assert binds[('do-immersion-varnish', '?m')] == binds[('do-grind', '?m')] == binds[('do-plane', '?m')]
    assert binds[('do-immersion-varnish', '?newcolour')] == binds[('do-grind', '?oldcolour')] == binds[('do-plane', '?oldcolour')]
    assert amount_of_params == len(binds)


def test_extract_lma_predicates_from_cnf_returns_correct_effects_and_preconditions(
     woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain):
    action_names = [action for action in woodworking_ma_combined_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    cnf2 = LiteralCNF(action_names)

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = cnf1
    woodworking_ma_sam_plus.literals_cnf["(surface-condition2 ?obj ?surface)"] = cnf2

    cnf1.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    cnf1.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                              ("do-immersion-varnish", "(surface-condition ?m ?newcolour)"),
                              ("do-plane", "(surface-condition ?m ?oldcolour)")])
    cnf1.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                              ("do-plane", "(surface-condition ?m ?oldcolour)")])

    cnf2.add_possible_effect([("do-immersion-varnish", "(surface-condition2 ?m ?newcolour)")])
    cnf2.add_possible_effect([("do-grind", "(surface-condition2 ?m ?oldcolour)"),
                              ("do-immersion-varnish", "(surface-condition2 ?m ?newcolour)"),
                              ("do-plane", "(surface-condition2 ?m ?oldcolour)")])
    cnf2.add_possible_effect([("do-grind", "(surface-condition2 ?m ?newtreatment)"),
                              ("do-plane", "(surface-condition2 ?m ?oldsurface)")])

    lma = {woodworking_ma_sam_plus.partial_domain.actions["do-grind"],
           woodworking_ma_sam_plus.partial_domain.actions["do-plane"]}

    binding = {('do-grind', '?m'): '?mm', ('do-grind', '?x'): '?x0', ('do-grind', '?oldsurface'): '?oldsurface0',
               ('do-grind', '?oldcolour'): '?oldcolouroldcolour', ('do-grind', '?oldtreatment'): '?oldtreatment0',
               ('do-grind', '?newtreatment'): '?newtreatment0', ('do-plane', '?m'): '?mm', ('do-plane', '?x'): '?x1',
               ('do-plane', '?oldsurface'): '?oldsurface1', ('do-plane', '?oldcolour'): '?oldcolouroldcolour',
               ('do-plane', '?oldtreatment'): '?oldtreatment1'}

    effects, preconditions = woodworking_ma_sam_plus.extract_lma_predicates_from_cnf(lma, binding)

    assert list(map(lambda x: x.untyped_representation, effects)) == ['(surface-condition ?mm ?oldcolouroldcolour)']
    assert len(preconditions) == 6




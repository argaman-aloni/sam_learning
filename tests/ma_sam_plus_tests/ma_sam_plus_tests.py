"""Module test for the multi-agent action model learning."""
from pddl_plus_parser.models import Domain, MultiAgentObservation, ActionCall, MultiAgentComponent, \
    GroundedPredicate
from pytest import fixture

from sam_learning.core import LiteralCNF
from sam_learning.learners import MASAMPlus, group_params_from_clause
from tests.consts import sync_ma_snapshot

WOODWORKING_AGENT_NAMES = ["glazer0", "grinder0", "highspeed-saw0", "immersion-varnisher0", "planer0", "saw0",
                           "spray-varnisher0"]
ROVERS_AGENT_NAMES = [f"rovers{i}" for i in range(10)]


@fixture()
def woodworking_ma_sam_plus(woodworking_ma_combined_domain: Domain) -> MASAMPlus:
    return MASAMPlus(woodworking_ma_combined_domain)


@fixture()
def rovers_ma_sam_plus(ma_rovers_domain) -> MASAMPlus:
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


@fixture()
def driverlog_ma_sam_plus(ma_driverlog_domain) -> MASAMPlus:
    return MASAMPlus(ma_driverlog_domain)

def test_learn_action_model_with_colliding_actions_returns_model_with_macro_actions(
        rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation):
    learned_domain, learning_report = (
        rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation]))
    print(learned_domain.to_pddl())


def test_learn_action_model_with_colliding_actions_returns_model_with_macro_actions_driverlog(
        driverlog_ma_sam_plus: MASAMPlus, ma_driverlog_observation):
    learned_domain, learning_report = (
        driverlog_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_driverlog_observation]))
    print(learned_domain.to_pddl())


def test_extract_relevant_action_groups_with_no_observed_actions_returns_no_action_group(
     woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])

    action_group = woodworking_ma_sam_plus.extract_relevant_action_groups()
    assert len(action_group) == 0


def test_extract_relevant_action_groups_with_observed_actions_with_no_colliding_actions_returns_no_action_group(
     woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf

    woodworking_literals_cnf.add_possible_effect([("do-immersion-varnish", "(surface-condition ?m ?newcolour)")])
    woodworking_ma_sam_plus.observed_actions = ["do-immersion-varnish", "do-grind", "do-plane"]

    action_group = woodworking_ma_sam_plus.extract_relevant_action_groups()
    assert len(action_group) == 0


def test_extract_relevant_action_groups_with_observed_actions_with_colliding_actions_returns_action_group(
        woodworking_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        woodworking_ma_combined_domain: Domain,
        woodworking_literals_cnf):

    woodworking_ma_sam_plus.literals_cnf["(surface-condition ?obj ?surface)"] = woodworking_literals_cnf
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-plane", "(surface-condition ?m ?colour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])
    woodworking_literals_cnf.add_possible_effect([("do-grind", "(surface-condition ?m ?oldcolour)"),
                                                  ("do-plane", "(surface-condition ?m ?colour)"),
                                                  ("do-immersion-varnish", "(surface-condition ?m ?colour)")])

    woodworking_ma_sam_plus.observed_actions = ["do-grind", "do-plane", "do-immersion-varnish"]

    action_groups = woodworking_ma_sam_plus.extract_relevant_action_groups()
    action_groups_names = list(map(lambda x: sorted(list(map(lambda u: u.name, x))), action_groups))
    assert len(action_groups_names) == 4
    assert sorted(["do-immersion-varnish", "do-grind"]) in action_groups_names
    assert sorted(["do-grind", "do-plane"]) in action_groups_names
    assert sorted(["do-plane", "do-immersion-varnish"]) in action_groups_names
    assert sorted(["do-grind", "do-plane", "do-immersion-varnish"]) in action_groups_names


def test_extract_relevant_parameter_groupings_with_valid_action_group_returns_informative_parameter_groupings(
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

    action_group_names = ["do-immersion-varnish", "do-grind", "do-plane"]

    parameter_grouping = woodworking_ma_sam_plus.extract_relevant_parameter_groupings(action_group_names)[0]
    real_parameter_grouping = [{('do-immersion-varnish', '?m'), ('do-grind', '?m'), ('do-plane', '?m')},
                    {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour'), ('do-plane', '?oldcolour')}]

    assert parameter_grouping == real_parameter_grouping


def test_extract_relevant_parameter_groupings_with_no_valid_action_group_returns_uninformative_parameter_groupings(
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

    action_group_names = ["do-glaze", "do-grind"]
    parameter_grouping = woodworking_ma_sam_plus.extract_relevant_parameter_groupings(action_group_names)[0]

    assert len(parameter_grouping) == 0


def test_group_params_from_non_unit_clause():
    clause = [("do-grind", "(m ?x ?y)"),
              ("do-plane", "(m ?x ?z)")]

    group = group_params_from_clause(clause)
    real_group = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]

    assert group == real_group


def test_group_params_from_unit_clause():
    clause = [("do-grind", "(m ?x ?y)"),]

    group = group_params_from_clause(clause)

    assert group == [{('do-grind', '?x')}, {('do-grind', '?y')}]


def test_extract_effects_for_macro_from_cnf(
        rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        ma_rovers_domain: Domain):
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    cnf2 = LiteralCNF(action_names)
    cnf3 = LiteralCNF(action_names)
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = cnf1
    rovers_ma_sam_plus.literals_cnf["fluent_test2 ?w"] = cnf2
    rovers_ma_sam_plus.literals_cnf["fluent_test3 ?x"] = cnf3

    cnf1.add_possible_effect([('communicate_rock_data', '(fluent_test1 ?p)'),
                              ('navigate', '(fluent_test1 ?y)')])
    cnf2.add_possible_effect([('communicate_rock_data', '(fluent_test2 ?p)'),
                              ('navigate', '(fluent_test2 ?y)')])
    cnf2.add_possible_effect([('communicate_rock_data', '(fluent_test3 ?p)'),
                              ('navigate', '(fluent_test3 ?x)')])

    binding = [{('communicate_rock_data', '?p'), ('navigate', '?y')}]
    lma_names = ["navigate", "communicate_rock_data"]
    action_group = [action for action in rovers_ma_sam_plus.partial_domain.actions.values() if action.name in lma_names]

    mapping = {('navigate', '?x'): "?x'0", ('navigate', '?y'): '?yp', ('navigate', '?z'): "?z'0",
               ('communicate_rock_data', '?r'): "?r'1", ('communicate_rock_data', '?l'): "?l'1",
               ('communicate_rock_data', '?p'): '?yp', ('communicate_rock_data', '?x'): "?x'1",
               ('communicate_rock_data', '?y'): "?y'1"}

    effects = rovers_ma_sam_plus.extract_effects_for_macro_from_cnf(action_group, binding, mapping)

    effects_rep = [effect.untyped_representation for effect in effects]
    assert "(fluent_test1 ?yp)" in effects_rep
    assert "(fluent_test2 ?yp)" in effects_rep
    assert len(effects_rep) == 2


def test_extract_preconditions_for_macro_from_cnf(
        rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        ma_rovers_domain: Domain):
    action_names = [action for action in ma_rovers_domain.actions.keys()]
    cnf1 = LiteralCNF(action_names)
    cnf2 = LiteralCNF(action_names)
    cnf3 = LiteralCNF(action_names)
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = cnf1
    rovers_ma_sam_plus.literals_cnf["fluent_test2 ?w"] = cnf2
    rovers_ma_sam_plus.literals_cnf["fluent_test3 ?x"] = cnf3

    cnf1.add_possible_effect([('communicate_rock_data', '(fluent_test1 ?p)'),
                              ('navigate', '(fluent_test1 ?y)')])
    cnf2.add_possible_effect([('communicate_rock_data', '(fluent_test2 ?p)'),
                              ('navigate', '(fluent_test2 ?y)')])
    cnf2.add_possible_effect([('communicate_rock_data', '(fluent_test3 ?p)'),
                              ('navigate', '(fluent_test3 ?x)')])

    parameter_grouping = [{('communicate_rock_data', '?p'), ('navigate', '?y')}]
    action_group = {rovers_ma_sam_plus.partial_domain.actions["navigate"],
           rovers_ma_sam_plus.partial_domain.actions["communicate_rock_data"]}

    mapping = {('navigate', '?x'): "?x'0", ('navigate', '?y'): '?yp', ('navigate', '?z'): "?z'0",
               ('communicate_rock_data', '?r'): "?r'1", ('communicate_rock_data', '?l'): "?l'1",
               ('communicate_rock_data', '?p'): '?yp', ('communicate_rock_data', '?x'): "?x'1",
               ('communicate_rock_data', '?y'): "?y'1"}

    precondition = rovers_ma_sam_plus.extract_preconditions_for_macro_from_cnf(action_group, parameter_grouping, mapping)

    precondition_rep = [precondition.untyped_representation for precondition in precondition.root.operands]
    assert len(precondition_rep) == 2
    assert "(fluent_test3 ?x'0)" in precondition_rep
    assert "(fluent_test3 ?yp)" in precondition_rep


def test_extract_preconditions_for_macro_from_cnf_of_non_consistent_clause_and_contains_grouped_params_returns_one_prec(
        rovers_ma_sam_plus: MASAMPlus, do_plane_first_action_call: ActionCall,
        ma_rovers_domain: Domain, rovers_literals_cnf):
    rovers_ma_sam_plus.literals_cnf["fluent_test1 ?w"] = rovers_literals_cnf

    rovers_literals_cnf.add_possible_effect([('communicate_rock_data', '(fluent_test1 ?p)'),
                                             ('navigate', '(fluent_test1 ?y)'),
                                             ('sample_rock', '(fluent_test1 ?x)')])

    parameter_grouping = [{('communicate_rock_data', '?p'), ('navigate', '?y')}]
    action_group = {rovers_ma_sam_plus.partial_domain.actions["navigate"],
           rovers_ma_sam_plus.partial_domain.actions["communicate_rock_data"]}

    mapping = {('navigate', '?x'): "?x'0", ('navigate', '?y'): '?yp', ('navigate', '?z'): "?z'0",
               ('communicate_rock_data', '?r'): "?r'1", ('communicate_rock_data', '?l'): "?l'1",
               ('communicate_rock_data', '?p'): '?yp', ('communicate_rock_data', '?x'): "?x'1",
               ('communicate_rock_data', '?y'): "?y'1", ('sample_rock', '?x'): "?x'2"}

    precondition = rovers_ma_sam_plus.extract_preconditions_for_macro_from_cnf(action_group, parameter_grouping, mapping)

    precondition_rep = [precondition.untyped_representation for precondition in precondition.root.operands]
    assert len(precondition_rep) == 1
    assert "(fluent_test1 ?yp)" in precondition_rep


def test_extract_actions_from_macro_action(
        rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation):

    rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation])
    macro_name = list(rovers_ma_sam_plus.mapping.keys())[0]
    action_line = f"({macro_name} a b c d e f h)"
    new_action_lines = rovers_ma_sam_plus.extract_actions_from_macro_action(action_line)
    print(new_action_lines)
    assert ({'(communicate_rock_data d e b f h)', '(navigate a b c)'} == new_action_lines
            or {'(communicate_rock_data a b c d e)', '(navigate f c h)'} == new_action_lines)


def test_extract_actions_from_macro_actions_from_line_without_macro(
        rovers_ma_sam_plus: MASAMPlus, ma_rovers_observation):

    rovers_ma_sam_plus.learn_combined_action_model_with_macro_actions([ma_rovers_observation])
    macro_name = list(rovers_ma_sam_plus.mapping.keys())[0]
    action_line = f"(navigate a b c d e f h)"
    new_action_lines = rovers_ma_sam_plus.extract_actions_from_macro_action(action_line)
    print(new_action_lines)
    assert {'(navigate a b c d e f h)'} == new_action_lines



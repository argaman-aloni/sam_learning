"""Test the macro functionality of the MA literals CNF class."""
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain
from pytest import fixture

from sam_learning.core import LiteralCNF, is_clause_consistent
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH


@fixture()
def combined_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def literals_cnf(combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in combined_domain.actions.keys()]
    return LiteralCNF(action_names)


def test_extract_macro_action_effects_with_no_preconditions_returns_effects_that_are_not_preconditions(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    action_names = ["do-immersion-varnish", "do-grind"]
    fluent1 = "(has-colour ?agent ?newcolour)"
    fluent2 = "(has-colour ?agent ?oldcolour)"
    grouping = [{('do-immersion-varnish', '?agent'), ('do-grind', '?agent')},
                {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour')}]
    literals_cnf.add_possible_effect([("do-immersion-varnish", fluent1),
                                      ("do-grind", fluent2)])

    # No preconditions, expected to get all effects
    effects = literals_cnf.extract_macro_action_effects(action_names=action_names,
                                                        action_preconditions={},
                                                        param_grouping=grouping)
    assert (("do-grind", fluent2) in effects)
    assert (("do-immersion-varnish", fluent1) in effects)

    # One precondition, expected to not get this precondition
    effects = literals_cnf.extract_macro_action_effects(action_names=action_names,
                                                        action_preconditions={fluent1},
                                                        param_grouping=grouping)
    assert effects == [("do-grind", fluent2)]

    # Both fluents are precondition, expected to not get effects at all
    effects = literals_cnf.extract_macro_action_effects(action_names=action_names,
                                                        action_preconditions={fluent1, fluent2},
                                                        param_grouping=grouping)
    assert len(effects) == 0


def test_extract_macro_action_effects_extract_an_action_effect_from_unit_clause(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-immersion-varnish", "(has-colour ?agent ?newcolour)")])
    grouping = [{('do-immersion-varnish', '?agent'), ('do-grind', '?agent')},
                {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour')}]
    effects = literals_cnf.extract_macro_action_effects(["do-immersion-varnish", "do-grind"],
                                                        {},
                                                        param_grouping=grouping)
    assert effects == [("do-immersion-varnish", "(has-colour ?agent ?newcolour)")]


def test_extract_macro_action_precondition_extract_an_action_precondition_only_for_relevant_action_in_clause(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-plane", "(has-colour ?agent ?colour)"),
                                      ("do-grind", "(has-colour ?agent ?oldcolour)")])
    # inserting non-consistent grouping
    preconditions = literals_cnf.extract_macro_action_preconditions(["do-immersion-varnish", "do-grind"],
                                                                    param_grouping=[])
    assert preconditions == [("do-grind", "(has-colour ?agent ?oldcolour)")]


def test_extract_macro_action_precondition_extract_all_fluents_as_preconditions_when_param_grouping_does_not_encapsulate_fluents(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    clause = [("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
              ("do-grind", "(has-colour ?agent ?oldcolour)")]
    literals_cnf.add_possible_effect(clause)
    # inserting non-consistent grouping
    preconditions = literals_cnf.extract_macro_action_preconditions(action_names=["do-immersion-varnish", "do-grind"],
                                                                    param_grouping=[])
    assert len(preconditions) == 2
    assert clause[0] in preconditions
    assert clause[1] in preconditions


def test_is_clause_consistent_with_different_actions_return_false():
    clause = [("do-grind", "(m ?x ?y)"),
              ("do-plane", "(m ?x ?z)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-glaze"]

    assert not is_clause_consistent(clause=clause,
                                    macro_action_group_names=action_group_names,
                                    parameter_grouping_of_macro=parameter_grouping)


def test_is_clause_consistent_with_unit_clause_return_true():
    clause = [("do-grind", "(m ?x ?y)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-plane"]

    assert is_clause_consistent(clause=clause,
                                macro_action_group_names=action_group_names,
                                parameter_grouping_of_macro=parameter_grouping)


def test_is_clause_consistent_with_relevant_lma_and_binding_return_true():
    clause = [("do-grind", "(m ?x ?y)"),
              ("do-plane", "(m ?x ?z)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-plane"]

    assert is_clause_consistent(clause=clause,
                                macro_action_group_names=action_group_names,
                                parameter_grouping_of_macro=parameter_grouping)

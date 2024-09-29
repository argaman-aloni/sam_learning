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


def test_extract_macro_action_effects_extract_an_action_effect_only_if_not_from_preconditions(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                                      ("do-grind", "(has-colour ?agent ?oldcolour)")])
    grouping = [{('do-immersion-varnish', '?agent'), ('do-grind', '?agent')},
                {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour')}]
    effects = literals_cnf.extract_macro_action_effects(["do-immersion-varnish", "do-grind"],
                                                        {"(has-colour ?agent ?oldcolour)",
                                                         "(has-colour ?agent ?newcolour)"},
                                                        param_grouping=grouping)
    assert len(effects) == 0


def test_extract_macro_action_effects_extract_an_action_effect_with_one_precondition_returns_all_effects(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                                      ("do-grind", "(has-colour ?agent ?oldcolour)")])
    grouping = [{('do-immersion-varnish', '?agent'), ('do-grind', '?agent')},
                {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour')}]
    effects = literals_cnf.extract_macro_action_effects(["do-immersion-varnish", "do-grind"],
                                                        {"(has-colour ?agent ?oldcolour)"}, param_grouping=grouping)
    assert len(effects) == 1


def test_extract_macro_action_effects_extract_an_action_effect_from_unit_clause(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-immersion-varnish", "(has-colour ?agent ?newcolour)")])
    grouping = [{('do-immersion-varnish', '?agent'), ('do-grind', '?agent')},
                {('do-immersion-varnish', '?newcolour'), ('do-grind', '?oldcolour')}]
    effects = literals_cnf.extract_macro_action_effects(["do-immersion-varnish", "do-grind"],
                                                        {},
                                                        param_grouping=grouping)
    assert len(effects) == 1


def test_extract_macro_action_precondition_extract_an_action_precondition_only_for_relevant_action_in_clause(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-plane", "(has-colour ?agent ?colour)"),
                                      ("do-grind", "(has-colour ?agent ?oldcolour)")])
    # inserting non-consistent grouping
    preconditions = literals_cnf.extract_macro_action_preconditions(["do-immersion-varnish", "do-grind"],
                                                                    param_grouping=[])
    assert len(preconditions) == 1


def test_extract_macro_action_precondition_extract_all_action_preconditions(
        literals_cnf: LiteralCNF, combined_domain: Domain):
    literals_cnf.add_possible_effect([("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                                      ("do-grind", "(has-colour ?agent ?oldcolour)")])
    # inserting non-consistent grouping
    preconditions = literals_cnf.extract_macro_action_preconditions(["do-immersion-varnish", "do-grind"],
                                                              param_grouping=[])
    assert len(preconditions) == 2


def test_is_clause_consistent_with_different_actions_return_false():
    clause = [("do-grind", "(m ?x ?y)"),
              ("do-plane", "(m ?x ?z)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-glaze"]

    assert not is_clause_consistent(action_group_names=action_group_names,
                                    clause=clause, parameter_grouping=parameter_grouping)


def test_is_clause_consistent_with_unit_clause_return_true():
    clause = [("do-grind", "(m ?x ?y)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-plane"]

    assert is_clause_consistent(action_group_names=action_group_names,
                                clause=clause, parameter_grouping=parameter_grouping)


def test_is_clause_consistent_with_relevant_lma_and_binding_return_true():
    clause = [("do-grind", "(m ?x ?y)"),
              ("do-plane", "(m ?x ?z)")]
    parameter_grouping = [{('do-grind', '?x'), ('do-plane', '?x')}, {('do-grind', '?y'), ('do-plane', '?z')}]
    action_group_names = ["do-grind", "do-plane"]

    assert is_clause_consistent(action_group_names=action_group_names,
                                clause=clause, parameter_grouping=parameter_grouping)

"""Test the basic functionality of the MA literals CNF class."""
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain
from pytest import fixture

from sam_learning.core import LiteralCNF
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH


@fixture()
def combined_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def ma_literals_cnf(combined_domain: Domain) -> LiteralCNF:
    action_names = [action for action in combined_domain.actions.keys()]
    return LiteralCNF(action_names)


def test_add_not_effect_to_an_action_adds_the_predicate_to_the_not_effects_set(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    ma_literals_cnf.add_not_effect("do-plane", combined_domain.predicates["has-colour"])
    assert "(has-colour ?agent ?colour)" in ma_literals_cnf.not_effects["do-plane"]


def test_add_possible_effect_to_an_action_adds_the_predicate_to_the_possible_effects_set(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    assert len(ma_literals_cnf.possible_lifted_effects) == 1
    assert ma_literals_cnf.possible_lifted_effects[0] == possible_effects


def test_add_not_effect_to_an_action_removes_relevant_options_from_maybe_effects_in_the_cnf(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    ma_literals_cnf.add_not_effect("do-plane", combined_domain.predicates["has-colour"])
    assert len(ma_literals_cnf.possible_lifted_effects[0]) == 2
    assert ma_literals_cnf.possible_lifted_effects[0] == possible_effects[1:]


def test_add_possible_effect_does_not_add_effect_to_an_action_if_effect_declared_as_not_effect(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_not_effect("do-plane", combined_domain.predicates["has-colour"])
    ma_literals_cnf.add_possible_effect(possible_effects)
    assert len(ma_literals_cnf.possible_lifted_effects[0]) == 2
    assert ma_literals_cnf.possible_lifted_effects[0] == possible_effects[1:]


def test_is_action_safe_when_an_action_contains_ambiguity_returns_that_action_is_not_safe(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    assert not ma_literals_cnf.is_action_safe("do-plane", {"(has-colour ?agent ?colour)"})


def test_is_action_safe_when_an_actions_effect_declared_as_not_effect_considered_as_safe(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    ma_literals_cnf.add_not_effect("do-plane", combined_domain.predicates["has-colour"])
    assert ma_literals_cnf.is_action_safe("do-plane", {"(has-colour ?agent ?colour)"})


def test_is_action_acting_in_cnf_recognize_when_action_is_in_the_cnf(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-plane", "(has-colour ?agent ?colour)"),
                        ("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    assert ma_literals_cnf.is_action_acting_in_cnf("do-plane")


def test_is_action_acting_in_cnf_recognize_when_action_is_in_the_cnf_as_not_effect(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    ma_literals_cnf.add_not_effect("do-plane", combined_domain.predicates["has-colour"])
    assert ma_literals_cnf.is_action_acting_in_cnf("do-plane")


def test_is_action_acting_in_cnf_recognize_when_action_is_not_in_cnf(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    assert not ma_literals_cnf.is_action_acting_in_cnf("do-spray-varnish")


def test_extract_action_effects_extract_an_action_effect_only_from_unit_clause_cnfs(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    ma_literals_cnf.add_possible_effect([("do-plane", "(has-colour ?agent ?colour)")])
    effects = ma_literals_cnf.extract_action_effects("do-plane")
    assert len(effects) == 1


def test_extract_action_effects_extract_an_action_effect_from_non_unit_clause_returns_empty_list(
        ma_literals_cnf: LiteralCNF, combined_domain: Domain):
    possible_effects = [("do-immersion-varnish", "(has-colour ?agent ?newcolour)"),
                        ("do-grind", "(has-colour ?agent ?oldcolour)"),
                        ("do-plane", "(has-colour ?agent ?colour)")]
    ma_literals_cnf.add_possible_effect(possible_effects)
    effects = ma_literals_cnf.extract_action_effects("do-plane")
    assert len(effects) == 0

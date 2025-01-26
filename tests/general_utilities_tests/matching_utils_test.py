"""Module test for some of the matching utility functionality."""
from typing import Dict, Set

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, GroundedPredicate, State
from pytest import fixture

from sam_learning.core import extract_effects, extract_not_effects
from tests.consts import DEPOTS_NUMERIC_DOMAIN_PATH

TEST_LIFTED_SIGNATURE_ITEMS = ["?s", "?d", "?i", "?m"]
TEST_GROUNDED_ACTION_CALL = ["s1", "test_direction", "test_instrument", "test_mode"]


@fixture()
def domain() -> Domain:
    domain_parser = DomainParser(DEPOTS_NUMERIC_DOMAIN_PATH)
    return domain_parser.parse_domain()


@fixture()
def previous_state_predicates(domain: Domain) -> Dict[str, Set[GroundedPredicate]]:
    at_predicate = domain.predicates["at"]
    on_predicate = domain.predicates["on"]
    in_predicate = domain.predicates["in"]
    lifting_predicate = domain.predicates["lifting"]
    available_predicate = domain.predicates["available"]
    clear_predicate = domain.predicates["clear"]
    return {
        "(at ?x ?y)": {
            GroundedPredicate(name="at", signature=at_predicate.signature,
                              object_mapping={"?x": "x1", "?y": "y1"})},
        "(on ?x ?y)": {
            GroundedPredicate(name="on", signature=on_predicate.signature,
                              object_mapping={"?x": "x1", "?y": "y1"})},
        "(in ?x ?y)": {
            GroundedPredicate(name="in", signature=in_predicate.signature,
                              object_mapping={"?x": "x1", "?y": "y1"})},
        "(lifting ?x ?y)": {
            GroundedPredicate(name="lifting", signature=lifting_predicate.signature,
                              object_mapping={"?x": "x1", "?y": "y1"})},
        "(available ?x)": {
            GroundedPredicate(name="available", signature=available_predicate.signature,
                              object_mapping={"?x": "x1"})},
        "(clear ?x)": {
            GroundedPredicate(name="clear", signature=clear_predicate.signature,
                              object_mapping={"?x": "x1"})}
    }


@fixture()
def valid_previous_state(previous_state_predicates: Dict[str, Set[GroundedPredicate]]) -> State:
    return State(predicates=previous_state_predicates, fluents={})


def test_extract_effects_when_no_change_occurs_returns_empty_dictionaries(valid_previous_state: State):
    add_effects, delete_effects = extract_effects(valid_previous_state, valid_previous_state)
    assert len(add_effects) == 0
    assert len(delete_effects) == 0


def test_extract_effects_with_only_predicate_added_adds_to_add_effects(
        valid_previous_state: State, previous_state_predicates: Dict[str, Set[GroundedPredicate]], domain: Domain):
    clear_predicate = domain.predicates["clear"]
    next_state_predicates = {name: {*predicates} for name, predicates in previous_state_predicates.items()}
    next_state_predicates["(clear ?x)"].add(
        GroundedPredicate(name="clear", signature=clear_predicate.signature,
                          object_mapping={"?x": "x2"}))
    next_state = State(predicates=next_state_predicates, fluents={})
    add_effects, delete_effects = extract_effects(valid_previous_state, next_state)

    assert len(add_effects) == 1
    assert len(delete_effects) == 0


def test_extract_effects_with_only_predicate_removed_adds_to_delete_effects(
        valid_previous_state: State, previous_state_predicates: Dict[str, Set[GroundedPredicate]], domain: Domain):
    clear_predicate = domain.predicates["clear"]
    next_state_predicates = {name: {*predicates} for name, predicates in previous_state_predicates.items()}
    next_state_predicates["(clear ?x)"].add(
        GroundedPredicate(name="clear", signature=clear_predicate.signature,
                          object_mapping={"?x": "x2"}))
    next_state = State(predicates=next_state_predicates, fluents={})
    add_effects, delete_effects = extract_effects(next_state, valid_previous_state)

    assert len(add_effects) == 0
    assert len(delete_effects) == 1

def test_extract_not_effects_when_has_add_effect(
        valid_previous_state: State, previous_state_predicates: Dict[str, Set[GroundedPredicate]], domain: Domain):
    clear_predicate = domain.predicates["clear"]
    next_state_predicates = {name: {*predicates} for name, predicates in previous_state_predicates.items()}
    new_pred = GroundedPredicate(name="clear", signature=clear_predicate.signature,
                          object_mapping={"?x": "x2"})
    next_state_predicates["(clear ?x)"].add(new_pred)
    next_state = State(predicates=next_state_predicates, fluents={})
    add_effects, delete_effects = extract_effects(valid_previous_state, next_state)


    not_effects = {p.untyped_representation for p in extract_not_effects(next_state)}


    for predicates_set in next_state.state_predicates.values():
        for p in predicates_set:
            p_copy = p.copy(is_negated=True)
            assert p_copy.untyped_representation in not_effects


def test_extract_not_effects_when_has_delete_effect(
        valid_previous_state: State, previous_state_predicates: Dict[str, Set[GroundedPredicate]], domain: Domain):
    clear_predicate = domain.predicates["clear"]
    next_state_predicates = {name: {*predicates} for name, predicates in previous_state_predicates.items()}
    new_pred = GroundedPredicate(name="clear", signature=clear_predicate.signature,
                          object_mapping={"?x": "x2"})
    neg_new_pred = new_pred.copy(is_negated=True)
    next_state_predicates["(clear ?x)"].add(neg_new_pred)
    next_state = State(predicates=next_state_predicates, fluents={})
    valid_previous_state.state_predicates["(clear ?x)"].add(new_pred)

    not_effects = {p.untyped_representation for p in extract_not_effects(next_state)}

    for predicates_set in next_state.state_predicates.values():
        for p in predicates_set:
            p_copy = p.copy(is_negated=True)
            assert p_copy.untyped_representation in not_effects
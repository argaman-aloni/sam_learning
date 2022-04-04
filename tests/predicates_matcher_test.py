"""Test the extended functionality of the predicate matcher."""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import GroundedPredicate, Domain, ActionCall, Predicate, Problem, Observation
from pytest import fixture

from sam_learning.core import PredicatesMatcher
from tests.consts import LOCATION_TYPE, AGENT_TYPE, DOMAIN_WITH_CONSTS_PATH, DOMAIN_NO_CONSTS_PATH, PART_TYPE, \
    TREATMENT_STATUS_TYPE, SURFACE_TYPE, CITY_TYPE, OBJECT_TYPE, AIRPLANE_TYPE, TRUCK_TYPE, ELEVATORS_DOMAIN_PATH, \
    ELEVATORS_PROBLEM_PATH, ELEVATORS_TRAJECTORY_PATH

TRUCK_AT_LOCATION_GROUNDED_PREDICATE = GroundedPredicate(
    name="at",
    signature={"?obj": OBJECT_TYPE,
               "?loc": LOCATION_TYPE},
    object_mapping={"?obj": "tru1", "?loc": "pos1"})

IN_CITY_GROUNDED_PREDICATE = GroundedPredicate(
    name="in-city",
    signature={"?agent": TRUCK_TYPE,
               "?loc": LOCATION_TYPE,
               "?city": CITY_TYPE},
    object_mapping={"?agent": "tru1", "?loc": "pos1", "?city": "city1"})

IN_GROUNDED_PREDICATE = GroundedPredicate(
    name="in",
    signature={"?obj": OBJECT_TYPE,
               "?airplane": AIRPLANE_TYPE},
    object_mapping={"?obj": "pack1", "?airplane": "airplane1"})

TREATMENT_GROUNDED_PREDICATE = GroundedPredicate(
    name="treatment",
    signature={"?obj": PART_TYPE,
               "?treatment": TREATMENT_STATUS_TYPE},
    object_mapping={"?obj": "obj1", "?treatment": "untreated"})

IS_SMOOTH_GROUNDED_PREDICATE = GroundedPredicate(
    name="is-smooth",
    signature={"?surface": SURFACE_TYPE},
    object_mapping={"?surface": "verysmooth"})


@fixture()
def discrete_domain() -> Domain:
    parser = DomainParser(DOMAIN_NO_CONSTS_PATH, partial_parsing=True)
    return parser.parse_domain()


@fixture()
def elevators_domain() -> Domain:
    domain_parser = DomainParser(ELEVATORS_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def elevators_problem(elevators_domain: Domain) -> Problem:
    return ProblemParser(problem_path=ELEVATORS_PROBLEM_PATH, domain=elevators_domain).parse_problem()


@fixture()
def elevators_observation(elevators_domain: Domain, elevators_problem: Problem) -> Observation:
    return TrajectoryParser(elevators_domain, elevators_problem).parse_trajectory(ELEVATORS_TRAJECTORY_PATH)


@fixture()
def discrete_domain_with_consts() -> Domain:
    parser = DomainParser(DOMAIN_WITH_CONSTS_PATH, partial_parsing=True)
    return parser.parse_domain()


@fixture()
def predicate_matcher_no_consts(discrete_domain: Domain) -> PredicatesMatcher:
    return PredicatesMatcher(domain=discrete_domain)


@fixture()
def predicate_matcher_with_consts(discrete_domain_with_consts: Domain) -> PredicatesMatcher:
    return PredicatesMatcher(domain=discrete_domain_with_consts)


@fixture()
def elevators_predicate_matcher(elevators_domain: Domain) -> PredicatesMatcher:
    return PredicatesMatcher(domain=elevators_domain)


def test_match_predicate_to_action_with_no_match_returns_empty_list(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru2", "pos1", "pos2", "city1"])
    actual_predicates = predicate_matcher_no_consts.match_predicate_to_action_literals(
        grounded_predicate=TRUCK_AT_LOCATION_GROUNDED_PREDICATE, action_call=test_action_call)

    assert len(actual_predicates) == 0


def test_match_predicate_to_action_with_no_duplicated_parameters_returns_correct_match(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru1", "pos1", "pos2", "city1"])
    actual_predicates = predicate_matcher_no_consts.match_predicate_to_action_literals(
        grounded_predicate=TRUCK_AT_LOCATION_GROUNDED_PREDICATE, action_call=test_action_call)

    expected_predicate = Predicate(name="at", signature={"?truck": TRUCK_TYPE, "?loc-from": LOCATION_TYPE})

    assert len(actual_predicates) == 1
    assert expected_predicate.untyped_representation == actual_predicates[0].untyped_representation


def test_match_predicate_to_action_with_out_of_order_parameters_returns_correct_match(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="unload-airplane", grounded_parameters=["airplane1", "pack1", "pos1"])
    actual_predicates = predicate_matcher_no_consts.match_predicate_to_action_literals(
        grounded_predicate=IN_GROUNDED_PREDICATE, action_call=test_action_call)

    expected_predicate = Predicate(name="in", signature={"?obj": OBJECT_TYPE, "?airplane": AIRPLANE_TYPE})

    assert len(actual_predicates) == 1
    assert expected_predicate.untyped_representation == actual_predicates[0].untyped_representation


def test_match_predicate_to_action_with_no_duplicated_parameters_with_consts_returns_correct_match(
        predicate_matcher_with_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="do-spray-varnish",
                                  grounded_parameters=["m1", "obj1", "natural", "rough"])
    actual_predicates = predicate_matcher_with_consts.match_predicate_to_action_literals(
        grounded_predicate=TREATMENT_GROUNDED_PREDICATE, action_call=test_action_call)

    expected_predicate = Predicate(name="treatment", signature={"?x": AGENT_TYPE, "untreated": LOCATION_TYPE})

    assert len(actual_predicates) == 1
    assert expected_predicate.untyped_representation == actual_predicates[0].untyped_representation


def test_match_predicate_to_action_with_no_duplicated_parameters_where_predicate_contains_only_consts_returns_correct_match(
        predicate_matcher_with_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="do-spray-varnish",
                                  grounded_parameters=["m1", "obj1", "natural", "verysmooth"])
    actual_predicates = predicate_matcher_with_consts.match_predicate_to_action_literals(
        grounded_predicate=IS_SMOOTH_GROUNDED_PREDICATE, action_call=test_action_call)

    expected_predicates = [Predicate(name="is-smooth", signature={"?surface": SURFACE_TYPE}),
                           Predicate(name="is-smooth", signature={"verysmooth": SURFACE_TYPE})]

    assert len(actual_predicates) == 2
    assert [p.untyped_representation for p in expected_predicates] == [p.untyped_representation for p in
                                                                       actual_predicates]


def test_match_predicate_to_action_literals_with_complex_action_returns_only_correct_matches(
        elevators_predicate_matcher: PredicatesMatcher, elevators_domain: Domain):
    test_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n16"])
    next_predicate = elevators_domain.predicates["next"]
    test_grounded_predicate = GroundedPredicate(
        name=next_predicate.name,
        signature=next_predicate.signature,
        object_mapping={
            "?n1": "n16",
            "?n2": "n17",
        })

    actual_predicates = elevators_predicate_matcher.match_predicate_to_action_literals(
        grounded_predicate=test_grounded_predicate, action_call=test_action_call)
    assert len(actual_predicates) == 1
    assert actual_predicates[0].untyped_representation == "(next ?f2 ?f1)"


def test_match_predicate_to_action_literals_with_complex_action_with_duplicates_returns_only_correct_matches(
        elevators_predicate_matcher: PredicatesMatcher, elevators_domain: Domain):
    test_action_call = ActionCall(name="move-down-slow", grounded_parameters=["slow2-0", "n17", "n17"])
    next_predicate = elevators_domain.predicates["next"]
    test_grounded_predicate = GroundedPredicate(
        name=next_predicate.name,
        signature=next_predicate.signature,
        object_mapping={
            "?n1": "n17",
            "?n2": "n17",
        })

    actual_predicates = elevators_predicate_matcher.match_predicate_to_action_literals(
        grounded_predicate=test_grounded_predicate, action_call=test_action_call)
    assert len(actual_predicates) == 2
    assert [p.untyped_representation for p in actual_predicates] == ["(next ?f1 ?f2)", "(next ?f2 ?f1)"]


def test_match_predicate_to_action_with_duplicated_objects_finds_all_possible_matches(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru1", "pos1", "pos1", "city1"])
    actual_predicates = predicate_matcher_no_consts.match_predicate_to_action_literals(
        grounded_predicate=TRUCK_AT_LOCATION_GROUNDED_PREDICATE, action_call=test_action_call)
    expected_predicates = [Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-from": LOCATION_TYPE}),
                           Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-to": LOCATION_TYPE})]

    assert len(actual_predicates) == 2
    assert [p.untyped_representation for p in expected_predicates] == [p.untyped_representation for p in
                                                                       actual_predicates]


def test_get_possible_literal_matches_with_single_predicate_without_duplicates_returns_correct_matches(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru1", "pos1", "pos2", "city1"])
    test_state_predicates = [TRUCK_AT_LOCATION_GROUNDED_PREDICATE]
    possible_matches = predicate_matcher_no_consts.get_possible_literal_matches(test_action_call,
                                                                                test_state_predicates)

    expected_predicate = Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-from": LOCATION_TYPE})

    assert len(possible_matches) == 1
    assert expected_predicate.untyped_representation == possible_matches[0].untyped_representation


def test_get_possible_literal_matches_with_single_predicate_with_duplicates_returns_correct_matches(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru1", "pos1", "pos1", "city1"])
    test_state_predicates = [TRUCK_AT_LOCATION_GROUNDED_PREDICATE]
    possible_matches = predicate_matcher_no_consts.get_possible_literal_matches(test_action_call,
                                                                                test_state_predicates)

    expected_predicates = [Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-from": LOCATION_TYPE}),
                           Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-to": LOCATION_TYPE})]

    assert len(possible_matches) == 2
    assert [p.untyped_representation for p in expected_predicates] == [p.untyped_representation for p in
                                                                       possible_matches]


def test_get_possible_literal_matches_with_two_predicate_returns_correct_matches(
        predicate_matcher_no_consts: PredicatesMatcher):
    test_action_call = ActionCall(name="drive-truck", grounded_parameters=["tru1", "pos1", "pos2", "city1"])
    test_state_predicates = [TRUCK_AT_LOCATION_GROUNDED_PREDICATE, IN_CITY_GROUNDED_PREDICATE]
    possible_matches = predicate_matcher_no_consts.get_possible_literal_matches(test_action_call,
                                                                                test_state_predicates)

    expected_predicates = [
        Predicate(name="at", signature={"?truck": AGENT_TYPE, "?loc-from": LOCATION_TYPE}),
        Predicate(name="in-city", signature={"?truck": AGENT_TYPE, "?loc-from": LOCATION_TYPE, "?city": CITY_TYPE})]

    assert len(possible_matches) == 2
    assert [p.untyped_representation for p in expected_predicates] == [p.untyped_representation for p in
                                                                       possible_matches]


def test_get_possible_literal_matches_from_actual_trajectory_state(
        elevators_predicate_matcher: PredicatesMatcher, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    test_action_call = observation_component.grounded_action_call

    previous_state_predicates = []
    for predicate_set in observation_component.previous_state.state_predicates.values():
        previous_state_predicates.extend(predicate_set)

    possible_matches = elevators_predicate_matcher.get_possible_literal_matches(test_action_call,
                                                                                previous_state_predicates)
    for matched_lifted_predicate in possible_matches:
        for parameter in matched_lifted_predicate.signature:
            assert parameter in ["?lift", "?f1", "?f2"]

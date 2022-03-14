"""Module tests for the numeric function matcher."""
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, PDDLFunction, ActionCall
from pytest import fixture

from sam_learning.core import NumericFunctionMatcher
from tests.consts import NUMERIC_DOMAIN_WITH_PATH, TRUCK_TYPE

TEST_NUMERIC_LOAD_LIMIT_FUNCTION = PDDLFunction(name="load_limit", signature={
    "?t": TRUCK_TYPE
})

TEST_GROUNDED_NUMERIC_LOAD_LIMIT_FUNCTION = PDDLFunction(name="load_limit", signature={
    "tru1": TRUCK_TYPE
})

FUEL_COST_FUNCTION = PDDLFunction(name="fuel-cost", signature={})
LOAD_LIMIT_TRAJECTORY_FUNCTION = PDDLFunction(name="load_limit", signature={"truck1": TRUCK_TYPE})
CURRENT_LIMIT_TRAJECTORY_FUNCTION = PDDLFunction(name="current_load", signature={"truck1": TRUCK_TYPE})


@fixture()
def numeric_domain() -> Domain:
    parser = DomainParser(NUMERIC_DOMAIN_WITH_PATH, partial_parsing=True)
    return parser.parse_domain()


@fixture()
def numeric_function_matcher(numeric_domain: Domain) -> NumericFunctionMatcher:
    return NumericFunctionMatcher(domain=numeric_domain)


def test_create_possible_function_signatures_with_simple_action_creates_all_possible_signature_permutations(
        numeric_function_matcher: NumericFunctionMatcher):
    test_action_call_params = ActionCall(name="load", grounded_parameters=["hoist1", "crate1", "tru1", "loc1"])
    possible_signatures = numeric_function_matcher.create_possible_function_signatures(
        action_parameters=test_action_call_params.parameters, numeric_function=TEST_NUMERIC_LOAD_LIMIT_FUNCTION)

    assert len(possible_signatures) == 4
    assert possible_signatures == ["(load_limit hoist1)", "(load_limit crate1)", "(load_limit tru1)",
                                   "(load_limit loc1)"]

def test_lift_matched_parameters_creates_lifted_version_of_function(
        numeric_function_matcher: NumericFunctionMatcher, numeric_domain: Domain):
    TEST_GROUNDED_NUMERIC_LOAD_LIMIT_FUNCTION.set_value(132.87)
    test_lifted_action = numeric_domain.actions["load"]
    test_action_call_params = ActionCall(name="load", grounded_parameters=["hoist1", "crate1", "tru1", "loc1"])
    lifted_function = numeric_function_matcher.lift_matched_parameters(
        executed_action=test_lifted_action, grounded_call_parameters=test_action_call_params.parameters,
        grounded_function=TEST_GROUNDED_NUMERIC_LOAD_LIMIT_FUNCTION)

    assert lifted_function.untyped_representation == "(load_limit ?z)"
    assert lifted_function.value == 132.87

def test_match_state_functions_with_single_parameterized_grounded_function_finds_all_possible_matches(
        numeric_function_matcher: NumericFunctionMatcher):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit truck1)" : LOAD_LIMIT_TRAJECTORY_FUNCTION
    }
    test_action_call = ActionCall(name="load", grounded_parameters=["hoist2", "crate1", "truck1", "distributor1"])
    matches = numeric_function_matcher.match_state_functions(action_call=test_action_call,
                                                             grounded_state_fluents=simple_state_fluents)
    assert len(matches) == 2
    assert list(matches.keys()) == ["(load_limit ?z)", "(fuel-cost )"]
    assert matches["(fuel-cost )"].value == 34.0
    assert matches["(load_limit ?z)"].value == 411.0


def test_match_state_functions_with_small_number_of_grounded_functions_finds_all_possible_matches(
        numeric_function_matcher: NumericFunctionMatcher):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)

    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit truck1)" : LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load truck1)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
    }
    test_action_call = ActionCall(name="load", grounded_parameters=["hoist2", "crate1", "truck1", "distributor1"])
    matches = numeric_function_matcher.match_state_functions(action_call=test_action_call,
                                                             grounded_state_fluents=simple_state_fluents)
    assert len(matches) == 3
    assert list(matches.keys()) == ["(load_limit ?z)", "(current_load ?z)", "(fuel-cost )"]
    assert matches["(fuel-cost )"].value == 34.0
    assert matches["(load_limit ?z)"].value == 411.0
    assert matches["(current_load ?z)"].value == 121.0

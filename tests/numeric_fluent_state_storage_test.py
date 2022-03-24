"""Module test for the numeric state storage."""
import numpy as np
from pddl_plus_parser.lisp_parsers import PDDLTokenizer
from pddl_plus_parser.models import PDDLFunction, construct_expression_tree, NumericalExpressionTree
from pytest import fixture, fail

from sam_learning.core import NumericFluentStateStorage
from tests.consts import TRUCK_TYPE

FUEL_COST_FUNCTION = PDDLFunction(name="fuel-cost", signature={})
LOAD_LIMIT_TRAJECTORY_FUNCTION = PDDLFunction(name="load_limit", signature={"?z": TRUCK_TYPE})
CURRENT_LIMIT_TRAJECTORY_FUNCTION = PDDLFunction(name="current_load", signature={"?z": TRUCK_TYPE})

TEST_DOMAIN_FUNCTIONS = {
    "load_limit": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    "current_load": CURRENT_LIMIT_TRAJECTORY_FUNCTION,
    "fuel-cost": FUEL_COST_FUNCTION
}


@fixture()
def load_action_state_fluent_storage() -> NumericFluentStateStorage:
    return NumericFluentStateStorage(action_name="load")


def test_add_to_previous_state_storage_can_add_single_item_to_the_storage(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    assert load_action_state_fluent_storage.previous_state_storage["(fuel-cost )"] == [34.0]
    assert load_action_state_fluent_storage.previous_state_storage["(load_limit ?z)"] == [411.0]
    assert load_action_state_fluent_storage.previous_state_storage["(current_load ?z)"] == [121.0]


def test_add_to_next_state_storage_can_add_single_item_to_the_storage(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_next_state_storage(simple_state_fluents)
    assert load_action_state_fluent_storage.next_state_storage["(fuel-cost )"] == [34.0]
    assert load_action_state_fluent_storage.next_state_storage["(load_limit ?z)"] == [411.0]
    assert load_action_state_fluent_storage.next_state_storage["(current_load ?z)"] == [121.0]


def test_add_to_previous_state_storage_can_add_multiple_state_values_correctly(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    for i in range(10):
        FUEL_COST_FUNCTION.set_value(i)
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 1)
        CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    assert len(load_action_state_fluent_storage.previous_state_storage["(fuel-cost )"]) == 10
    assert len(load_action_state_fluent_storage.previous_state_storage["(load_limit ?z)"]) == 10
    assert len(load_action_state_fluent_storage.previous_state_storage["(current_load ?z)"]) == 10


def test_add_to_next_state_storage_can_add_multiple_state_values_correctly(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    for i in range(10):
        FUEL_COST_FUNCTION.set_value(i)
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 1)
        CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_next_state_storage(simple_state_fluents)

    assert len(load_action_state_fluent_storage.next_state_storage["(fuel-cost )"]) == 10
    assert len(load_action_state_fluent_storage.next_state_storage["(load_limit ?z)"]) == 10
    assert len(load_action_state_fluent_storage.next_state_storage["(current_load ?z)"]) == 10


def test_convert_to_array_format_with_simple_state_fluents_returns_correct_array(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    for i in range(10):
        FUEL_COST_FUNCTION.set_value(i)
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 1)
        CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    array = load_action_state_fluent_storage.convert_to_array_format(storage_name="previous_state")
    assert array.shape == (10, 3)


def test_create_convex_hull_linear_inequalities_generates_correct_equations_with_simple_points(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    rng = np.random.default_rng(42)
    points = rng.random((30, 2))
    A, b = load_action_state_fluent_storage.create_convex_hull_linear_inequalities(points)

    for index, (point, coeff) in enumerate(zip(points, A)):
        value = sum([p * c for p, c in zip(point, coeff)])
        assert value <= b[index]


def test_construct_pddl_inequality_scheme_with_simple_2d_four_equations_returns_correct_representation(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(121.0)
    simple_state_fluents = {
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 2))
    right_side_points = np.random.randint(10, size=4)

    inequlities = load_action_state_fluent_storage.construct_pddl_inequality_scheme(left_side_coefficients,
                                                                                    right_side_points)

    joint_inequalities = "\n".join(inequlities)
    joint_inequalities = f"({joint_inequalities})"
    pddl_tokenizer = PDDLTokenizer(pddl_str=joint_inequalities)

    parsed_expressions = pddl_tokenizer.parse()
    assert len(parsed_expressions) == 4
    for expression in parsed_expressions:
        try:
            expression_node = construct_expression_tree(expression, TEST_DOMAIN_FUNCTIONS)
            expression_tree = NumericalExpressionTree(expression_node)
            print(str(expression_tree))

        except Exception:
            fail()


def test_construct_pddl_inequality_scheme_with_simple_23_four_equations_returns_correct_representation(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LIMIT_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LIMIT_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 3))
    right_side_points = np.random.randint(10, size=4)

    inequalities = load_action_state_fluent_storage.construct_pddl_inequality_scheme(left_side_coefficients,
                                                                                    right_side_points)

    joint_inequalities = "\n".join(inequalities)
    joint_inequalities = f"({joint_inequalities})"
    pddl_tokenizer = PDDLTokenizer(pddl_str=joint_inequalities)

    parsed_expressions = pddl_tokenizer.parse()
    assert len(parsed_expressions) == 4
    for expression in parsed_expressions:
        try:
            expression_node = construct_expression_tree(expression, TEST_DOMAIN_FUNCTIONS)
            expression_tree = NumericalExpressionTree(expression_node)
            print(str(expression_tree))

        except Exception:
            fail()
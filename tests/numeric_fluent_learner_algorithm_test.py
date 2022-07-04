"""Module test for the numeric state storage."""
import numpy as np
from pddl_plus_parser.lisp_parsers import PDDLTokenizer
from pddl_plus_parser.models import PDDLFunction, construct_expression_tree, NumericalExpressionTree
from pytest import fixture, fail, raises

from sam_learning.core import NumericFluentStateStorage, ConditionType, NotSafeActionError
from tests.consts import TRUCK_TYPE

FUEL_COST_FUNCTION = PDDLFunction(name="fuel-cost", signature={})
LOAD_LIMIT_TRAJECTORY_FUNCTION = PDDLFunction(name="load_limit", signature={"?z": TRUCK_TYPE})
CURRENT_LOAD_TRAJECTORY_FUNCTION = PDDLFunction(name="current_load", signature={"?z": TRUCK_TYPE})

TEST_DOMAIN_FUNCTIONS = {
    "load_limit": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    "current_load": CURRENT_LOAD_TRAJECTORY_FUNCTION,
    "fuel-cost": FUEL_COST_FUNCTION
}


@fixture()
def load_action_state_fluent_storage() -> NumericFluentStateStorage:
    return NumericFluentStateStorage(action_name="load")


def test_add_to_previous_state_storage_can_add_single_item_to_the_storage(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    assert load_action_state_fluent_storage.previous_state_storage["(fuel-cost )"] == [34.0]
    assert load_action_state_fluent_storage.previous_state_storage["(load_limit ?z)"] == [411.0]
    assert load_action_state_fluent_storage.previous_state_storage["(current_load ?z)"] == [121.0]


def test_add_to_next_state_storage_can_add_single_item_to_the_storage(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
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
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
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
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
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
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(i + 2)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    array = load_action_state_fluent_storage._convert_to_array_format(storage_name="previous_state")
    assert array.shape == (10, 3)


def test_create_convex_hull_linear_inequalities_generates_correct_equations_with_simple_points(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    rng = np.random.default_rng(42)
    points = rng.random((30, 2))
    A, b = load_action_state_fluent_storage._create_convex_hull_linear_inequalities(points)

    for index, (point, coeff) in enumerate(zip(points, A)):
        value = sum([p * c for p, c in zip(point, coeff)])
        assert value <= b[index]


def test_construct_pddl_inequality_scheme_with_simple_2d_four_equations_returns_correct_representation(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    simple_state_fluents = {
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 2))
    right_side_points = np.random.randint(10, size=4)

    inequalities = load_action_state_fluent_storage._construct_pddl_inequality_scheme(
        left_side_coefficients, right_side_points, ["(load_limit ?z)", "(current_load ?z)"])

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


def test_construct_non_circular_assignment_constructs_correct_equation_with_correct_coefficient_sign_on_increase(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    lifted_function = "(current_load ?z)"
    coefficient_map = {
        "(current_load ?z)": 1.0,
        "(weight ?y)": 1.0
    }
    previous_value = 0.0
    next_value = 1.0
    increase_statement = load_action_state_fluent_storage._construct_non_circular_assignment(
        lifted_function, coefficient_map, previous_value, next_value)
    assert increase_statement == "(increase (current_load ?z) (* (weight ?y) 1.0))"


def test_construct_non_circular_assignment_constructs_correct_equation_with_correct_coefficient_sign_on_decrease(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    lifted_function = "(current_load ?z)"
    coefficient_map = {
        "(current_load ?z)": 1.0,
        "(weight ?y)": -1.0,
        "(dummy)": 0.0
    }
    previous_value = 1.0
    next_value = 0.0
    increase_statement = load_action_state_fluent_storage._construct_non_circular_assignment(
        lifted_function, coefficient_map, previous_value, next_value)
    assert increase_statement == "(decrease (current_load ?z) (* (weight ?y) 1.0))"


def test_construct_pddl_inequality_scheme_with_simple_3d_four_equations_returns_correct_representation(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 3))
    right_side_points = np.random.randint(10, size=4)

    inequalities = load_action_state_fluent_storage._construct_pddl_inequality_scheme(
        left_side_coefficients, right_side_points, ["(load_limit ?z)", "(current_load ?z)", "(fuel-cost )"])

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


def test_construct_assignment_equations_with_simple_2d_equations_when_no_change_in_variables_returns_empty_list(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    for i in range(3):
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 1)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(i)
        simple_prev_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
        load_action_state_fluent_storage.add_to_next_state_storage(simple_prev_state_fluents)

    assignment_equations = load_action_state_fluent_storage.construct_assignment_equations()
    assert len(assignment_equations) == 0


def test_construct_assignment_equations_when_change_is_caused_by_constant_returns_correct_value(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    # This tests is meant to validate that cases such as (assign (battery-level ?r) 10) can be handled.
    for i in range(3):
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(i + 1)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(i)
        simple_prev_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(10)
        simple_next_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_next_state_storage(simple_next_state_fluents)

    assignment_equations = load_action_state_fluent_storage.construct_assignment_equations()
    assert assignment_equations == ["(assign (current_load ?z) 10.0)"]


def test_construct_assignment_equations_with_simple_2d_equations_returns_correct_string_representation(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    previous_state_values = [(1, 7), (2, -1), (2, 14), (1, 0)]
    next_state_values = [9, 18, 18, 9]
    for prev_values, next_state_value in zip(previous_state_values, next_state_values):
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(prev_values[0])
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(prev_values[1])
        simple_prev_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(next_state_value)
        simple_next_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_next_state_storage(simple_next_state_fluents)

    assignment_equations = load_action_state_fluent_storage.construct_assignment_equations()
    assert len(assignment_equations) == 1
    assert assignment_equations == [
        "(assign (current_load ?z) (* (load_limit ?z) 9.0))"]


def test_construct_assignment_equations_with_two_equations_result_in_multiple_changes(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    previous_state_values = [(1, 7), (2, -1), (2, 14), (1, 0)]
    next_state_values = [(7, 9), (-16, 18), (14, 18), (-7, 9)]
    for prev_values, next_state_values in zip(previous_state_values, next_state_values):
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(prev_values[0])
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(prev_values[1])
        simple_prev_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(next_state_values[0])
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(next_state_values[1])
        simple_next_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_next_state_storage(simple_next_state_fluents)

    assignment_equations = load_action_state_fluent_storage.construct_assignment_equations()
    assert len(assignment_equations) == 2
    assert set(assignment_equations) == {
        "(increase (load_limit ?z) (* (current_load ?z) 0.2857142857142857))",
        "(assign (current_load ?z) (* (load_limit ?z) 9.0))"}


def test_construct_assignment_equations_with_an_increase_change_results_in_correct_values(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    previous_state_values = [(0, 7), (2, -1), (12, 32)]
    next_state_values = [(0, 8), (2, 0), (12, 33)]
    for prev_values, next_state_values in zip(previous_state_values, next_state_values):
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(prev_values[0])
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(prev_values[1])
        simple_prev_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
        LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(next_state_values[0])
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(next_state_values[1])
        simple_next_state_fluents = {
            "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_next_state_storage(simple_next_state_fluents)

    assignment_equations = load_action_state_fluent_storage.construct_assignment_equations()
    assert len(assignment_equations) == 1
    print(assignment_equations)


def test_construct_assignment_equations_only_one_observation_raises_not_safe_error(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(7)
    simple_prev_state_fluents = {
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_prev_state_fluents)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(8)
    simple_next_state_fluents = {
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_next_state_storage(simple_next_state_fluents)

    with raises(NotSafeActionError):
        load_action_state_fluent_storage.construct_assignment_equations()


def test_construct_safe_linear_inequalities_when_given_only_one_state_returns_degraded_conditions(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities(
        ["(fuel-cost )", "(load_limit ?z)", "(current_load ?z)"])
    assert condition_type == ConditionType.injunctive
    assert output_conditions == ["(= (fuel-cost ) 34.0) (= (load_limit ?z) 411.0) (= (current_load ?z) 121.0)"]


def test_construct_safe_linear_inequalities_when_given_only_two_states_returns_two_disjunctive_preconditions(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(411.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(34.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(413.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(121.0)
    FUEL_COST_FUNCTION.set_value(35.0)
    another_simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    load_action_state_fluent_storage.add_to_previous_state_storage(another_simple_state_fluents)
    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities(
        ["(fuel-cost )", "(load_limit ?z)", "(current_load ?z)"])
    assert condition_type == ConditionType.disjunctive
    assert output_conditions == ["(and (= (fuel-cost ) 34.0) (= (load_limit ?z) 411.0) (= (current_load ?z) 121.0))",
                                 "(and (= (fuel-cost ) 35.0) (= (load_limit ?z) 413.0) (= (current_load ?z) 121.0))"]


def test_construct_safe_linear_inequalities_will_create_correct_inequalities_when_given_three_points_for_two_variables(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    pre_state_input_values = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
    for fuel_cost_val, current_limit_val in pre_state_input_values:
        FUEL_COST_FUNCTION.set_value(fuel_cost_val)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(current_limit_val)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities(
        ["(fuel-cost )", "(current_load ?z)"])

    expected_conditions = ["(<= (* (fuel-cost ) -1.0) 0.0)",
                           "(<= (* (current_load ?z) -1.0) 0.0)",
                           "(<= (+ (* (fuel-cost ) 0.71) (* (current_load ?z) 0.71)) 0.71)"]
    assert set(output_conditions) == set(expected_conditions)


def test_construct_safe_linear_inequalities_will_create_correct_inequalities_when_given_four_points_for_two_variables(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    pre_state_input_values = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for fuel_cost_val, current_limit_val in pre_state_input_values:
        FUEL_COST_FUNCTION.set_value(fuel_cost_val)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(current_limit_val)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities(
        ["(fuel-cost )", "(current_load ?z)"])
    expected_conditions = ["(<= (* (current_load ?z) 1.0) 1.0)",
                           "(<= (* (fuel-cost ) -1.0) 0.0)",
                           "(<= (* (fuel-cost ) 1.0) 1.0)",
                           "(<= (* (current_load ?z) -1.0) 0.0)"]
    assert set(output_conditions) == set(expected_conditions)


def test_construct_safe_linear_inequalities_with_one_dimension_variable_select_min_and_max_values(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    pre_state_input_values = [(-19.0, 32.0), (14.0, 52.0), (28.0, 12.0), (-7.0, 13.0)]
    for fuel_cost_val, current_limit_val in pre_state_input_values:
        FUEL_COST_FUNCTION.set_value(fuel_cost_val)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(current_limit_val)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities(
        ["(fuel-cost )"])
    expected_conditions = ["(<= (fuel-cost ) 28.0)",
                           "(>= (fuel-cost ) -19.0)"]
    assert set(output_conditions) == set(expected_conditions)


def test_construct_safe_linear_when_not_given_relevant_fluents_uses_all_variables_in_previous_state(
        load_action_state_fluent_storage: NumericFluentStateStorage):
    pre_state_input_values = [(-19.0, 32.0), (14.0, 52.0), (28.0, 12.0), (-7.0, 13.0)]
    for fuel_cost_val, current_limit_val in pre_state_input_values:
        FUEL_COST_FUNCTION.set_value(fuel_cost_val)
        CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(current_limit_val)
        simple_state_fluents = {
            "(fuel-cost )": FUEL_COST_FUNCTION,
            "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
        }
        load_action_state_fluent_storage.add_to_previous_state_storage(simple_state_fluents)

    output_conditions, condition_type = load_action_state_fluent_storage.construct_safe_linear_inequalities()
    print(output_conditions)
    assert all(["(fuel-cost )" in condition and "(current_load ?z)" in condition for condition in output_conditions])
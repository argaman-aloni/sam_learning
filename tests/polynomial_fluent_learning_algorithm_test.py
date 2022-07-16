"""Module test for the polynomial state storage."""

from sam_learning.core import PolynomialFluentsLearningAlgorithm
from tests.consts import FUEL_COST_FUNCTION, LOAD_LIMIT_TRAJECTORY_FUNCTION, \
    CURRENT_LOAD_TRAJECTORY_FUNCTION


def test_create_polynomial_string_returns_correct_string():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=1)
    test_fluents_names = ["(fuel-cost)", "(load_limit ?z)", "(current_load ?z)"]
    polynomial_str = polynomial_fluent_storage.create_polynomial_string(test_fluents_names)
    assert polynomial_str == "(* (fuel-cost) (* (load_limit ?z) (current_load ?z)))"


def test_add_to_previous_state_storage_reverts_to_super_when_polynom_degree_is_zero():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=0)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_keys = list(polynomial_fluent_storage.previous_state_storage.keys())
    storage_values = list(polynomial_fluent_storage.previous_state_storage.values())
    assert storage_keys == ["(fuel-cost )", "(load_limit ?z)"]
    assert storage_values == [[2.0], [3.0]]


def test_add_to_previous_state_storage_adds_correct_keys_combinations_with_polynom_degree_one():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=1)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_keys = list(polynomial_fluent_storage.previous_state_storage.keys())
    assert storage_keys == ["(fuel-cost )", "(load_limit ?z)", "(* (fuel-cost ) (load_limit ?z))"]


def test_add_to_previous_state_storage_adds_correct_values_combinations_with_polynom_degree_one():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=1)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_values = list(polynomial_fluent_storage.previous_state_storage.values())
    assert storage_values == [[2.0], [3.0], [6.0]]


def test_add_to_previous_state_storage_adds_correct_values_combinations_with_polynom_degree_one_with_complex_setting():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=1)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(4.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_keys = list(polynomial_fluent_storage.previous_state_storage.keys())
    storage_values = list(polynomial_fluent_storage.previous_state_storage.values())
    assert storage_keys == ["(fuel-cost )", "(load_limit ?z)", "(current_load ?z)",
                            "(* (fuel-cost ) (load_limit ?z))",
                            "(* (fuel-cost ) (current_load ?z))",
                            "(* (load_limit ?z) (current_load ?z))"]
    assert storage_values == [[2.0], [3.0], [4.0], [6.0], [8.0], [12.0]]


def test_add_to_previous_state_storage_adds_correct_values_combinations_with_polynom_degree_two():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=2)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(4.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_keys = list(polynomial_fluent_storage.previous_state_storage.keys())
    storage_values = list(polynomial_fluent_storage.previous_state_storage.values())
    assert storage_keys == ['(fuel-cost )', '(load_limit ?z)', '(current_load ?z)', '(* (fuel-cost ) (fuel-cost ))',
                            '(* (fuel-cost ) (load_limit ?z))', '(* (fuel-cost ) (current_load ?z))',
                            '(* (load_limit ?z) (load_limit ?z))', '(* (load_limit ?z) (current_load ?z))',
                            '(* (current_load ?z) (current_load ?z))']
    assert storage_values == [[2.0], [3.0], [4.0], [4.0], [6.0], [8.0], [9.0], [12.0], [16.0]]


def test_add_to_previous_state_storage_adds_correct_values_combinations_with_polynom_degree_three():
    polynomial_fluent_storage = PolynomialFluentsLearningAlgorithm(action_name="load", polynom_degree=3)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    CURRENT_LOAD_TRAJECTORY_FUNCTION.set_value(4.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
        "(current_load ?z)": CURRENT_LOAD_TRAJECTORY_FUNCTION
    }
    polynomial_fluent_storage.add_to_previous_state_storage(simple_state_fluents)
    storage_keys = list(polynomial_fluent_storage.previous_state_storage.keys())
    assert len(storage_keys) == 19

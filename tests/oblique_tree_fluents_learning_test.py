"""Module tests for the oblique tree fluents learning functionality."""
from collections import defaultdict

from pddl_plus_parser.models import Domain, Observation, PDDLFunction, State, ActionCall
from pytest import fixture

from sam_learning.core import ObliqueTreeFluentsLearning
from sam_learning.core.unsafe_numeric_fluents_learning_base import UnsafeFluentsLearning
from tests.consts import FUEL_COST_FUNCTION, \
    LOAD_LIMIT_TRAJECTORY_FUNCTION, TRUCK_TYPE


@fixture()
def oblique_tree_fluents_learning_zero_degree_polynom(depot_domain: Domain) -> ObliqueTreeFluentsLearning:
    return ObliqueTreeFluentsLearning(action_name="drive", polynomial_degree=0, partial_domain=depot_domain)


@fixture()
def oblique_tree_fluents_learning_first_degree_polynom(depot_domain: Domain) -> ObliqueTreeFluentsLearning:
    return ObliqueTreeFluentsLearning(action_name="drive", polynomial_degree=1, partial_domain=depot_domain)


def test_create_polynomial_string_returns_correct_string(
        oblique_tree_fluents_learning_first_degree_polynom: ObliqueTreeFluentsLearning):
    test_fluents_names = ["(fuel-cost)", "(load_limit ?z)", "(current_load ?z)"]
    monomial_str = oblique_tree_fluents_learning_first_degree_polynom._create_monomial_string(test_fluents_names)
    assert monomial_str == "(* (fuel-cost) (* (load_limit ?z) (current_load ?z)))"


def test_create_polynomial_string_returns_correct_string_when_given_the_same_fluent_twice(
        oblique_tree_fluents_learning_first_degree_polynom: ObliqueTreeFluentsLearning):
    test_fluents_names = ["(load_limit ?z)", "(load_limit ?z)"]
    monomial_str = oblique_tree_fluents_learning_first_degree_polynom._create_monomial_string(test_fluents_names)
    assert monomial_str == "(* (load_limit ?z) (load_limit ?z))"


def test_add_polynomial_adds_correct_polynom_when_polynomial_degree_is_one(
        oblique_tree_fluents_learning_first_degree_polynom: ObliqueTreeFluentsLearning):
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    }
    test_dataset = defaultdict(list)
    oblique_tree_fluents_learning_first_degree_polynom._add_polynomial(lifted_fluents=simple_state_fluents,
                                                                       dataset=test_dataset)
    storage_keys = list(test_dataset.keys())
    assert storage_keys == ["(* (fuel-cost ) (load_limit ?z))"]


def test_add_polynomial_adds_correct_polynom_when_polynomial_degree_is_two(depot_domain: Domain):
    oblique_tree = ObliqueTreeFluentsLearning(action_name="drive", polynomial_degree=2, partial_domain=depot_domain)
    FUEL_COST_FUNCTION.set_value(2.0)
    LOAD_LIMIT_TRAJECTORY_FUNCTION.set_value(3.0)
    simple_state_fluents = {
        "(fuel-cost )": FUEL_COST_FUNCTION,
        "(load_limit ?z)": LOAD_LIMIT_TRAJECTORY_FUNCTION,
    }
    test_dataset = defaultdict(list)
    oblique_tree._add_polynomial(lifted_fluents=simple_state_fluents, dataset=test_dataset)
    storage_keys = list(test_dataset.keys())
    assert storage_keys == ["(* (fuel-cost ) (fuel-cost ))", "(* (fuel-cost ) (load_limit ?z))",
                            "(* (load_limit ?z) (load_limit ?z))"]
    assert test_dataset["(* (fuel-cost ) (fuel-cost ))"] == [4.0]
    assert test_dataset["(* (fuel-cost ) (load_limit ?z))"] == [6.0]
    assert test_dataset["(* (load_limit ?z) (load_limit ?z))"] == [9.0]


def test_add_lifted_functions_to_dataset_lifts_grounded_observation_and_adds_correct_grounded_values_to_dataset(
        oblique_tree_fluents_learning_zero_degree_polynom: ObliqueTreeFluentsLearning, depot_observation: Observation):
    observed_component = depot_observation.components[0]
    test_dataset = defaultdict(list)

    parent: UnsafeFluentsLearning = oblique_tree_fluents_learning_zero_degree_polynom
    parent._add_lifted_functions_to_dataset(observed_component, test_dataset)
    assert (len(test_dataset) == 3)
    assert (test_dataset["(fuel-cost )"] == [0.0])
    assert (test_dataset["(load_limit ?x)"] == [411.0])
    assert (test_dataset["(current_load ?x)"] == [0.0])


def test_create_pre_state_classification_dataset_with_no_negative_observations_creates_correct_dataset(
        oblique_tree_fluents_learning_zero_degree_polynom: ObliqueTreeFluentsLearning, depot_observation: Observation):
    drive_action_components = [component for component in depot_observation.components
                               if component.grounded_action_call.name == "drive"]
    action_observation = Observation()
    action_observation.components = drive_action_components
    positive_observations = [action_observation]
    negative_observations = []

    parent: UnsafeFluentsLearning = oblique_tree_fluents_learning_zero_degree_polynom
    df = parent._create_pre_state_classification_dataset(positive_observations, negative_observations)
    assert df.shape[0] == len(action_observation.components)
    assert df.shape[1] == 4


def test_construct_linear_equation_string_returns_correct_string(
        oblique_tree_fluents_learning_zero_degree_polynom: ObliqueTreeFluentsLearning):
    multiplication_parts = ["(* (fuel-cost ) (load_limit ?z))", "(* (load_limit ?z) (current_load ?z))"]
    linear_equation_string = oblique_tree_fluents_learning_zero_degree_polynom._construct_linear_equation_string(
        multiplication_parts)
    assert linear_equation_string == "(+ (* (fuel-cost ) (load_limit ?z)) (* (load_limit ?z) (current_load ?z)))"


def test_create_inequality_constraint_strings_returns_correct_strings(
        oblique_tree_fluents_learning_zero_degree_polynom: ObliqueTreeFluentsLearning):
    feature_names = ["(fuel-cost )", "(load_limit ?x)", "(current_load ?x)"]
    coefficients_paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    intercept_path = [10, 11, 12]
    inequality_constraint_strings = \
        oblique_tree_fluents_learning_zero_degree_polynom._create_inequality_constraint_strings(
            feature_names, coefficients_paths, intercept_path)
    expected_inequalities = ['(and',
                             '(>= (+ (+ (* (fuel-cost ) 7) (+ (* (load_limit ?x) 8) (* (current_load ?x) 9))) 12) 0.0)',
                             '(>= (+ (+ (* (fuel-cost ) 1) (+ (* (load_limit ?x) 2) (* (current_load ?x) 3))) 10) 0.0)',
                             '(>= (+ (+ (* (fuel-cost ) 4) (+ (* (load_limit ?x) 5) (* (current_load ?x) 6))) 11) 0.0)',
                             ')']

    assert set(inequality_constraint_strings) == set(expected_inequalities)


def test_learn_preconditions_with_single_negative_observations_learns_preconditions(
        oblique_tree_fluents_learning_zero_degree_polynom: ObliqueTreeFluentsLearning, depot_observation: Observation):
    drive_action_components = [component for component in depot_observation.components
                               if component.grounded_action_call.name == "drive"]
    load_limit_trajectory_function = PDDLFunction(name="load_limit", signature={"?x": TRUCK_TYPE})
    current_load_trajectory_function = PDDLFunction(name="current_load", signature={"?x": TRUCK_TYPE})
    action_observation = Observation()
    action_observation.components = drive_action_components
    positive_observations = [action_observation]

    FUEL_COST_FUNCTION.set_value(89.0)
    load_limit_trajectory_function.set_value(328.0)
    current_load_trajectory_function.set_value(26.0)
    simple_state_fluents = {
        "(load_limit ?x)": load_limit_trajectory_function,
        "(current_load ?x)": current_load_trajectory_function,
        "(fuel-cost )": FUEL_COST_FUNCTION,
    }
    previous_state = State(predicates={}, fluents=simple_state_fluents, is_init=False)
    action_call = ActionCall(name="drive", grounded_parameters=["truck1", "from", "to"])
    negative_observation = Observation()
    negative_observation.add_component(previous_state, action_call, next_state=previous_state)
    negative_observations = [negative_observation]
    result = oblique_tree_fluents_learning_zero_degree_polynom.learn_preconditions(positive_observations,
                                                                                   negative_observations)
    print(result)

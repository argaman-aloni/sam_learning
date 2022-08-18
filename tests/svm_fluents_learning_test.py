"""Module test for the linear SVM classifier fluents learning module."""
import numpy as np
import pandas as pd
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation
from pytest import fixture

from sam_learning.core import SVMFluentsLearning
from tests.consts import NUMERIC_DOMAIN_PATH, NUMERIC_PROBLEM_PATH, DEPOT_NUMERIC_TRAJECTORY_PATH


@fixture()
def numeric_domain() -> Domain:
    parser = DomainParser(NUMERIC_DOMAIN_PATH, partial_parsing=True)
    return parser.parse_domain()


@fixture()
def numeric_problem(numeric_domain: Domain) -> Problem:
    return ProblemParser(problem_path=NUMERIC_PROBLEM_PATH, domain=numeric_domain).parse_problem()


@fixture()
def depot_observation(numeric_domain: Domain, numeric_problem: Problem) -> Observation:
    return TrajectoryParser(numeric_domain, numeric_problem).parse_trajectory(DEPOT_NUMERIC_TRAJECTORY_PATH)


@fixture()
def svc_fluents_learning_zero_degree_polynom(numeric_domain: Domain) -> SVMFluentsLearning:
    return SVMFluentsLearning(action_name="drive", polynomial_degree=0, partial_domain=numeric_domain)


def test_calculate_expression_class_identifies_that_row_that_does_not_classify_correctly_should_return_true(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning):
    row_values = np.array([0.1, 3.7, 11.0, -1])
    coefficients = [10.0, -3.0, 1.8]
    intercept = 1.0

    should_row_remain = svc_fluents_learning_zero_degree_polynom._calculate_expression_class(
        row_values, coefficients, intercept)
    assert should_row_remain


def test_calculate_expression_class_identifies_that_row_that_classified_correctly_should_return_false(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning):
    row_values = np.array([0.1, 3.7, 11.0, 1])
    coefficients = [10.0, -3.0, 1.8]
    intercept = 1.0

    should_row_remain = svc_fluents_learning_zero_degree_polynom._calculate_expression_class(
        row_values, coefficients, intercept)
    assert not should_row_remain


def test_remove_rows_with_accurate_classification_should_remove_one_line_the_only_one_with_correct_classification(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning):
    dataset = {
        "(fuel-cost )": [1, 2, 3, 4],
        "(load_limit ?x)": [5, 6, 7, 8],
        "(current_load ?x)": [9, 10, 19, 11],
        "class": [1, -1, 1, -1]
    }
    dataframe = pd.DataFrame.from_dict(dataset)
    coefficients = [1.0, 1.0, -1.0]
    intercept = 4.0
    output_df = svc_fluents_learning_zero_degree_polynom._remove_rows_with_accurate_classification(
        dataframe, coefficients, intercept)
    assert dataframe.shape[0] == output_df.shape[0] + 1


def test_construct_regression_data_with_single_triplet_creates_correct_datasets_from_positive_observation(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning, depot_observation: Observation):
    test_observation = Observation()
    component = depot_observation.components[0]
    test_observation.add_component(previous_state=component.previous_state, call=component.grounded_action_call,
                                   next_state=component.next_state)
    test_observation.add_problem_objects(depot_observation.grounded_objects)
    post_state_values, pre_state_values = svc_fluents_learning_zero_degree_polynom._construct_regression_data(
        [test_observation])
    assert len(post_state_values) == len(pre_state_values)


def test_construct_regression_data_with_multiple_triplets_creates_correct_datasets_from_positive_observation(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning, depot_observation: Observation):
    post_state_values, pre_state_values = svc_fluents_learning_zero_degree_polynom._construct_regression_data(
        [depot_observation])
    assert len(post_state_values) == len(pre_state_values)


def test_run_linear_svc_returns_correct_equations(svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning):
    dataset = {
        "(fuel-cost )": [1, 2, 3, 4],
        "(load_limit ?x)": [5, 6, 7, 8],
        "(current_load ?x)": [9, 10, 19, 11],
        "class": [1, -1, 1, -1]
    }
    dataframe = pd.DataFrame.from_dict(dataset)
    coef, intercept = svc_fluents_learning_zero_degree_polynom.run_linear_svc(dataframe)
    positive_points = dataframe[dataframe["class"] == 1].iloc[0].values[:-1]
    assert sum(positive_points * coef) + intercept >= 0


def test_solve_regression_returns_valid_coefficients_map(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning, depot_observation: Observation):
    label_col_name = "(load_limit ?x)_post_state"
    dataframe = pd.DataFrame.from_dict({
        "(fuel-cost )": [0, 2, 12, 12, 2],
        "(load_limit ?x)": [7, -1, 32, 32, -1],
        label_col_name: [8, 0, 33, 33, 0]
    })
    coefficients_map = svc_fluents_learning_zero_degree_polynom._solve_regression(dataframe, label_col_name)
    print(coefficients_map)
    assert coefficients_map["(fuel-cost )"] == 0.0
    assert coefficients_map["(load_limit ?x)"] == 1.0
    assert coefficients_map["(dummy)"] - 1 < 0.00001


def test_learn_effects_learns_effects_correctly_and_outputs_equation_strings(
        svc_fluents_learning_zero_degree_polynom: SVMFluentsLearning, depot_observation: Observation):
    equations = svc_fluents_learning_zero_degree_polynom.learn_effects([depot_observation])
    assert len(equations) == 1

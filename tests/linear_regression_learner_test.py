import numpy as np
import pytest
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction

from sam_learning.core import NotSafeActionError
from sam_learning.core.linear_regression_learner import LinearRegressionLearner

TEST_ACTION_NAME = 'test_action'


@pytest.fixture
def linear_regression_learner() -> LinearRegressionLearner:
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
        "w": PDDLFunction(name="w", signature={})
    }
    return LinearRegressionLearner(TEST_ACTION_NAME, domain_functions)


def test_validate_legal_equations_returns_true_when_the_number_of_equations_is_valid_and_the_matrix_has_one_solution(
        linear_regression_learner: LinearRegressionLearner):
    pre_state_data = {
        "x": [2, 1, 3, 0],
        "y": [3, -1, 2, 0],
        "z": [-1, 2, 4, 1]
    }
    dataframe = DataFrame(pre_state_data)
    assert linear_regression_learner._validate_legal_equations(dataframe)


def test_validate_legal_returns_false_when_there_are_not_enough_independent_equations(
        linear_regression_learner: LinearRegressionLearner):
    pre_state_data = {
        "x": [0, 1, 2, 4],
        "y": [0, 2, 4, 8],
        "z": [0, 0, 18, 6]
    }
    dataframe = DataFrame(pre_state_data)
    assert not linear_regression_learner._validate_legal_equations(dataframe)


def test_validate_legal_equations_returns_false_when_the_number_of_equations_is_too_small(
        linear_regression_learner: LinearRegressionLearner):
    pre_state_data = {
        "x": [2, 1, 3],
        "y": [3, -1, 2],
        "z": [-1, 2, 4]
    }
    dataframe = DataFrame(pre_state_data)
    assert not linear_regression_learner._validate_legal_equations(dataframe)


def test_solve_function_linear_equations_returns_correct_solution_for_a_solvable_matrix(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "x": [2, 1, 3, 0],
        "y": [3, -1, 2, 0],
        "z": [-1, 2, 4, 1],
        "label": [4, -1, 9, 15 / 35]
    }
    dataframe = DataFrame(equation_matrix)
    regression_array = np.array(dataframe.loc[:, dataframe.columns != "label"])
    function_post_values = np.array(dataframe["label"])
    try:
        coefficients, learning_score = linear_regression_learner._solve_regression_problem(
            regression_array, function_post_values)
        assert learning_score == 1
        assert len(coefficients) == 4

    except NotSafeActionError:
        pytest.fail()


def test_solve_function_linear_equations_raises_error_when_there_is_no_solution_with_the_correct_score_required(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "x": [0, 1, 1, 0],
        "y": [0, 0, 0, 0],
        "z": [-1, 0, -1, 1],
        "label": [4, -1, 9, 15 / 35]
    }
    dataframe = DataFrame(equation_matrix)
    regression_array = np.array(dataframe.loc[:, dataframe.columns != "label"])
    function_post_values = np.array(dataframe["label"])
    with pytest.raises(NotSafeActionError) as e:
        linear_regression_learner._solve_regression_problem(
            regression_array, function_post_values, allow_unsafe_learning=False)


def test_compute_non_constant_change_returns_correct_pddl_equation_form_for_variable_with_assignment_sign(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "x": [2, 1, 3, 0],
        "y": [3, -1, 2, 0],
        "z": [-1, 2, 4, 1],
        "label": [4, -1, 9, 15 / 35]
    }
    dataframe = DataFrame(equation_matrix)
    try:
        equation = linear_regression_learner._solve_safe_independent_equations("w", dataframe)
        assert equation is not None
        assert equation.startswith("(assign")

    except NotSafeActionError:
        pytest.fail()


def test_compute_non_constant_change_returns_correct_pddl_equation_form_for_variable_with_circular_assignment_statement(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "x": [2, 1, 3, 0],
        "y": [3, -1, 2, 0],
        "z": [-1, 2, 4, 1],
        "label": [4, -1, 9, 15 / 35]
    }
    dataframe = DataFrame(equation_matrix)
    try:
        equation = linear_regression_learner._solve_safe_independent_equations("x", dataframe)
        assert equation is not None
        assert equation.startswith("(increase")

    except NotSafeActionError:
        pytest.fail()


def test_construct_safe_conditional_effect_constructs_a_correct_conditional_effect(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1],
        "next_state_(x)": [21],
        "next_state_(y)": [32],
        "next_state_(z)": [-12],
    }
    dataframe = DataFrame(equation_matrix)
    result = linear_regression_learner._construct_multiple_safe_conditional_effects(dataframe)
    assert result is not None

    conditional_effects, disjunctive_preconditions = result
    assert len(conditional_effects) == 1
    conditional_effect = conditional_effects[0]
    assert len(conditional_effect.antecedents.root.operands) == 3
    assert len(conditional_effect.numeric_effects) == 3
    assert disjunctive_preconditions is not None
    assert disjunctive_preconditions.binary_operator == "and"
    assert len(disjunctive_preconditions.operands) == 3


def test_action_not_affects_fluent_returns_true_when_action_does_not_affect_fluent(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1],
        "label": [2],
    }
    dataframe = DataFrame(equation_matrix)
    result = linear_regression_learner._action_not_affects_fluent("(x)", dataframe)
    assert result


def test_action_not_affects_fluent_returns_false_when_action_does_affect_fluent(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1],
        "label": [21],
    }
    dataframe = DataFrame(equation_matrix)
    result = linear_regression_learner._action_not_affects_fluent("(x)", dataframe)
    assert not result


def test_combine_states_data_creates_correct_dataframe_with_correct_values(
        linear_regression_learner: LinearRegressionLearner):
    prev_state = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1],
    }
    next_state = {
        "(x)": [21],
        "(y)": [32],
        "(z)": [-12],
    }
    combined_state = linear_regression_learner._combine_states_data(prev_state, next_state)
    assert len(combined_state.columns) == 6


def test_solve_safe_independent_equations_returns_correct_pddl_equation_form_for_variable_with_assignment_sign(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "(x)": [2, 1, 3, 0],
        "(y)": [3, -1, 2, 0],
        "(z)": [-1, 2, 4, 1],
        "label": [21, 20, 22, 19]
    }
    dataframe = DataFrame(equation_matrix)
    constant_change = linear_regression_learner._solve_safe_independent_equations("(x)", dataframe)
    assert constant_change is not None
    assert constant_change == "(increase (x) 19.0)"


def test_solve_safe_independent_equations_equation_form_when_linear_equations_are_solvable(
        linear_regression_learner: LinearRegressionLearner):
    equation_matrix = {
        "(x)": [2, 1, 3, 0],
        "(y)": [3, -1, 2, 0],
        "(z)": [-1, 2, 4, 1],
        "label": [21, 23, 252, 129]
    }
    dataframe = DataFrame(equation_matrix)
    non_constant_change = linear_regression_learner._solve_safe_independent_equations("(x)", dataframe)
    assert non_constant_change is not None


def test_construct_assignment_equations_returns_correct_equations_for_all_fluents(
        linear_regression_learner: LinearRegressionLearner):
    pre_state_matrix = {
        "(x)": [2, 1, 3, 0],
        "(y)": [3, -1, 2, 0],
        "(z)": [-1, 2, 4, 1],
    }
    next_state_matrix = {
        "(x)": [4, -1, 9, 15 / 35],
        "(y)": [3, -1, 2, 0],
        "(z)": [1, 1, 1, 1]
    }
    result = linear_regression_learner.construct_assignment_equations(pre_state_matrix, next_state_matrix)
    assert result is not None
    assert len(result) == 2
    assert isinstance(result, set)
    effects = result
    assert len(effects) == 2
    pddl_effects = [eff.to_pddl() for eff in effects]
    assert "(assign (z ) 1.0)" in pddl_effects
    for eff in effects:
        print(eff.to_pddl())

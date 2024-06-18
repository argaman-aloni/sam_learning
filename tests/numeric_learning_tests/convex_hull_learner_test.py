import random

import numpy as np
import pytest
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Precondition, NumericalExpressionTree

from sam_learning.core.numeric_learning.convex_hull_learner import ConvexHullLearner

TEST_ACTION_NAME = "test_action"

random.seed(42)


@pytest.fixture
def convex_hull_learner() -> ConvexHullLearner:
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
        "w": PDDLFunction(name="w", signature={}),
    }
    return ConvexHullLearner(TEST_ACTION_NAME, domain_functions)


def test_construct_pddl_inequality_scheme_with_simple_2d_four_equations_returns_correct_representation(convex_hull_learner: ConvexHullLearner,):
    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 2))
    right_side_points = np.random.randint(10, size=4)

    inequalities = convex_hull_learner._construct_pddl_inequality_scheme(left_side_coefficients, right_side_points, ["(x)", "(y)"])
    assert len(inequalities) == 4
    for ineq in inequalities:
        assert ineq.startswith("(<=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_one_returns_a_single_precondition_with_and_condition(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2], "(y)": [3], "(z)": [-1]}
    dataframe = DataFrame(pre_state_data)
    equality_condition = convex_hull_learner._create_disjunctive_preconditions(dataframe, equality_conditions=[])
    assert len(equality_condition.operands) == 3
    assert equality_condition.binary_operator == "and"
    for operand in equality_condition.operands:
        assert operand.to_pddl().startswith("(=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_two_returns_a_single_precondition_with_or_condition(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2, 18], "(y)": [3, 32], "(z)": [-1, 17]}
    dataframe = DataFrame(pre_state_data)
    equality_condition = convex_hull_learner._create_disjunctive_preconditions(dataframe, equality_conditions=[])
    assert len(equality_condition.operands) == 2
    assert equality_condition.binary_operator == "or"
    for operand in equality_condition.operands:
        assert isinstance(operand, Precondition)
        assert operand.binary_operator == "and"
        assert len(operand.operands) == 3
        for sub_operand in operand.operands:
            assert sub_operand.to_pddl().startswith("(=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_one_and_equality_condition_creates_combined_condition_correctly(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [20], "(y)": [39], "(z)": [-12]}
    test_equality_conditions = ["(= (x ) 2)", "(= (y ) 3)", "(= (z ) -1)"]
    dataframe = DataFrame(pre_state_data)
    equality_condition = convex_hull_learner._create_disjunctive_preconditions(dataframe, equality_conditions=test_equality_conditions)
    assert len(equality_condition.operands) == 2 * len(test_equality_conditions)
    assert equality_condition.binary_operator == "and"
    for operand in equality_condition.operands:
        assert operand.to_pddl().startswith("(=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_two_and_equality_condition_creates_combined_condition_correctly(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2, 18], "(y)": [3, 32], "(z)": [-1, 17]}
    dataframe = DataFrame(pre_state_data)
    test_equality_conditions = ["(= (x ) 97)", "(= (y ) 21)"]
    result_condition = convex_hull_learner._create_disjunctive_preconditions(dataframe, equality_conditions=test_equality_conditions)
    assert len(result_condition.operands) == 3
    assert len([op for op in result_condition.operands if isinstance(op, NumericalExpressionTree)]) == 2
    assert len([op for op in result_condition.operands if isinstance(op, Precondition)]) == 1


def test_construct_single_dimension_inequalities_with_input_dataframe_of_size_one_returns_correct_inequalities(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2, 18], "(y)": [3, 32], "(z)": [-1, 17]}
    dataframe = DataFrame(pre_state_data)
    inequality_precondition = convex_hull_learner._construct_single_dimension_inequalities(dataframe["(x)"])
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 2
    inequalities = [op.to_pddl() for op in inequality_precondition.operands]
    assert "(>= (x ) 2.00)" in inequalities
    assert "(<= (x ) 18.00)" in inequalities


def test_construct_single_dimension_inequalities_with_input_lower_bound_equal_to_upper_bound_returns_correct_condition(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2, 2], "(y)": [3, 32], "(z)": [-1, 17]}
    dataframe = DataFrame(pre_state_data)
    inequality_precondition = convex_hull_learner._construct_single_dimension_inequalities(dataframe["(x)"])
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 1
    equality = inequality_precondition.operands.pop()
    assert "(= (x ) 2.00)" == equality.to_pddl()


def test_construct_safe_linear_inequalities_when_the_number_of_samples_is_one_creates_a_single_condition(convex_hull_learner: ConvexHullLearner,):
    pre_state_data = {"(x)": [2], "(y)": [3], "(z)": [-1]}
    inequality_precondition = convex_hull_learner.construct_safe_linear_inequalities(pre_state_data, None)
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 3
    assert {op.to_pddl() for op in inequality_precondition.operands} == {
        "(= (x ) 2.00)",
        "(= (y ) 3.00)",
        "(= (z ) -1.00)",
    }


def test_construct_safe_linear_inequalities_when_the_number_of_samples_is_two_creates_a_two_conditions_in_or_statement(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [2, 3], "(y)": [3, 5], "(z)": [-1, 7]}
    inequality_precondition = convex_hull_learner.construct_safe_linear_inequalities(pre_state_data, None)
    print(str(inequality_precondition))
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 2 + 2  # 2 for the CH and 2 for complementing the base
    assert all([isinstance(op, NumericalExpressionTree) for op in inequality_precondition.operands])
    assert len([cond.to_pddl() for cond in inequality_precondition.operands if cond.to_pddl().startswith("(=")]) == 2
    assert len([cond.to_pddl() for cond in inequality_precondition.operands if cond.to_pddl().startswith("(<=")]) == 2


def test_construct_safe_linear_inequalities_when_the_number_of_is_smaller_than_needed_dimensions_but_larger_than_two_creates_a_convex_hull_from_the_samples(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {
        "(x)": random.sample(range(100), 3),
        "(y)": random.sample(range(100), 3),
        "(z)": random.sample(range(100), 3),
    }
    inequality_precondition = convex_hull_learner.construct_safe_linear_inequalities(pre_state_data, None)
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 3 + 1  # 3 for the CH and 1 for complementing the base
    assert all([isinstance(op, NumericalExpressionTree) for op in inequality_precondition.operands])
    print(str(inequality_precondition))


def test_create_convex_hull_linear_inequalities_returns_correct_conditions_when_constant_is_observed(convex_hull_learner: ConvexHullLearner,):
    pre_state_data = {"(x)": [1, 0, 0], "(y)": [0, 1, 0], "(z)": [0, 0, 0]}
    pre_state_df = DataFrame(pre_state_data)
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._create_convex_hull_linear_inequalities(
        pre_state_df, display_mode=False
    )
    assert span_verification_conditions == ["(= (z) 0.0)"]
    print(span_verification_conditions)


def test_create_convex_hull_linear_inequalities_returns_correct_conditions_when_constant_is_observed_and_there_is_enough_data_to_create_convex_hull(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {"(x)": [1, 0, 0, 1], "(y)": [0, 1, 0, 1], "(z)": [0, 0, 1, 1]}
    pre_state_df = DataFrame(pre_state_data)
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._create_convex_hull_linear_inequalities(
        pre_state_df, display_mode=False
    )
    assert span_verification_conditions == []


def test_create_convex_hull_linear_inequalities_returns_correct_conditions_when_when_remaining_only_one_dimension_returns_correct_values(
    convex_hull_learner: ConvexHullLearner,
):
    pre_state_data = {
        "(x)": [2, 1, 3, 4],
        "(y)": [1, 1, 1, 1],
    }
    pre_state_df = DataFrame(pre_state_data)
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._create_convex_hull_linear_inequalities(
        pre_state_df, display_mode=False
    )
    assert span_verification_conditions == ["(= (- (y) 1) 0.0)"]
    assert coefficients == [[-1], [1]]
    assert transformed_vars == ["(* (- (x) 2) -1)"]
    assert border_point == [2, 1]


def test_create_convex_hull_linear_inequalities_returns_correct_conditions_with_paper_example(convex_hull_learner: ConvexHullLearner,):
    pre_state_data = {
        "(x)": [1, 0, 0],
        "(y)": [0, 1, 0],
        "(z)": [0, 0, 1],
    }
    pre_state_df = DataFrame(pre_state_data)
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._create_convex_hull_linear_inequalities(
        pre_state_df, display_mode=False
    )
    print(span_verification_conditions)
    assert len(span_verification_conditions) == 1  # One dimension was reduced.
    assert len(coefficients) == 3
    assert len(transformed_vars) == 2

import numpy as np
import pytest
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Precondition, NumericalExpressionTree

from sam_learning.core.convex_hull_learner import ConvexHullLearner

TEST_ACTION_NAME = 'test_action'


@pytest.fixture
def convex_hull_learner() -> ConvexHullLearner:
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
        "w": PDDLFunction(name="w", signature={})
    }
    return ConvexHullLearner(TEST_ACTION_NAME, domain_functions)


def test_filter_all_convex_hull_inconsistencies_does_not_maintain_columns_with_constant_values(
        convex_hull_learner: ConvexHullLearner):
    previous_state_df = DataFrame({
        "(x)": [1, 1, 1],
        "(y)": [1, 2, 3],
        "(z)": [12, 233, 23]
    })
    _, filtered_df = convex_hull_learner._filter_all_convex_hull_inconsistencies(previous_state_df)
    assert len(filtered_df.columns) == 2


def test_filter_all_convex_hull_inconsistencies_does_not_maintain_columns_with_linear_dependency(
        convex_hull_learner: ConvexHullLearner):
    previous_state_df = DataFrame({
        "(x)": [1, 1, 1],
        "(y)": [1, 2, 3],
        "(z)": [2, 2, 2]
    })
    _, filtered_df = convex_hull_learner._filter_all_convex_hull_inconsistencies(previous_state_df)
    assert len(filtered_df.columns) == 1


def test_construct_pddl_inequality_scheme_with_simple_2d_four_equations_returns_correct_representation(
        convex_hull_learner: ConvexHullLearner):
    np.random.seed(42)
    left_side_coefficients = np.random.randint(10, size=(4, 2))
    right_side_points = np.random.randint(10, size=4)

    inequalities = convex_hull_learner._construct_pddl_inequality_scheme(
        left_side_coefficients, right_side_points, ["(x)", "(y)"])
    assert len(inequalities) == 4
    for ineq in inequalities:
        assert ineq.startswith("(<=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_one_returns_a_single_precondition_with_and_condition(
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1]
    }
    dataframe = DataFrame(pre_state_data)
    equality_condition = convex_hull_learner._create_disjunctive_preconditions(dataframe, equality_conditions=[])
    assert len(equality_condition.operands) == 3
    assert equality_condition.binary_operator == "and"
    for operand in equality_condition.operands:
        assert operand.to_pddl().startswith("(=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_two_returns_a_single_precondition_with_or_condition(
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2, 18],
        "(y)": [3, 32],
        "(z)": [-1, 17]
    }
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
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2],
        "(y)": [3],
        "(z)": [-1]
    }
    test_equality_conditions = ["(= (x ) 2)", "(= (y ) 3)", "(= (z ) -1)"]
    dataframe = DataFrame(pre_state_data)
    equality_condition = convex_hull_learner._create_disjunctive_preconditions(
        dataframe, equality_conditions=test_equality_conditions)
    assert len(equality_condition.operands) == len(test_equality_conditions)
    assert equality_condition.binary_operator == "and"
    for operand in equality_condition.operands:
        assert operand.to_pddl().startswith("(=")


def test_create_disjunctive_preconditions_with_input_dataframe_of_size_two_and_equality_condition_creates_combined_condition_correctly(
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2, 18],
        "(y)": [3, 32],
        "(z)": [-1, 17]
    }
    dataframe = DataFrame(pre_state_data)
    test_equality_conditions = ["(= (x ) 97)", "(= (y ) 21)"]
    result_condition = convex_hull_learner._create_disjunctive_preconditions(
        dataframe, equality_conditions=test_equality_conditions)
    assert len(result_condition.operands) == 3
    assert len([op for op in result_condition.operands if isinstance(op, NumericalExpressionTree)]) == 2
    assert len([op for op in result_condition.operands if isinstance(op, Precondition)]) == 1


def test_construct_single_dimension_inequalities_with_input_dataframe_of_size_one_returns_correct_inequalities(
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2, 18],
        "(y)": [3, 32],
        "(z)": [-1, 17]
    }
    dataframe = DataFrame(pre_state_data)
    inequality_precondition = convex_hull_learner._construct_single_dimension_inequalities(dataframe["(x)"])
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 2
    inequalities = [op.to_pddl() for op in inequality_precondition.operands]
    assert "(>= (x ) 2.0)" in inequalities
    assert "(<= (x ) 18.0)" in inequalities


def test_construct_single_dimension_inequalities_with_input_lower_bound_equal_to_upper_bound_returns_correct_condition(
        convex_hull_learner: ConvexHullLearner):
    pre_state_data = {
        "(x)": [2, 2],
        "(y)": [3, 32],
        "(z)": [-1, 17]
    }
    dataframe = DataFrame(pre_state_data)
    inequality_precondition = convex_hull_learner._construct_single_dimension_inequalities(dataframe["(x)"])
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 1
    equality = inequality_precondition.operands.pop()
    assert "(= (x ) 2.0)" == equality.to_pddl()

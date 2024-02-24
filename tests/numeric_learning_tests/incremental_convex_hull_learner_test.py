"""Module tests the incremental convex hull learner."""
import random
import time

import pytest
from pddl_plus_parser.models import PDDLFunction

from sam_learning.core.numeric_learning.incremental_convex_hull_learner import IncrementalConvexHullLearner
from sam_learning.core.numeric_learning.numeric_utils import display_convex_hull

TEST_ACTION_NAME = "test_action"

random.seed(42)


@pytest.fixture
def convex_hull_learner() -> IncrementalConvexHullLearner:
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
    }
    return IncrementalConvexHullLearner(TEST_ACTION_NAME, domain_functions)


def test_creating_the_convex_hull_learner_creates_the_dataframe_with_the_correct_columns_and_zero_rows(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    assert convex_hull_learner.data.shape == (0, 3)
    assert convex_hull_learner.data.columns.tolist() == ["(x )", "(y )", "(z )"]


def test_add_new_point_when_adding_a_point_for_the_first_time_only_adds_it_to_the_class_data(convex_hull_learner):
    test_sample = {"(x )": 1, "(y )": 2, "(z )": 3}
    convex_hull_learner.add_new_point(test_sample)
    assert convex_hull_learner.data.shape == (1, 3)
    assert convex_hull_learner.data.iloc[0].to_dict() == test_sample
    assert convex_hull_learner._convex_hull is None
    assert convex_hull_learner._gsp_base is None
    assert convex_hull_learner._complementary_base is None
    assert not convex_hull_learner._spanning_standard_base


def test_add_new_point_when_adding_a_point_for_the_second_time_does_not_add_it_again(convex_hull_learner):
    test_sample = {"(x )": 1, "(y )": 2, "(z )": 3}
    convex_hull_learner.add_new_point(test_sample)
    convex_hull_learner.add_new_point(test_sample)
    assert convex_hull_learner.data.shape == (1, 3)


def test_add_new_point_when_adding_two_points_creates_a_base_for_the_points_and_an_orthonormal_complementary_base(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 1, "(y )": 2, "(z )": 3}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 2, "(y )": 3, "(z )": 4}
    convex_hull_learner.add_new_point(second_sample)
    assert convex_hull_learner.data.shape == (2, 3)
    assert convex_hull_learner.data.iloc[0].to_dict() == first_sample
    assert convex_hull_learner.data.iloc[1].to_dict() == second_sample
    assert convex_hull_learner._convex_hull is None
    assert convex_hull_learner._gsp_base is not None
    assert len(convex_hull_learner._complementary_base) == 2


def test_add_new_point_when_adding_three_points_with_that_create_larger_base_changes_the_base_created_by_the_algorithm(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    assert convex_hull_learner.data.shape == (2, 3)
    assert convex_hull_learner.data.iloc[0].to_dict() == first_sample
    assert convex_hull_learner.data.iloc[1].to_dict() == second_sample
    normal_base = convex_hull_learner._gsp_base.copy()
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    assert convex_hull_learner.data.shape == (3, 3)
    assert len(convex_hull_learner._gsp_base) == len(normal_base) + 1
    assert convex_hull_learner._gsp_base[0] == normal_base[0]


def test_add_new_point_when_adding_three_points_when_gsp_returns_two_dimensions_sets_the_convex_hull_with_enough_data(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    assert convex_hull_learner._convex_hull is not None
    assert len(convex_hull_learner._convex_hull.points) == 3
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)


def test_add_new_point_when_adding_four_points_and_the_last_one_is_linearly_dependent_does_not_change_the_base(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    normal_base = convex_hull_learner._gsp_base.copy()
    fourth_sample = {"(x )": 2, "(y )": 2, "(z )": 0}
    convex_hull_learner.add_new_point(fourth_sample)
    assert normal_base == convex_hull_learner._gsp_base


def test_add_new_point_when_adding_four_points_and_the_last_one_is_linearly_dependent_adds_the_point_to_the_convex_hull(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    fourth_sample = {"(x )": 2, "(y )": 2, "(z )": 0}
    convex_hull_learner.add_new_point(fourth_sample)
    assert len(convex_hull_learner._convex_hull.points) == 4
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)


def test_add_new_point_when_adding_a_point_with_feature_not_existing_in_previous_sample_does_not_add_new_feature_to_dataset(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 1, "(w )": 1}
    convex_hull_learner.add_new_point(second_sample)
    assert convex_hull_learner.data.shape[0] == 2
    assert convex_hull_learner.data.shape[1] == 3


def test_add_new_point_when_adding_four_points_that_span_the_entire_space_returns_standard_basis(convex_hull_learner):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    fourth_sample = {"(x )": 0, "(y )": 0, "(z )": 1}
    convex_hull_learner.add_new_point(fourth_sample)
    assert len(convex_hull_learner._convex_hull.points) == 4
    assert convex_hull_learner._gsp_base == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)


def test_incremental_create_ch_inequalities_with_one_dimension_returns_min_max_conditions(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    (
        coefficients,
        border_point,
        transformed_vars,
        span_verification_conditions,
    ) = convex_hull_learner._incremental_create_ch_inequalities(display_mode=True)
    assert coefficients == [[-1], [1]]
    assert border_point == [0, 1.4142]
    assert transformed_vars == ["(+ (* (x ) 0.7071) (* (- (y ) 1) -0.7071))"]
    assert set(span_verification_conditions) == {"(= (z ) 0.0)", "(= (+ (* (x ) 0.7071) (* (- (y ) 1) 0.7071)) 0.0)"}


def test_incremental_create_ch_inequalities_with_one_dimension_returns_min_max_conditions_and_correct_complementary_conditions(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 3, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    fourth_sample = {"(x )": 4, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(fourth_sample)
    (
        coefficients,
        border_point,
        transformed_vars,
        span_verification_conditions,
    ) = convex_hull_learner._incremental_create_ch_inequalities(display_mode=True)
    assert coefficients == [[-1], [1]]
    assert border_point == [0, 4]
    assert set(span_verification_conditions) == {"(= (z ) 0.0)", "(= (- (y ) 1) 0.0)"}


def test_incremental_create_ch_inequalities_with_point_spanning_standard_base_returns_correct_convex_hull_with_no_verification_conditions(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    third_sample = {"(x )": 1, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(third_sample)
    fourth_sample = {"(x )": 0, "(y )": 0, "(z )": 1}
    convex_hull_learner.add_new_point(fourth_sample)
    (
        coefficients,
        border_point,
        transformed_vars,
        span_verification_conditions,
    ) = convex_hull_learner._incremental_create_ch_inequalities(display_mode=True)
    assert len(coefficients) == 4
    assert len(border_point) == 4
    assert span_verification_conditions == []


def test_construct_safe_linear_inequalities_when_the_number_of_samples_is_one_creates_a_single_condition(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 2, "(y )": 3, "(z )": -1}
    convex_hull_learner.add_new_point(first_sample)
    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities(None)
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 3
    assert {op.to_pddl() for op in inequality_precondition.operands} == {
        "(= (x ) 2.0)",
        "(= (y ) 3.0)",
        "(= (z ) -1.0)",
    }


def test_construct_safe_linear_inequalities_when_creating_convex_hull_with_large_number_of_samples_does_not_fail(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    for _ in range(100):
        convex_hull_learner.add_new_point(
            {"(x )": random.uniform(-100, 100), "(y )": random.uniform(-100, 100), "(z )": random.uniform(-100, 100)}
        )

    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities(None)
    assert inequality_precondition.binary_operator == "and"
    assert len(convex_hull_learner._convex_hull.points) == 100
    assert convex_hull_learner._gsp_base == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert convex_hull_learner._complementary_base == []
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)


def test_construct_safe_linear_inequalities_when_adding_extra_large_number_of_samples_does_not_take_extremely_long_to_calculate(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    start_time = time.time()
    for _ in range(1000):
        convex_hull_learner.add_new_point(
            {"(x )": random.uniform(-100, 100), "(y )": random.uniform(-100, 100), "(z )": random.uniform(-100, 100)}
        )

    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities(None)
    end_time = time.time()
    assert end_time - start_time < 5 * 60
    assert inequality_precondition.binary_operator == "and"
    assert len(convex_hull_learner._convex_hull.points) == 1000
    assert convex_hull_learner._gsp_base == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert convex_hull_learner._complementary_base == []
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)

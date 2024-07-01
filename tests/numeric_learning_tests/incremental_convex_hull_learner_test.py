"""Module tests the incremental convex hull learner."""
import random
import time

import numpy
import numpy as np
import pytest
import pandas as pd
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import PDDLFunction, Domain

from sam_learning.core import VocabularyCreator
from sam_learning.core.numeric_learning.incremental_convex_hull_learner import IncrementalConvexHullLearner
from sam_learning.core.numeric_learning.numeric_utils import display_convex_hull, create_monomials, create_polynomial_string
from tests.consts import FARMLAND_EXAMPLES_PATH, FARMLAND_DOMAIN_PATH, FARMLAND_PAPER_EXAMPLES_PATH

TEST_ACTION_NAME = "test_action"

random.seed(42)


@pytest.fixture
def farmland_domain() -> Domain:
    return DomainParser(FARMLAND_DOMAIN_PATH, partial_parsing=True).parse_domain()


@pytest.fixture
def convex_hull_learner() -> IncrementalConvexHullLearner:
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
    }
    return IncrementalConvexHullLearner(TEST_ACTION_NAME, domain_functions)


@pytest.fixture
def polynomial_convex_hull_learner() -> IncrementalConvexHullLearner:
    domain_functions = {
        "(x )": PDDLFunction(name="x", signature={}),
        "(y )": PDDLFunction(name="y", signature={}),
        "(z )": PDDLFunction(name="z", signature={}),
    }
    return IncrementalConvexHullLearner(TEST_ACTION_NAME, domain_functions, polynom_degree=1)


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


def test_add_new_point_when_adding_one_point_at_a_time_returns_correct_values_for_convex_hull_points_without_subtracting_first_sample(
    farmland_domain: Domain,
):
    dataframe = pd.read_csv(FARMLAND_EXAMPLES_PATH)
    learner = IncrementalConvexHullLearner(TEST_ACTION_NAME, domain_functions=farmland_domain.functions)
    for index, (_, row) in enumerate(dataframe.iterrows()):
        if learner._spanning_standard_base:
            assert (learner._convex_hull.points == (dataframe[:index].to_numpy(dtype=numpy.float32))).all()

        point_data = {key: row[key] for key in dataframe.columns.tolist()}
        learner.add_new_point(point_data)


def test_add_new_point_when_adding_a_point_with_feature_not_existing_in_previous_sample_does_not_add_new_feature_to_dataset(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 1, "(w )": 1}
    convex_hull_learner.add_new_point(second_sample)
    assert convex_hull_learner.data.shape[0] == 2
    assert convex_hull_learner.data.shape[1] == 3


def test_add_new_point_when_adding_multiple_points_with_single_dimension_does_not_try_to_create_a_convex_hull(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    try:
        for i in range(10):
            sample = {"(x )": i}
            convex_hull_learner.add_new_point(sample)
    except Exception as e:
        pytest.fail(f"Exception was raised: {e}")


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


def test_incremental_create_ch_inequalities_with_one_dimension_returns_min_max_conditions(convex_hull_learner: IncrementalConvexHullLearner,):
    first_sample = {"(x )": 0, "(y )": 1, "(z )": 0}
    convex_hull_learner.add_new_point(first_sample)
    second_sample = {"(x )": 1, "(y )": 0, "(z )": 0}
    convex_hull_learner.add_new_point(second_sample)
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._incremental_create_ch_inequalities(
        display_mode=True
    )
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
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._incremental_create_ch_inequalities(
        display_mode=True
    )
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
    (coefficients, border_point, transformed_vars, span_verification_conditions,) = convex_hull_learner._incremental_create_ch_inequalities(
        display_mode=True
    )
    assert len(coefficients) == 4
    assert len(border_point) == 4
    assert span_verification_conditions is None


def test_construct_safe_linear_inequalities_when_the_number_of_samples_is_one_creates_a_single_condition(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    first_sample = {"(x )": 2, "(y )": 3, "(z )": -1}
    convex_hull_learner.add_new_point(first_sample)
    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities()
    assert inequality_precondition.binary_operator == "and"
    assert len(inequality_precondition.operands) == 3
    assert {op.to_pddl() for op in inequality_precondition.operands} == {
        "(= (x ) 2)",
        "(= (y ) 3)",
        "(= (z ) -1)",
    }


def test_construct_safe_linear_inequalities_when_creating_convex_hull_with_large_number_of_samples_does_not_fail(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    for _ in range(100):
        convex_hull_learner.add_new_point({"(x )": random.uniform(-100, 100), "(y )": random.uniform(-100, 100), "(z )": random.uniform(-100, 100)})

    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities()
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
        convex_hull_learner.add_new_point({"(x )": random.uniform(-100, 100), "(y )": random.uniform(-100, 100), "(z )": random.uniform(-100, 100)})

    inequality_precondition = convex_hull_learner.construct_convex_hull_inequalities()
    end_time = time.time()
    assert end_time - start_time < 5 * 60
    assert inequality_precondition.binary_operator == "and"
    assert len(convex_hull_learner._convex_hull.points) == 1000
    assert convex_hull_learner._gsp_base == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert convex_hull_learner._complementary_base == []
    display_convex_hull(TEST_ACTION_NAME, True, convex_hull_learner._convex_hull)


def test_incremental_create_ch_inequalities_creates_convex_hull_with_all_variables_in_the_vectors_including_polynoms(
    polynomial_convex_hull_learner: IncrementalConvexHullLearner,
):
    k = 50
    numbers = list(range(100))
    state_matrix = {
        "(x )": random.sample(numbers, k=k),
        "(y )": random.sample(numbers, k=k),
        "(z )": random.sample(numbers, k=k),
    }
    monomials = create_monomials(list(state_matrix.keys()), polynom_degree=1)
    for i in range(k):
        polynomial_convex_hull_learner.add_new_point(
            {create_polynomial_string(monomial): np.prod([state_matrix[name][i] for name in monomial]) for monomial in monomials},
        )

    A, _, _, _ = polynomial_convex_hull_learner._incremental_create_ch_inequalities()
    assert all([len(item) == 6 for item in A])


def test_construct_convex_hull_inequalities_when_given_polynomial_inequalities_returns_convex_hull_with_all_points_as_conditions(
    polynomial_convex_hull_learner: IncrementalConvexHullLearner,
):
    k = 50
    numbers = list(range(100))
    state_matrix = {
        "(x )": random.sample(numbers, k=k),
        "(y )": random.sample(numbers, k=k),
        "(z )": random.sample(numbers, k=k),
    }
    monomials = create_monomials(list(state_matrix.keys()), polynom_degree=1)
    for i in range(k):
        polynomial_convex_hull_learner.add_new_point(
            {create_polynomial_string(monomial): np.prod([state_matrix[name][i] for name in monomial]) for monomial in monomials},
        )

    precondition = polynomial_convex_hull_learner.construct_convex_hull_inequalities()
    print(str(precondition))


def test_construct_convex_hull_inequalities_when_adding_multiple_points_with_single_dimension_creates_min_max_conditions_and_does_not_raise_an_error(
    convex_hull_learner: IncrementalConvexHullLearner,
):
    try:
        for i in range(10):
            sample = {"(x )": i}
            convex_hull_learner.add_new_point(sample)

        precondition = convex_hull_learner.construct_convex_hull_inequalities()
        assert precondition.binary_operator == "and"
        assert len(precondition.operands) == 2
        assert {op.to_pddl() for op in precondition.operands} == {"(>= (x ) 0)", "(<= (x ) 9)"}

    except Exception as e:
        pytest.fail(f"Exception was raised: {e}")


def test_construct_convex_hull_inequalities_when_spanning_standard_basis_returns_correct_number_of_equations(farmland_domain: Domain):
    dataframe = pd.read_csv(FARMLAND_EXAMPLES_PATH)
    vocabulary_creator = VocabularyCreator()
    possible_bounded_functions = vocabulary_creator.create_lifted_functions_vocabulary(
        domain=farmland_domain, possible_parameters=farmland_domain.actions["move-slow"].signature
    )
    learner = IncrementalConvexHullLearner("move-slow", domain_functions=possible_bounded_functions)
    for index, (_, row) in enumerate(dataframe.iterrows()):
        if learner._spanning_standard_base:
            assert (learner._convex_hull.points == (dataframe[:index].to_numpy(dtype=numpy.float32))).all()

        point_data = {key: row[key] for key in dataframe.columns.tolist()}
        learner.add_new_point(point_data)

    precondition = learner.construct_convex_hull_inequalities()
    assert len(precondition.operands) == 7


def test_construct_convex_hull_inequalities_when_given_too_few_examples_returns_lower_dimensional_convex_hull_with_real_dataset_example(
    farmland_domain: Domain,
):
    dataframe = pd.read_csv(FARMLAND_PAPER_EXAMPLES_PATH)
    vocabulary_creator = VocabularyCreator()
    possible_bounded_functions = vocabulary_creator.create_lifted_functions_vocabulary(
        domain=farmland_domain, possible_parameters=farmland_domain.actions["move-slow"].signature
    )
    learner = IncrementalConvexHullLearner("move-slow", domain_functions=possible_bounded_functions)
    for index, (_, row) in enumerate(dataframe.iterrows()):
        point_data = {key: row[key] for key in dataframe.columns.tolist()}
        learner.add_new_point(point_data)

    assert learner._gsp_base is not None
    assert learner._complementary_base is not None
    precondition = learner.construct_convex_hull_inequalities()
    print(str(precondition))

# File: tests\test_incremental_svm_learner.py
from typing import Dict

import numpy as np
import csv
import pytest
from pddl_plus_parser.models import PDDLFunction, Precondition

from sam_learning.core.online_learning.incremental_svm_learner import IncrementalSVMLearner
from tests.consts import TWO_SIDES_OF_BOX_PATH, CLOSE_TO_BOX_PATH, CLOSE_TO_LINEAR_CONDITION_PATH

FUNC_NAMES = ["(x )", "(y )", "(z )", "(w )"]


@pytest.fixture
def domain_functions():
    return {
        "(x )": PDDLFunction(name="x", signature={}),
        "(y )": PDDLFunction(name="y", signature={}),
        "(z )": PDDLFunction(name="z", signature={}),
        "(w )": PDDLFunction(name="w", signature={}),
    }


@pytest.fixture
def two_dim_svm_learner():
    return IncrementalSVMLearner(
        action_name="test_action",
        domain_functions={"(x )": PDDLFunction(name="x", signature={}), "(y )": PDDLFunction(name="y", signature={}),},
        polynom_degree=0,
    )


@pytest.fixture
def svm_learner(domain_functions: Dict[str, PDDLFunction]):
    return IncrementalSVMLearner(action_name="test_action", domain_functions=domain_functions, polynom_degree=0)


def test_initialization_of_learner_creates_correct_columns(svm_learner: IncrementalSVMLearner):
    expected_columns = ["(x )", "(y )", "(z )", "(w )", "label"]
    assert list(svm_learner.data.columns) == expected_columns


def test_add_new_point_adds_single_sample_to_data(svm_learner: IncrementalSVMLearner):
    func_values = [1.0, 0.5, 0.0, 0.0]
    sample_point = {}
    for val, name in zip(func_values, FUNC_NAMES):
        sample_point[name] = PDDLFunction(name=name, signature={})
        sample_point[name].set_value(val)

    svm_learner.add_new_point(point=sample_point, is_successful=True)

    assert len(svm_learner.data) == 1
    assert svm_learner.data.iloc[0]["(x )"] == 1.0
    assert svm_learner.data.iloc[0]["(y )"] == 0.5
    assert svm_learner.data.iloc[0]["(z )"] == 0.0
    assert svm_learner.data.iloc[0]["(w )"] == 0.0
    assert svm_learner.data.iloc[0]["label"] == 1


def test_add_new_point_with_failure_label_successfully(svm_learner: IncrementalSVMLearner):
    func_values = [1.0, 0.5, 0.0, 0.0]
    sample_point = {}
    for val, name in zip(func_values, FUNC_NAMES):
        sample_point[name] = PDDLFunction(name=name, signature={})
        sample_point[name].set_value(val)

    svm_learner.add_new_point(point=sample_point, is_successful=False)

    assert len(svm_learner.data) == 1
    assert svm_learner.data.iloc[0]["label"] == -1


def test_create_svm_conditions_returns_empty_preconditions_when_no_data_was_added(svm_learner: IncrementalSVMLearner):
    result = svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    assert len(result.operands) == 0  # No conditions should exist for empty data


def test_create_svm_conditions_when_given_multiple_samples_returns_moderatly_accurate_conditions(svm_learner: IncrementalSVMLearner):
    N = 200
    x = [np.random.randint(0, 100) for _ in range(N)]
    y = [np.random.randint(0, 100) for _ in range(N)]
    z = [np.random.randint(0, 100) for _ in range(N)]
    w = [0.0 for _ in range(N)]
    for i in range(N):
        label = x[i] >= 2 and y[i] >= 4 and z[i] >= 1
        point = {name: PDDLFunction(name=name, signature={}) for name in FUNC_NAMES}
        point["(x )"].set_value(x[i])
        point["(y )"].set_value(y[i])
        point["(z )"].set_value(z[i])
        point["(w )"].set_value(w[i])
        svm_learner.add_new_point(point=point, is_successful=label)

    result = svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))


def test_create_svm_conditions_when_given_multiple_samples_returns_moderatly_accurate_conditions_when_conditions_are_circular(
    domain_functions: Dict[str, PDDLFunction]
):
    learner = IncrementalSVMLearner(action_name="test_action", domain_functions=domain_functions, polynom_degree=2)
    N = 200
    x = [np.random.randint(-100, 100) for _ in range(N)]
    y = [np.random.randint(-100, 100) for _ in range(N)]
    z = [0.0 for _ in range(N)]
    w = [0.0 for _ in range(N)]
    for i in range(N):
        label = (x[i] ^ 2) + (y[i] ^ 2) <= 9
        point = {name: PDDLFunction(name=name, signature={}) for name in FUNC_NAMES}
        point["(x )"].set_value(x[i])
        point["(y )"].set_value(y[i])
        point["(z )"].set_value(z[i])
        point["(w )"].set_value(w[i])
        learner.add_new_point(point=point, is_successful=label)

    result = learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))


def test_create_svm_conditions_when_given_multiple_samples_returns_moderatly_accurate_conditions_when_conditions_are_box(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    N = 500
    x = [np.random.randint(-10, 10 + 1) for _ in range(N // 2)] + [np.random.randint(-100, 100 + 1) for _ in range(N // 2)]
    y = [np.random.randint(-10, 10 + 1) for _ in range(N // 2)] + [np.random.randint(-100, 100 + 1) for _ in range(N // 2)]
    for i in range(N):
        label = -10 <= x[i] <= 10 and -10 <= y[i] <= 10
        point = {name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]}
        point["(x )"].set_value(x[i])
        point["(y )"].set_value(y[i])
        two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert len(result.operands) >=  2 + 1 # at least number of dimensions + 1

def test_create_svm_conditions_when_given_hundredth_positive_samples_as_box(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    positive_propotion = 0.01
    N = 1000
    positive_count = int(N * positive_propotion)
    negative_count = N - positive_count

    x = [np.random.randint(-10, 10 + 1) for _ in range(positive_count)] + [np.random.randint(-100, 100 + 1) for _ in range(negative_count)]
    y = [np.random.randint(-10, 10 + 1) for _ in range(positive_count)] + [np.random.randint(-100, 100 + 1) for _ in range(negative_count)]
    for i in range(N):
        label = -10 <= x[i] <= 10 and -10 <= y[i] <= 10
        point = {name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]}
        point["(x )"].set_value(x[i])
        point["(y )"].set_value(y[i])
        two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert len(result.operands) >=  2 + 1 # at least number of dimensions + 1

def test_create_svm_conditions_when_negative_points_are_close_to_the_condition_box(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    with open(CLOSE_TO_BOX_PATH, "r") as csvfile:

        reader = csv.reader(csvfile)
        
        next(reader) # Skip the header   
        for row in reader: # For each row in the CSV file
            point = {
                name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]
            }
            point["(x )"].set_value(float(row[0]))
            point["(y )"].set_value(float(row[1]))
            label = row[2] == 'True'
            two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert len(result.operands) >=  2 + 1 # at least number of dimensions + 1


def test_create_svm_conditions_when_given_negative_points_from_two_sides_of_the_condition_box(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    with open(TWO_SIDES_OF_BOX_PATH, "r") as csvfile:

        reader = csv.reader(csvfile)
        
        next(reader) # Skip the header   
        for row in reader: # For each row in the CSV file
            point = {
                name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]
            }
            point["(x )"].set_value(float(row[0]))
            point["(y )"].set_value(float(row[1]))
            label = row[2] == 'True'
            two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert len(result.operands) ==  2 # two sides of the box

def test_create_svm_conditions_when_given_multiple_samples_returns_moderatly_accurate_conditions_when_conditions_are_linear(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    N = 500
    x = [np.random.randint(-100, 100 + 1) for _ in range(N)]
    y = [np.random.randint(-100, 100 + 1) for _ in range(N)]
    for i in range(N):
        label = x[i] > 5 + y[i]
        point = {name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]}
        point["(x )"].set_value(x[i])
        point["(y )"].set_value(y[i])
        two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert len(result.operands) >=  1 # at least one condition

def test_create_svm_conditions_when_negative_points_are_close_to_the_linear_condition(
    two_dim_svm_learner: IncrementalSVMLearner,
):
    with open(CLOSE_TO_LINEAR_CONDITION_PATH, "r") as csvfile:

        reader = csv.reader(csvfile)
        
        next(reader) # Skip the header   
        for row in reader: # For each row in the CSV file
            point = {
                name: PDDLFunction(name=name, signature={}) for name in ["(x )", "(y )"]
            }
            point["(x )"].set_value(float(row[0]))
            point["(y )"].set_value(float(row[1]))
            label = row[2] == 'True'
            two_dim_svm_learner.add_new_point(point=point, is_successful=label)

    result = two_dim_svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))
    assert 1 <= len(result.operands) <=  2 # Have 4 planes when 2 of them are subset of the other 2

# File: tests\test_incremental_svm_learner.py
import numpy as np
import pandas as pd
import pytest
from pddl_plus_parser.models import PDDLFunction, Precondition
from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.online_learning.incremental_svm_learner import IncrementalSVMLearner


@pytest.fixture
def svm_learner():
    domain_functions = {
        "x": PDDLFunction(name="x", signature={}),
        "y": PDDLFunction(name="y", signature={}),
        "z": PDDLFunction(name="z", signature={}),
        "w": PDDLFunction(name="w", signature={}),
    }
    return IncrementalSVMLearner(action_name="test_action", domain_functions=domain_functions, polynom_degree=0)


def test_initialization_of_learner_creates_correct_columns(svm_learner: IncrementalSVMLearner):
    expected_columns = ["(x )", "(y )", "(z )", "(w )", "label"]
    assert list(svm_learner.data.columns) == expected_columns


def test_add_new_point_adds_single_sample_to_data(svm_learner: IncrementalSVMLearner):
    sample_point = {"(x )": 1.0, "(y )": 0.5, "(z )": 0.0, "(w )": 0.0}
    svm_learner.add_new_point(point=sample_point, is_successful=True)

    assert len(svm_learner.data) == 1
    assert svm_learner.data.iloc[0]["(x )"] == 1.0
    assert svm_learner.data.iloc[0]["(y )"] == 0.5
    assert svm_learner.data.iloc[0]["(z )"] == 0.0
    assert svm_learner.data.iloc[0]["(w )"] == 0.0
    assert svm_learner.data.iloc[0]["label"] == 1


def test_add_new_point_with_failure_label_successfully(svm_learner: IncrementalSVMLearner):
    sample_point = {"(x )": 1.0, "(y )": 0.5, "(z )": 0.0, "(w )": 0.0}
    svm_learner.add_new_point(point=sample_point, is_successful=False)

    assert len(svm_learner.data) == 1
    assert svm_learner.data.iloc[0]["label"] == 0


def test_create_svm_conditions_returns_empty_preconditions_when_no_data_was_added(svm_learner: IncrementalSVMLearner):
    result = svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    assert len(result.operands) == 0  # No conditions should exist for empty data


def test_create_svm_conditions_when_given_multiple_samples_returns_moderatly_accurate_conditions(svm_learner: IncrementalSVMLearner):
    N = 200
    x = [np.random.randint(0, 100) for _ in range(N)]
    y = [np.random.randint(0, 100) for _ in range(N)]
    z = [np.random.randint(0, 100) for _ in range(N)]
    for i in range(N):
        label = x[i] >= 2 and y[i] >= 4 and z[i] >= 1
        svm_learner.add_new_point(point={"(x )": x[i], "(y )": y[i], "(z )": z[i], "(w )": 0.0}, is_successful=label)

    result = svm_learner.construct_linear_inequalities()
    assert isinstance(result, Precondition)
    print(str(result))

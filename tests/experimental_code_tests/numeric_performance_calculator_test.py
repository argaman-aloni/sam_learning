"""Module test for the numeric performance calculation."""
import os
from pathlib import Path

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation, ActionCall
from pytest import fixture

from experiments import NumericPerformanceCalculator
from statistics.performance_calculation_utils import _ground_executed_action
from tests.consts import SAILING_EXPECTED_DOMAIN_PATH, SAILING_PROBLEM_PATH, SAILING_TRAJECTORY_PATH, \
    SAILING_LEARNED_DOMAIN_PATH
from utilities import LearningAlgorithmType

TEST_WORKING_DIRECTORY = Path(os.getcwd())


@fixture()
def sailing_expected_domain() -> Domain:
    domain_parser = DomainParser(SAILING_EXPECTED_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def sailing_learned_domain() -> Domain:
    domain_parser = DomainParser(SAILING_LEARNED_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def sailing_problem(sailing_expected_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SAILING_PROBLEM_PATH, domain=sailing_expected_domain).parse_problem()


@fixture()
def sailing_expected_observation(sailing_expected_domain: Domain, sailing_problem: Problem) -> Observation:
    return TrajectoryParser(sailing_expected_domain, sailing_problem).parse_trajectory(SAILING_TRAJECTORY_PATH)


@fixture()
def numeric_performance_calculator(sailing_expected_domain: Domain,
                                   sailing_expected_observation: Observation) -> NumericPerformanceCalculator:
    return NumericPerformanceCalculator(model_domain=sailing_expected_domain,
                                        observations=[sailing_expected_observation],
                                        working_directory_path=TEST_WORKING_DIRECTORY,
                                        learning_algorithm=LearningAlgorithmType.numeric_sam)


def test_ground_tested_operator_is_able_to_ground_properly_with_negative_preconditions(
        numeric_performance_calculator: NumericPerformanceCalculator, sailing_learned_domain: Domain):
    test_action_call = ActionCall(name="save_person", grounded_parameters=["b3", "p2"])
    try:
        _ground_executed_action(action_call=test_action_call, learned_domain=sailing_learned_domain)
    except Exception as e:
        pytest.fail(f"Failed to ground the tested action properly. Exception: {e}")


def test_calculate_performance_is_able_to_ground_properly_with_negative_preconditions(
        numeric_performance_calculator: NumericPerformanceCalculator, sailing_learned_domain: Domain):
    try:
        numeric_performance_calculator.calculate_performance(learned_domain_path=SAILING_LEARNED_DOMAIN_PATH,
                                                             num_used_observations=1)
    except Exception as e:
        pytest.fail(f"Failed to ground the tested action properly. Exception: {e}")

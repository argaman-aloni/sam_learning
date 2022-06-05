"""Module test for the precision and recall calculation part."""
import pytest
from pddl_plus_parser.lisp_parsers import DomainParser
from pytest import fixture
from pddl_plus_parser.models import Predicate, Domain

from experiments import calculate_number_true_positives, calculate_number_false_negatives, \
    calculate_number_false_positives, \
    calculate_precision, calculate_recall, PrecisionRecallCalculator
from tests.consts import LOCATION_TYPE, OBJECT_TYPE, NUMERIC_DOMAIN_PATH, TRUCK_TYPE
from sam_learning.core import LearnerAction

TEST_SYMMETRIC_LEARNED_PRECONDITIONS = [
    Predicate(
        name="at",
        signature={"?truck": OBJECT_TYPE, "?loc": LOCATION_TYPE}),
    Predicate(
        name="at", signature={"?obj": OBJECT_TYPE, "?loc": LOCATION_TYPE}),
]

TEST_SYMMETRIC_EXPECTED_PRECONDITIONS = [
    Predicate(
        name="at", signature={"?truck": OBJECT_TYPE, "?loc": LOCATION_TYPE}),
    Predicate(
        name="at", signature={"?obj": OBJECT_TYPE, "?loc": LOCATION_TYPE}),
]

TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS = [
    Predicate(
        name="at", signature={"?truck": OBJECT_TYPE, "?loc-from": LOCATION_TYPE}),
    Predicate(
        name="in-city", signature={"?truck": OBJECT_TYPE, "?loc-to": LOCATION_TYPE})
]

TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS = [
    Predicate(
        name="at", signature={"?truck": OBJECT_TYPE, "?loc-from": LOCATION_TYPE}),
    Predicate(
        name="in-city", signature={"?truck": OBJECT_TYPE, "?loc-to": LOCATION_TYPE}),
    Predicate(
        name="in-city", signature={"?truck": OBJECT_TYPE, "?loc-from": LOCATION_TYPE})
]

TEST_ACTION_NAME = "drive"


@fixture()
def expected_domain() -> Domain:
    domain_parser = DomainParser(NUMERIC_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def precision_recall_calculator() -> PrecisionRecallCalculator:
    return PrecisionRecallCalculator()


def test_calculate_true_positives_with_symmetric_predicates_return_correct_value():
    learned_preconditions = {p.untyped_representation for p in TEST_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_SYMMETRIC_EXPECTED_PRECONDITIONS}
    assert calculate_number_true_positives(learned_preconditions, expected_preconditions) == 2


def test_calculate_true_positives_with_non_symmetric_predicates_return_correct_value():
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    assert calculate_number_true_positives(learned_preconditions, expected_preconditions) == 2


def test_calculate_false_negatives_with_non_symmetric_predicates_return_correct_value():
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    assert calculate_number_false_negatives(learned_preconditions, expected_preconditions) == 1


def test_calculate_false_positive_with_non_symmetric_predicates_return_correct_value():
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    assert calculate_number_false_positives(learned_preconditions, expected_preconditions) == 1


def test_calculate_precision_with_less_learned_precondition_results_in_lower_precision():
    expected_precision = 2 / 3
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    assert calculate_precision(learned_preconditions, expected_preconditions) == expected_precision


def test_calculate_precision_with_extra_precondition_results_in_not_perfect_precision_value():
    expected_precision = 5 / 6
    learned_preconditions = {'(on ?y ?z)', '(available ?x)', '(at ?x ?p)', '(at ?y ?p)', '(clear ?y)', '(at ?z ?p)'}
    expected_preconditions = {'(on ?y ?z)', '(available ?x)', '(at ?x ?p)', '(at ?y ?p)', '(clear ?y)'}
    assert calculate_precision(learned_preconditions, expected_preconditions) == expected_precision


def test_calculate_recall_with_extra_learned_precondition_results_in_lower_recall():
    expected_recall = 2 / 3
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    assert calculate_recall(learned_preconditions, expected_preconditions) == expected_recall


def test_calculate_precision_does_not_divide_by_zero():
    try:
        calculate_precision(set([]), {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS})

    except ZeroDivisionError:
        pytest.fail()


def test_calculate_recall_does_not_divide_by_zero():
    try:
        calculate_recall(set([]), {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS})

    except ZeroDivisionError:
        pytest.fail()


def test_precision_recall_not_having_the_same_values():
    learned_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in TEST_NOT_SYMMETRIC_EXPECTED_PRECONDITIONS}
    assert calculate_recall(
        learned_preconditions, expected_preconditions) != calculate_precision(learned_preconditions,
                                                                              expected_preconditions)


def test_add_action_data_with_learned_action_stores_correct_values_in_object(
        expected_domain: Domain, precision_recall_calculator: PrecisionRecallCalculator):
    expected_action = expected_domain.actions[TEST_ACTION_NAME]
    learned_action = LearnerAction(name=TEST_ACTION_NAME, signature=expected_action.signature)
    learned_action.positive_preconditions = set(TEST_SYMMETRIC_LEARNED_PRECONDITIONS)
    precision_recall_calculator.add_action_data(learned_action=learned_action,
                                                model_action=expected_action)

    learned_preconditions = {p.untyped_representation for p in TEST_SYMMETRIC_LEARNED_PRECONDITIONS}
    expected_preconditions = {p.untyped_representation for p in expected_action.positive_preconditions}
    assert precision_recall_calculator.preconditions[TEST_ACTION_NAME] == learned_preconditions
    assert precision_recall_calculator.ground_truth_preconditions[TEST_ACTION_NAME] == expected_preconditions
    assert len(precision_recall_calculator.add_effects[TEST_ACTION_NAME]) == 0
    assert len(precision_recall_calculator.delete_effects[TEST_ACTION_NAME]) == 0


def test_calculate_action_precision_when_action_has_no_preconditions_in_model_domain_but_learned_action_contains_preconditions_returns_zero(
        expected_domain: Domain, precision_recall_calculator: PrecisionRecallCalculator):
    expected_action = expected_domain.actions[TEST_ACTION_NAME]
    expected_action.positive_preconditions = set()
    expected_action.add_effects = set()
    expected_action.delete_effects = set()

    learned_action = LearnerAction(name=TEST_ACTION_NAME, signature=expected_action.signature)
    learned_action.positive_preconditions = set(TEST_SYMMETRIC_LEARNED_PRECONDITIONS)
    learned_action.add_effects = set()
    learned_action.delete_effects = set()

    precision_recall_calculator.add_action_data(learned_action=learned_action,
                                                model_action=expected_action)
    precision = precision_recall_calculator.calculate_action_precision(TEST_ACTION_NAME)
    assert precision == 0


def test_calculate_action_recall_when_action_has_no_preconditions_in_model_domain_but_learned_action_contains_preconditions_returns_one(
        expected_domain: Domain, precision_recall_calculator: PrecisionRecallCalculator):
    expected_action = expected_domain.actions[TEST_ACTION_NAME]
    expected_action.positive_preconditions = set()
    expected_action.add_effects = set()
    expected_action.delete_effects = set()

    learned_action = LearnerAction(name=TEST_ACTION_NAME, signature=expected_action.signature)
    learned_action.positive_preconditions = set(TEST_SYMMETRIC_LEARNED_PRECONDITIONS)
    learned_action.add_effects = set()
    learned_action.delete_effects = set()

    precision_recall_calculator.add_action_data(learned_action=learned_action,
                                                model_action=expected_action)
    recall = precision_recall_calculator.calculate_action_recall(TEST_ACTION_NAME)
    assert recall == 1


def test_export_action_statistics_calculates_statistics_correctly(
        expected_domain: Domain, precision_recall_calculator: PrecisionRecallCalculator):
    expected_action = expected_domain.actions[TEST_ACTION_NAME]
    learned_action = LearnerAction(name=TEST_ACTION_NAME, signature=expected_action.signature)
    learned_action.positive_preconditions = {
        Predicate(name="at", signature={"?x": TRUCK_TYPE, "?y": LOCATION_TYPE}),
        Predicate(name="at", signature={"?x": TRUCK_TYPE, "?z": LOCATION_TYPE}),
    }
    learned_action.add_effects = expected_action.add_effects
    learned_action.delete_effects = expected_action.delete_effects

    precision_recall_calculator.add_action_data(learned_action=learned_action,
                                                model_action=expected_action)
    statistics = precision_recall_calculator.export_action_statistics(TEST_ACTION_NAME)
    expected_f1_score = 2 * (0.75 / 1.75)
    assert statistics == {
        "preconditions_precision": 0.5,
        "add_effects_precision": 1,
        "delete_effects_precision": 1,
        "preconditions_recall": 1,
        "add_effects_recall": 1,
        "delete_effects_recall": 1,
        "action_precision": 0.75,
        "action_recall": 1,
        "f1_score": expected_f1_score
    }

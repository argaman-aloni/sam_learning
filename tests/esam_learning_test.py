"""module tests for the ESAM learning algorithm"""
import logging
from typing import Hashable, Set, Dict

from nnf import Or, Var
from pddl_plus_parser.lisp_parsers import TrajectoryParser, ProblemParser, DomainParser
from pddl_plus_parser.models import Domain, Problem, GroundedPredicate, SignatureType
from pddl_plus_parser.models.observation import Observation
import pytest

from sam_learning.core import extract_effects
from sam_learning.learners.esam import ExtendedSamLearner, minimize_parameters_equality_dict
from tests.consts import (
    ROVERS_COMBINED_ESAM_PROBLEM_PATH,
    ROVERS_COMBINED_ESAM_TRAJECTORY_PATH,
    ROVERS_ESAM_DOMAIN_PATH,
)


@pytest.fixture()
def rovers_esam_domain() -> Domain:
    domain_parser = DomainParser(ROVERS_ESAM_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@pytest.fixture()
def rovers_esam_learner(rovers_esam_domain: Domain) -> ExtendedSamLearner:
    return ExtendedSamLearner(rovers_esam_domain)


@pytest.fixture()
def rovers_esam_problem(rovers_esam_domain: Domain) -> Problem:
    problem_parser = ProblemParser(ROVERS_COMBINED_ESAM_PROBLEM_PATH, rovers_esam_domain)
    return problem_parser.parse_problem()


@pytest.fixture()
def rovers_esam_observation(rovers_esam_domain: Domain, rovers_esam_problem: Problem) -> Observation:
    return TrajectoryParser(rovers_esam_domain, rovers_esam_problem).parse_trajectory(ROVERS_COMBINED_ESAM_TRAJECTORY_PATH)


# Helper for testing minimize parameters equality dict
def assert_get_minimize_parameters_equality_dict(
    rovers_esam_learner: ExtendedSamLearner, model_dict: Dict[Hashable, bool], expected_output: Dict[str, str]
):
    types = rovers_esam_learner.partial_domain.types
    action_signature: SignatureType = {
        "?r": types["rover"],
        "?l": types["lander"],
        "?p": types["waypoint"],
        "?x": types["waypoint"],
        "?y": types["waypoint"],
    }
    output = minimize_parameters_equality_dict(model_dict=model_dict, act_signature=action_signature, domain_types=types)
    assert output == expected_output


# ---------------------------
# Tests for _get_is_eff_clause_for_predicate
# ---------------------------
def test_is_eff_clause_multiple_binding(rovers_esam_learner: ExtendedSamLearner, rovers_esam_observation: Observation):
    logging.getLogger().setLevel(logging.INFO)
    rovers_esam_learner.logger.setLevel(logging.WARNING)

    comp = rovers_esam_observation.components[-1]
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, _ = extract_effects(prev_state, next_state)

    # Initialize with a default value then update when found
    grounded_predicate: GroundedPredicate = GroundedPredicate("communicated_soil_data", {}, {}, True)
    for effect in add_grounded_effects:
        if effect.name == "communicated_soil_data":
            grounded_predicate = effect

    expected_strs: Set[str] = {"(communicated_soil_data ?p)", "(communicated_soil_data ?x)"}
    or_clause: Or[Var] = rovers_esam_learner._get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals = or_clause.vars()
    predicates: Set[str] = {str(var) for var in literals}

    print("Multiple binding predicates:", predicates)
    assert len(predicates) == 2
    assert expected_strs.issubset(predicates)


def test_is_eff_clause_injective_binding(rovers_esam_learner: ExtendedSamLearner, rovers_esam_observation: Observation):
    comp = rovers_esam_observation.components[0]
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, _ = extract_effects(prev_state, next_state)

    grounded_predicate = GroundedPredicate("", {}, {}, True)
    for effect in add_grounded_effects:
        if effect.name == "calibrated":
            grounded_predicate = effect

    expected_strs = {"(calibrated ?i ?r)"}
    or_clause = rovers_esam_learner._get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals = or_clause.vars()
    predicates = {str(var) for var in literals}

    print("Injective binding predicates:", predicates)
    assert predicates == expected_strs


# ---------------------------
# Tests for get_minimize_parameters_equality_dict (basic cases)
# ---------------------------
@pytest.mark.parametrize(
    "model_dict,expected",
    [
        (
            {"(communicated_soil_data ?p - waypoint)": True, "(communicated_soil_data ?x - waypoint)": True},
            {"?x": "?p", "?y": "?y", "?p": "?p", "?r": "?r", "?l": "?l"},
        ),
        (
            {"(communicated_soil_data ?p - waypoint)": False, "(communicated_soil_data ?x - waypoint)": True},
            {"?x": "?x", "?y": "?y", "?p": "?p", "?r": "?r", "?l": "?l"},
        ),
        (
            {"(communicated_soil_data ?p - waypoint)": True, "(communicated_soil_data ?x - waypoint)": False},
            {"?x": "?x", "?y": "?y", "?p": "?p", "?r": "?r", "?l": "?l"},
        ),
    ],
)
def test_minimize_parameters_equality_dict_basic(rovers_esam_learner: ExtendedSamLearner, model_dict: Dict[Hashable, bool], expected: Dict[str, str]):
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner, model_dict, expected)


# ---------------------------
# Tests for get_minimize_parameters_equality_dict (complex cases)
# ---------------------------
@pytest.mark.parametrize(
    "model_dict,expected",
    [
        (
            {
                "(communicated_soil_data ?p - waypoint)": True,
                "(communicated_soil_data ?x - waypoint)": True,
                "(communicated_soil_data ?y - waypoint)": True,
            },
            {"?p": "?p", "?x": "?p", "?y": "?p", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": True,
                "(communicated_soil_data ?x - waypoint)": True,
                "(communicated_soil_data ?y - waypoint)": False,
            },
            {"?p": "?p", "?x": "?p", "?y": "?y", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": True,
                "(communicated_soil_data ?x - waypoint)": False,
                "(communicated_soil_data ?y - waypoint)": True,
            },
            {"?p": "?p", "?x": "?x", "?y": "?p", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": False,
                "(communicated_soil_data ?x - waypoint)": True,
                "(communicated_soil_data ?y - waypoint)": True,
            },
            {"?p": "?p", "?x": "?x", "?y": "?x", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": True,
                "(communicated_soil_data ?x - waypoint)": False,
                "(communicated_soil_data ?y - waypoint)": False,
            },
            {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": False,
                "(communicated_soil_data ?x - waypoint)": True,
                "(communicated_soil_data ?y - waypoint)": False,
            },
            {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"},
        ),
        (
            {
                "(communicated_soil_data ?p - waypoint)": False,
                "(communicated_soil_data ?x - waypoint)": False,
                "(communicated_soil_data ?y - waypoint)": True,
            },
            {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"},
        ),
    ],
)
def test_minimize_parameters_equality_dict_complex(
    rovers_esam_learner: ExtendedSamLearner, model_dict: Dict[Hashable, bool], expected: Dict[str, str]
):
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner, model_dict, expected)

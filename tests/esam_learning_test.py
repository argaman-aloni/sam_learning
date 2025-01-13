"""module tests for the ESAM learning algorithm"""
import logging
from nnf import Or, Var
from pddl_plus_parser.lisp_parsers.parsing_utils import parse_predicate_from_string
from pytest import fixture
from pddl_plus_parser.lisp_parsers import TrajectoryParser, ProblemParser, DomainParser
from pddl_plus_parser.models import Domain, Problem, GroundedPredicate
from pddl_plus_parser.models.observation import Observation

from sam_learning.core import extract_effects
from sam_learning.learners.esam import ExtendedSamLearner
from tests.consts import ROVERS_COMBINED_ESAM_PROBLEM_PATH, ROVERS_COMBINED_ESAM_TRAJECTORY_PATH, \
    ROVERS_ESAM_DOMAIN_PATH

@fixture()
def rovers_esam_domain() -> Domain:
    domain_parser = DomainParser(ROVERS_ESAM_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()

@fixture()
def rovers_esam_learner(rovers_esam_domain: Domain) -> ExtendedSamLearner:
    return ExtendedSamLearner(rovers_esam_domain)

@fixture()
def rovers_esam_problem(rovers_esam_domain: Domain) -> Problem:
    problem_parser = ProblemParser(ROVERS_COMBINED_ESAM_PROBLEM_PATH, rovers_esam_domain)
    return problem_parser.parse_problem()
@fixture()
def rovers_esam_observation(rovers_domain: Domain, rovers_esam_problem: Problem, rovers_esam_domain) -> Observation:
    return TrajectoryParser(rovers_esam_domain,
                            rovers_esam_problem).parse_trajectory(ROVERS_COMBINED_ESAM_TRAJECTORY_PATH)

def test_get_is_eff_clause_for_predicate(rovers_esam_learner: ExtendedSamLearner, rovers_esam_observation: Observation):
    logging.getLogger().setLevel(level=logging.INFO)
    comp = rovers_esam_observation.components[-1]
# ======= first test, multiple binding=======
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, del_grounded_effects = extract_effects(prev_state, next_state)
    grounded_predicate: GroundedPredicate = GroundedPredicate(
        "communicated_soil_data", dict(),dict(), True )

    for add_grounded_effect in add_grounded_effects:
        if add_grounded_effect.name == "communicated_soil_data":
            grounded_predicate = add_grounded_effect

    expected_strs: set[str] = {"(communicated_soil_data ?p - waypoint)", "(communicated_soil_data ?x - waypoint)"}
    or_clause: Or[Var] = rovers_esam_learner.get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals = or_clause.vars()
    predicates: set[str] = {var.__str__() for var in literals}
    assert (len(list(predicates)) == 2)
    assert expected_strs.issubset(predicates)

# ======= second test, injective binding=======

    comp = rovers_esam_observation.components[0]
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, del_grounded_effects = extract_effects(prev_state, next_state)
    for add_grounded_effect in add_grounded_effects:
        if add_grounded_effect.name == "calibrated":
            grounded_predicate = add_grounded_effect
    expected_strs = {"(calibrated ?i - camera ?r - rover)"}
    or_clause = rovers_esam_learner.get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals = or_clause.vars()
    predicates = {var.__str__() for var in literals}
    assert (len(list(predicates)) == 1, f"lens of lists does not match")
    assert expected_strs.__eq__(predicates)








# def test_get_surely_not_eff(rovers_esam_learner: ExtendedSamLearner):
#     pass
#
# def test_build_cnf_formulas(rovers_esam_learner: ExtendedSamLearner):
#     pass
#
# def test_get_minimize_parameters_equality_dict(rovers_esam_learner: ExtendedSamLearner):
#     pass
#
# def test_modify_predicate_signature(rovers_esam_learner: ExtendedSamLearner):
#     pass




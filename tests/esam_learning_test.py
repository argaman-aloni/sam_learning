"""module tests for the ESAM learning algorithm"""
import logging
from typing import Hashable, Set, Dict

from nnf import Or, Var
from pddl_plus_parser.lisp_parsers.parsing_utils import parse_predicate_from_string
from pytest import fixture
from pddl_plus_parser.lisp_parsers import TrajectoryParser, ProblemParser, DomainParser
from pddl_plus_parser.models import Domain, Problem, GroundedPredicate, Predicate, SignatureType, PDDLType
from pddl_plus_parser.models.observation import Observation, ObservedComponent

from sam_learning.core import extract_effects
from sam_learning.learners.esam import (ExtendedSamLearner, modify_predicate_signature,
                                        get_minimize_parameters_equality_dict)
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
    rovers_esam_learner.logger.setLevel(level=logging.WARNING)
    comp = rovers_esam_observation.components[-1]
# ======= first test, multiple binding=======
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, _ = extract_effects(prev_state, next_state)
    grounded_predicate: GroundedPredicate = GroundedPredicate(
        "communicated_soil_data", {}, {}, True )

    for add_grounded_effect in add_grounded_effects:
        if add_grounded_effect.name == "communicated_soil_data":
            grounded_predicate = add_grounded_effect

    expected_strs: Set[str] = {"(communicated_soil_data ?p)", "(communicated_soil_data ?x)"}
    or_clause: Or[Var] = rovers_esam_learner.get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals = or_clause.vars()
    predicates: Set[str] = {str(var) for var in literals}
    print("first test predicates are")
    print(predicates)
    assert len(predicates) == 2
    assert expected_strs.issubset(predicates)

# ======= second test, injective binding=======

    comp = rovers_esam_observation.components[0]
    prev_state = comp.previous_state
    next_state = comp.next_state
    grounded_action = comp.grounded_action_call
    add_grounded_effects, _ = extract_effects(prev_state, next_state)
    for add_grounded_effect in add_grounded_effects:
        if add_grounded_effect.name == "calibrated":
            grounded_predicate = add_grounded_effect
    expected_strs = {"(calibrated ?i ?r)"}
    or_clause = rovers_esam_learner.get_is_eff_clause_for_predicate(grounded_action, grounded_predicate)
    literals: list = or_clause.vars()
    predicates = {var.__str__() for var in literals}
    print("second test predicates are")
    print(predicates)
    assert expected_strs.__eq__(predicates)

def assert_get_minimize_parameters_equality_dict(rovers_esam_learner: ExtendedSamLearner,
                                               model_dict: Dict[Hashable, bool],
                                               expected_output: Dict[str,str]):
    types = rovers_esam_learner.partial_domain.types
    action_signature: SignatureType = {
        '?r': types["rover"],
        '?l': types["lander"],
        '?p': types["waypoint"],
        '?x': types["waypoint"],
        '?y': types["waypoint"]}

    output = get_minimize_parameters_equality_dict(model_dict=model_dict,
                                                   act_signature=action_signature,
                                                   domain_types=types)
    assert output == expected_output



def test_get_minimize_parameters_equality_dict1(rovers_esam_learner: ExtendedSamLearner,
                                               rovers_esam_observation: Observation):
    # learning from action indexed -1 (communicate_soil_data 'rover0' 'general' 'waypoint2' 'waypoint2' 'waypoint0')
    # multiple binding

    pred1: str = "(communicated_soil_data ?p - waypoint)"
    pred2: str = "(communicated_soil_data ?x - waypoint)"

# ========================================================================
# ========================test set 1======================================
# ========================================================================
    logging.getLogger().log(logging.INFO, "test1 in test_get_minimize_parameters_equality_dict1")
    communicate_soil_dict1: Dict[Hashable, bool] = {pred1: True, pred2: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict1,
                                               {"?x": "?p",
                                                "?y": "?x",
                                                "?p": "?p",
                                                "?r": "?r",
                                                "?l": "?l"})

#===================================================================================
    logging.getLogger().log(logging.INFO, "test2 in test_get_minimize_parameters_equality_dict1")
    communicate_soil_dict2: Dict[Hashable, bool] = {pred1: False, pred2: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict2,
                                               {"?x": "?x",
                                                "?y": "?y",
                                                "?p": "?p",
                                                "?r": "?r",
                                                "?l": "?l"})
# ===================================================================================
    logging.getLogger().log(logging.INFO, "test3 in test_get_minimize_parameters_equality_dict1")
    communicate_soil_dict3: Dict[Hashable, bool] = {pred1: True, pred2: False}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict3,
                                               {"?x": "?x",
                                                "?y": "?y",
                                                "?p": "?p",
                                                "?r": "?r",
                                                "?l": "?l"})

# more complex formulas
def test_get_minimize_parameters_equality_dict2(rovers_esam_learner: ExtendedSamLearner,
                                                rovers_esam_observation: Observation):
#========================================================================
#========================test set 2======================================
#========================================================================
    pred1: str = "(communicated_soil_data ?p - waypoint)"
    pred2: str = "(communicated_soil_data ?x - waypoint)"
    pred3: str = "(communicated_soil_data ?y - waypoint)"
    # ============================================1


    logging.getLogger().log(logging.INFO, "test1 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict1: Dict[Hashable, bool] = {pred1: True, pred2: True, pred3: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict1,
                                               {"?p": "?p",
                                                "?x": "?p",
                                                "?y": "?p",
                                                "?r": "?r",
                                                "?l": "?l"})

    # ============================================2


    logging.getLogger().log(logging.INFO, "test2 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict2: Dict[Hashable, bool] = {pred1: True, pred2: True, pred3: False}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict2,
                                               {"?p": "?p",
                                                "?x": "?p",
                                                "?y": "?x",
                                                "?r": "?r",
                                                "?l": "?l"})

    #3============================================3

    logging.getLogger().log(logging.INFO, "test3 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict3: Dict[Hashable, bool] = {pred1: True, pred2: False, pred3: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict3,
                                               {"?p": "?p",
                                                "?x": "?x",
                                                "?y": "?p",
                                                "?r": "?r",
                                                "?l": "?l"})

    #4============================================4
    logging.getLogger().log(logging.INFO, "test4 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict4: Dict[Hashable, bool] = {pred1: False, pred2: True, pred3: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict4,
                                               {"?p": "?p",
                                                "?x": "?x",
                                                "?y": "?x",
                                                "?r": "?r",
                                                "?l": "?l"})
    #5============================================5


    logging.getLogger().log(logging.INFO, "test5 in test_get_minimize_parameters_equality_dict2")

    communicate_soil_dict5: Dict[Hashable, bool] = {pred1: True, pred2: False, pred3: False}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict5,
                                               {"?p": "?p",
                                                "?x": "?x",
                                                "?y": "?y",
                                                "?r": "?r",
                                                "?l": "?l"})

    #6============================================6
    logging.getLogger().log(logging.INFO, "test6 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict6: Dict[Hashable, bool] = {pred1: False, pred2: True, pred3: False}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict6,
                                               {"?p": "?p",
                                                "?x": "?x",
                                                "?y": "?y",
                                                "?r": "?r",
                                                "?l": "?l"})
    #7============================================7


    logging.getLogger().log(logging.INFO, "test7 in test_get_minimize_parameters_equality_dict2")
    communicate_soil_dict7: Dict[Hashable, bool] = {pred1: False, pred2: False, pred3: True}
    assert_get_minimize_parameters_equality_dict(rovers_esam_learner,
                                               communicate_soil_dict7,
                                               {"?p": "?p",
                                                "?x": "?x",
                                                "?y": "?y",
                                                "?r": "?r",
                                                "?l": "?l"})




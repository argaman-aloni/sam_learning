"""module tests for the ESAM learning algorithm"""
import logging
from typing import Hashable, Set, Dict, List

from nnf import Or, Var
from pddl_plus_parser.lisp_parsers import TrajectoryParser, ProblemParser, DomainParser
from pddl_plus_parser.models import Domain, Problem, GroundedPredicate, SignatureType, ActionCall
from pddl_plus_parser.models.observation import Observation
import pytest

from sam_learning.core import extract_effects, LearnerAction, NotSafeActionError
from sam_learning.learners.esam import ExtendedSamLearner, minimize_parameters_equality_dict, ProxyActionData
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

@pytest.fixture(scope="module")
def rover_types() -> Dict:
    domain_parser = DomainParser(ROVERS_ESAM_DOMAIN_PATH, partial_parsing=True)
    domain = domain_parser.parse_domain()
    return ExtendedSamLearner(domain).partial_domain.types


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


def make_learner_action(name, signature_params, types_dict):
    """Create a LearnerAction with the correct types from the types_dict"""
    signature = {}
    for param, type_name in signature_params.items():
        signature[param] = types_dict[type_name]
    return LearnerAction(name=name, signature=signature)

@pytest.mark.parametrize(
    "proxy_action_call,signature_params,proxy_data,action_name,expected",
    [
        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint"
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint0", "waypoint0"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint0", "waypoint1"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint0"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?x", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint1"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"])
        ),
    ],
)
def test_decoder_complex(rovers_esam_learner: ExtendedSamLearner,
                         proxy_action_call: ActionCall,
                         signature_params: Dict[str, str],
                         proxy_data: ProxyActionData,
                         action_name: str,
                         expected: ActionCall,
                         rover_types: Dict):  # Get rover_types from fixture


    # Create the LearnerAction with the types from the fixture
    new_proxy = make_learner_action(
        proxy_action_call.name,
        signature_params,
        rover_types
    )

    action_call: ActionCall = rovers_esam_learner.decoder_method(
        proxy_action_call,
        new_proxy,
        proxy_data,
        action_name
    )
    assert action_call.parameters == expected.parameters




@pytest.mark.parametrize(
    "original_action_call,signature_params,proxy_data,action_name,expected",
    [
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint0", "waypoint0"]),
            #create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint"
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0","waypoint1"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint1", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint1", "waypoint0"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"])
        ),
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint0", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint0", "waypoint0"])
        ),
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint0", "waypoint1", "waypoint0"])
        ),
    ]
)
def test_encoder_good_complex_examples(rovers_esam_learner: ExtendedSamLearner, original_action_call: ActionCall,
                                       signature_params: Dict[str, str], proxy_data: ProxyActionData, action_name: str,
                                       expected: ActionCall, rover_types: Dict):

    # Create the LearnerAction with the types from the fixture
    new_proxy = make_learner_action(
        expected.name,
        signature_params,
        rover_types
    )

    res: ActionCall = rovers_esam_learner.encoder_method(original_action_call, proxy_data, new_proxy, action_name)
    assert str(res)== str(expected)



@pytest.mark.parametrize(
    "original_action_call,signature_params,proxy_data,action_name",
    [
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            # test fails because in proxy action y=x=p, but waypoint1!=waypoint0 (while x=p) that results in a conflict
        ),

        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint1", "waypoint0", "waypoint0"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?p", "?y": "?y", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            #test fails because in proxy action x=p, but waypoint1!=waypoint0 that results in a conflict
        ),

        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint0", "waypoint1", "waypoint2"]),
            # create a factory function that takes types as param
            {
                "?r": "rover",
                "?l": "lander",
                "?p": "waypoint",
                "?x": "waypoint",
                "?y": "waypoint",
            },
            ProxyActionData(set(), set(), {"?p": "?p", "?x": "?x", "?y": "?p", "?r": "?r", "?l": "?l"}),
            "communicate_soil_data",
            #test fails because in proxy action y=p, but waypoint2!=waypoint0 that results in a conflict
        ),
    ]
)
def test_encoder_bad_complex_examples(rovers_esam_learner: ExtendedSamLearner, original_action_call: ActionCall,
                                       signature_params: Dict[str, str], proxy_data: ProxyActionData, action_name: str,
                                       rover_types: Dict):
    """
    call to encoder should result in a 'NotSafeActionError' since it's not a safe action if encoded
    """

    # Create the LearnerAction with the types from the fixture
    new_proxy = make_learner_action(
        "communicate_soil_data_1",
        signature_params,
        rover_types
    )
    try:
        rovers_esam_learner.encoder_method(original_action_call, proxy_data, new_proxy, action_name)
        assert False
    except NotSafeActionError:
        assert True


@pytest.mark.parametrize(
    "original_action_call,expected",
    [
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint1", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param

            [["rover0", "general", "waypoint1", "waypoint1", "waypoint0"],
             ["rover0", "general", "waypoint1", "waypoint1", "waypoint0"],
             ["rover0", "general", "waypoint1", "waypoint0"]]
        ),
        pytest.param(
            ActionCall("communicate_soil_data", ["rover0", "general", "waypoint1", "waypoint0", "waypoint1"]),
            # create a factory function that takes types as param
            [["rover0", "general", "waypoint1", "waypoint0", "waypoint1"]],
        ),
    ]
)
def test_encoder_full_example(rovers_esam_learner, rovers_esam_observation, original_action_call: ActionCall,
                            expected: List[List[str]], rover_types: Dict):
    """
    call to encoder should result in a 'NotSafeActionError' since it's not a safe action if encoded
    """
    rovers_esam_learner.learn_action_model([rovers_esam_observation])
    res = rovers_esam_learner.encode(action_call=original_action_call)
    for a in res:
        print (str(a))
    actual = [a.parameters for a in res]
    for a in expected:
        assert a in actual
    for a in actual:
        assert a in expected
@pytest.mark.parametrize(
    "proxy_action_call,expected",
    [
        pytest.param(
            ActionCall("communicate_soil_data_1", ["rover0", "general", "waypoint1", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            ActionCall("communicate_soil_data",["rover0", "general", "waypoint1", "waypoint1", "waypoint0"])
        ),

        pytest.param(
            ActionCall("communicate_soil_data_2", ["rover0", "general", "waypoint1", "waypoint0"]),
            # create a factory function that takes types as param
            ActionCall("communicate_soil_data",["rover0", "general", "waypoint1", "waypoint1", "waypoint0"]),
        ),
    ]
)
def test_decoder_full_example(rovers_esam_learner, rovers_esam_observation, proxy_action_call: ActionCall,
                            expected: ActionCall, rover_types: Dict):
    """
    call to encoder should result in a 'NotSafeActionError' since it's not a safe action if encoded
    """
    rovers_esam_learner.learn_action_model([rovers_esam_observation])
    output = rovers_esam_learner.decode(action_call=proxy_action_call)
    assert output.name == expected.name and output.name == expected.name


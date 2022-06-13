"""Module test for the fault repair functionality."""
from pddl_plus_parser.exporters import ENHSPParser
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, State, Operator
from pytest import fixture

from fault_detection import FaultRepair, FaultGenerator, DefectType
from tests.consts import EXAMPLES_DIR_PATH, NUMERIC_DOMAIN_PATH

FAULTY_DOMAIN_PATH = EXAMPLES_DIR_PATH / "faulty_domain.pddl"
DEPOT_FLUENTS_MAP_PATH = EXAMPLES_DIR_PATH / "depot_fluents_map.json"
DEPOT_REPAIR_TEST_PROBLEM_PATH = EXAMPLES_DIR_PATH / "depot_pfile1.pddl"
DEPOT_FAULTY_PLAN_PATH = EXAMPLES_DIR_PATH / "faulty_plan.solution"


@fixture()
def domain() -> Domain:
    domain_parser = DomainParser(NUMERIC_DOMAIN_PATH)
    return domain_parser.parse_domain()


@fixture()
def faulty_domain() -> Domain:
    domain_parser = DomainParser(FAULTY_DOMAIN_PATH)
    return domain_parser.parse_domain()


@fixture()
def problem(domain: Domain) -> Problem:
    return ProblemParser(DEPOT_REPAIR_TEST_PROBLEM_PATH, domain).parse_problem()


@fixture
def fault_generator() -> FaultGenerator:
    """Fixture for the fault generator."""
    return FaultGenerator(work_dir_path=EXAMPLES_DIR_PATH,
                          model_domain_file_name="depot_numeric.pddl")


@fixture()
def fault_repair() -> FaultRepair:
    return FaultRepair(working_directory_path=EXAMPLES_DIR_PATH,
                       model_domain_file_name="depot_numeric.pddl",
                       fluents_map_path=DEPOT_FLUENTS_MAP_PATH)


def test_validate_applied_action_returns_true_if_action_is_valid(
        fault_repair: FaultRepair, domain: Domain, problem: Problem):
    """Test that the validate_applied_action function returns true if the action is valid."""
    valid_previous_state = State(predicates=problem.initial_state_predicates,
                                 fluents=problem.initial_state_fluents, is_init=True)
    valid_operator = Operator(action=domain.actions["lift"], domain=domain,
                              grounded_action_call=["hoist3", "crate2", "crate0", "distributor1"])
    valid_next_state = valid_operator.apply(valid_previous_state)
    assert fault_repair._validate_applied_action(
        faulty_action_name="lift", valid_next_state=valid_next_state, faulty_next_state=valid_next_state) is True


def test_validate_applied_action_returns_false_if_the_faulty_action_differs_from_expected(
        fault_repair: FaultRepair, domain: Domain, problem: Problem, faulty_domain: Domain):
    """Test that validates that if the next state of the faulty action differs from the expected next state,
    the function returns false.
    """
    valid_previous_state = State(predicates=problem.initial_state_predicates,
                                 fluents=problem.initial_state_fluents, is_init=True)
    valid_operator = Operator(action=domain.actions["lift"], domain=domain,
                              grounded_action_call=["hoist3", "crate2", "crate0", "distributor1"])
    invalid_operator = Operator(action=faulty_domain.actions["lift"],
                                domain=faulty_domain,
                                grounded_action_call=["hoist3", "crate2", "crate0", "distributor1"])
    valid_next_state = valid_operator.apply(valid_previous_state)
    invalid_next_state = invalid_operator.apply(valid_previous_state)
    assert fault_repair._validate_applied_action(
        faulty_action_name="lift", valid_next_state=valid_next_state, faulty_next_state=invalid_next_state) is False


def test_observe_single_plan_on_a_faulty_plan_returns_lift_as_faulty_action(
        fault_repair: FaultRepair, problem: Problem, faulty_domain: Domain):
    """Test that the observe_single_plan function returns the faulty action name."""
    plan_sequence = ENHSPParser().parse_plan_content(DEPOT_FAULTY_PLAN_PATH)
    _, _, faulty_action_name = fault_repair._observe_single_plan(
        plan_sequence=plan_sequence, faulty_domain=faulty_domain, problem=problem)
    assert faulty_action_name == "lift"


def test_filter_redundant_observations_removes_states_of_actions_that_are_not_faulty(
        fault_repair: FaultRepair, problem: Problem, faulty_domain: Domain):
    """Test that the filter_redundant_observations function removes states of actions that are not faulty."""
    plan_sequence = ENHSPParser().parse_plan_content(DEPOT_FAULTY_PLAN_PATH)
    valid_observation, faulty_observation, faulty_action_name = fault_repair._observe_single_plan(
        plan_sequence=plan_sequence, faulty_domain=faulty_domain, problem=problem)
    valid_observations = [valid_observation]
    faulty_observations = [faulty_observation]
    fault_repair._filter_redundant_observations("lift", valid_observations, faulty_observations)
    assert all([comp.grounded_action_call.name == "lift" for comp in valid_observation.components])
    assert all([comp.grounded_action_call.name == "lift" for comp in faulty_observation.components])


def test_repair_model_fix_numeric_effect_when_given_valid_observation(
        fault_repair: FaultRepair, problem: Problem, faulty_domain: Domain, fault_generator: FaultGenerator):
    """Test that the repair_model fixes the numeric effect of the faulty action when given a valid observation."""
    plan_sequence = ENHSPParser().parse_plan_content(DEPOT_FAULTY_PLAN_PATH)
    valid_observation, faulty_observation, faulty_action_name = fault_repair._observe_single_plan(
        plan_sequence=plan_sequence, faulty_domain=faulty_domain, problem=problem)
    valid_observations = [valid_observation]
    faulty_observations = [faulty_observation]
    faulty_learner_domain = fault_generator.generate_faulty_domain(
        defect_type=DefectType.numeric_effect, action_to_alter="lift")

    fault_repair._filter_redundant_observations("lift", valid_observations, faulty_observations)
    repaired_model = fault_repair.repair_model(
        faulty_domain=faulty_learner_domain, valid_observations=valid_observations, faulty_action_name="lift")

    assert repaired_model.actions["lift"].numeric_effects[0] == "(increase (fuel-cost ) 1.0)"

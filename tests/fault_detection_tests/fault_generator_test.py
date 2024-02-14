"""Module test for the fault generation module."""
import unittest.mock as mock
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Action, Domain
from pytest import fixture

from fault_detection import FaultGenerator
from tests.consts import DEPOTS_NUMERIC_DOMAIN_PATH

WORKING_DIRECTORY_PATH = Path("tests/")


@fixture()
def domain() -> Domain:
    domain_parser = DomainParser(DEPOTS_NUMERIC_DOMAIN_PATH)
    return domain_parser.parse_domain()


@fixture
def drive_action(domain: Domain) -> Action:
    """Fixture that returns drive action from depot domain."""
    return domain.actions["drive"]


@fixture
def load_action(domain: Domain) -> Action:
    """Fixture that returns load action from depot domain."""
    return domain.actions["load"]


@fixture
def fault_generator() -> FaultGenerator:
    """Fixture for the fault generator."""
    return FaultGenerator(work_dir_path=WORKING_DIRECTORY_PATH,
                          model_domain_file_name="model_domain.pddl")


def test_alter_action_numeric_precondition_change_numeric_precondition_sign_to_geq(
        fault_generator: FaultGenerator, load_action: Action):
    """Test that the numeric precondition sign is changed to >=."""
    fault_generator.alter_action_numeric_precondition(faulty_action=load_action)
    assert load_action.numeric_preconditions.pop().root.value == ">="


def test_alter_action_numeric_effect_adds_an_additional_add_expression_to_the_expression_tree(
        fault_generator: FaultGenerator, load_action: Action):
    """Test that the numeric effect is increased by five."""
    mock_randint = lambda x, y: 5
    with mock.patch('random.randint', mock_randint):
        fault_generator.alter_action_numeric_effect(faulty_action=load_action)
        pddl_altered_effect = "(increase (current_load ?z) (+ (weight ?y) 5))"
        assert load_action.numeric_effects.pop().to_pddl() == pddl_altered_effect


def test_alter_action_numeric_effect_increases_numeric_expression_by_five(
        fault_generator: FaultGenerator, drive_action: Action):
    """Test that the numeric effect is increased by five."""
    mock_randint = lambda x, y: 5
    with mock.patch('random.randint', mock_randint):
        fault_generator.alter_action_numeric_effect(faulty_action=drive_action)
        pddl_altered_effect = "(increase (fuel-cost ) 15.0)"
        assert drive_action.numeric_effects.pop().to_pddl() == pddl_altered_effect


def test_alter_action_numeric_precondition_additional_add_expression_to_the_expression_tree(
        fault_generator: FaultGenerator, load_action: Action):
    """Test that the numeric effect is increased by five."""
    mock_randint = lambda x, y: 5
    with mock.patch('random.randint', mock_randint):
        fault_generator.alter_action_numeric_precondition_value(faulty_action=load_action)
        pddl_altered_preconditions = "(<= (+ (current_load ?z) (weight ?y)) (+ (load_limit ?z) 5))"
        assert load_action.numeric_preconditions.pop().to_pddl() == pddl_altered_preconditions


def test_remove_predicate_from_action_removes_predicate_from_action(
        fault_generator: FaultGenerator, load_action: Action):
    """Test that the predicate is removed from the action."""
    preconditions = list(load_action.positive_preconditions)
    precondition_to_remove = preconditions[0]
    mock_randchoice = lambda x: precondition_to_remove
    with mock.patch('random.choice', mock_randchoice):
        fault_generator.remove_predicate_from_action(faulty_action=load_action)
        assert precondition_to_remove not in list(load_action.positive_preconditions)


def test_select_action_to_alter_returns_correct_action(
        fault_generator: FaultGenerator, load_action: Action, domain: Domain):
    """Test that the correct action is returned."""
    mock_randchoice = lambda x: load_action
    with mock.patch('random.choice', mock_randchoice):
        selected_action = fault_generator._select_action_to_alter(altered_domain=domain)
        assert selected_action == load_action


def test_remove_numeric_precondition_from_action_removes_precondition_from_action(
        fault_generator: FaultGenerator, load_action: Action):
    """Test that the numeric expression is removed from the action."""
    preconditions = list(load_action.numeric_preconditions)
    precondition_to_remove = preconditions[0]
    mock_randchoice = lambda x: precondition_to_remove
    assert precondition_to_remove in list(load_action.numeric_preconditions)
    with mock.patch('random.choice', mock_randchoice):
        fault_generator.remove_numeric_precondition(faulty_action=load_action)
        assert precondition_to_remove not in list(load_action.numeric_preconditions)

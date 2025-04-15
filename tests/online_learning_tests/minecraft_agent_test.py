"""Module test for the IPC active learning agent."""
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Problem, State, ActionCall
from pytest import fixture

from sam_learning.core.online_learning_agents.minecraft_agent import MinecraftAgent
from tests.consts import MINECRAFT_LARGE_DOMAIN_PATH


@fixture()
def minecraft_large_domain() -> Domain:
    domain_parser = DomainParser(MINECRAFT_LARGE_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture
def minecraft_agent(minecraft_large_domain: Domain, minecraft_large_problem: Problem) -> MinecraftAgent:
    return MinecraftAgent(minecraft_large_domain, minecraft_large_problem)


def test_get_environment_actions_gets_only_legal_teleport_actions(minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_teleport_actions = 36
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len([action for action in total_grounded_actions if action.name == "tp_to"]) == num_expected_teleport_actions


def test_get_environment_actions_gets_only_one_craft_wooden_pogo_action(minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_pogo_actions = 1
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len([action for action in total_grounded_actions if action.name == "craft_wooden_pogo"]) == num_expected_pogo_actions


def test_get_environment_actions_gets_only_one_break_action(minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_pogo_actions = 1
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len([action for action in total_grounded_actions if action.name == "break"]) == num_expected_pogo_actions


def test_get_environment_actions_gets_only_one_creaft_tree_tap_action(minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_pogo_actions = 1
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len([action for action in total_grounded_actions if action.name == "craft_tree_tap"]) == num_expected_pogo_actions


def test_get_environment_actions_gets_the_correct_number_of_possible_actions_according_to_the_minecraft_setting(
    minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem
):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_actions = 42
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len(total_grounded_actions) == num_expected_actions


def test_observe_when_trying_to_teleport_to_the_same_position_does_not_remove_the_position_of_the_agent_from_the_predicates(
    minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem
):
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    # agent position (position cell15)
    tp_to_self = ActionCall(name="tp_to", grounded_parameters=["cell15", "cell15"])
    new_state, reward = minecraft_agent.observe(initial_state, tp_to_self)
    assert "(position cell15)" in new_state.serialize()


def test_observe_when_trying_to_apply_craft_plank_when_craft_plank_is_applicable_returns_new_state_with_less_logs_and_more_planks(
    minecraft_agent: MinecraftAgent, minecraft_large_problem: Problem
):
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    assert state_fluents["(count_log_in_inventory )"].value == 4
    assert state_fluents["(count_planks_in_inventory )"].value == 7

    craft_plank_action = ActionCall(name="craft_plank", grounded_parameters=[])
    new_state, reward = minecraft_agent.observe(initial_state, craft_plank_action)
    assert reward == 1
    assert new_state.state_fluents["(count_log_in_inventory )"].value == 3
    assert new_state.state_fluents["(count_planks_in_inventory )"].value == 11

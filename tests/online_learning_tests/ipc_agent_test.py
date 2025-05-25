"""Module test for the IPC active learning agent."""

from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, State, ActionCall
from pytest import fixture

from sam_learning.core.online_learning_agents import IPCAgent
from tests.consts import (
    DEPOTS_NUMERIC_DOMAIN_PATH,
    MINECRAFT_LARGE_DOMAIN_PATH,
    DEPOT_ONLINE_LEARNING_PROBLEM,
    DEPOT_ONLINE_LEARNING_PLAN,
    create_plan_actions,
    DEPOT_ONLINE_LEARNING_PROBLEM_WITH_NUMERIC_GOAL,
)


@fixture()
def depot_numeric_domain() -> Domain:
    domain_parser = DomainParser(DEPOTS_NUMERIC_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def minecraft_large_domain() -> Domain:
    domain_parser = DomainParser(MINECRAFT_LARGE_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture
def minecraft_agent(minecraft_large_domain: Domain, minecraft_large_problem: Problem) -> IPCAgent:
    agent = IPCAgent(minecraft_large_domain)
    agent.initialize_problem(minecraft_large_problem)
    return agent


@fixture
def depot_discrete_agent(depot_discrete_domain: Domain, depot_discrete_problem: Problem) -> IPCAgent:
    agent = IPCAgent(depot_discrete_domain)
    agent.initialize_problem(depot_discrete_problem)
    return agent


@fixture
def depot_numeric_agent(depot_numeric_domain: Domain, depot_problem: Problem) -> IPCAgent:
    agent = IPCAgent(depot_numeric_domain)
    agent.initialize_problem(depot_problem)
    return agent


def test_observe_on_state_with_applicable_action_returns_correct_next_state_and_that_the_action_was_applied_successfully(
    depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent
):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)
    expected_predicate_in_state = "(at truck1 distributor1)"
    assert expected_predicate_in_state not in initial_state.serialize()
    action = ActionCall(name="drive", grounded_parameters=["truck1", "depot0", "distributor1"])
    expected_predicate_not_in_state = "(at truck1 depot0)"

    # Act
    next_state, is_applicable, reward = depot_discrete_agent.observe(initial_state, action)

    # Assert
    assert expected_predicate_in_state in next_state.serialize()
    assert expected_predicate_not_in_state not in next_state.serialize()
    assert is_applicable


def test_observe_on_state_with_inapplicable_action_returns_the_same_state_as_before_and_states_that_the_action_was_inapplicable(
    depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent
):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck1", "distributor1", "depot0"])

    # Act
    next_state, is_applicable, reward = depot_discrete_agent.observe(initial_state, action)

    # Assert
    assert next_state == initial_state
    assert not is_applicable


def test_observe_on_state_with_applicable_action_returns_correct_next_state_when_the_domain_numeric(
    depot_problem: Problem, depot_numeric_agent: IPCAgent
):
    # Arrange
    state_predicates = depot_problem.initial_state_predicates
    state_fluents = depot_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    expected_predicate_in_state = "(at truck0 distributor1)"
    assert expected_predicate_in_state not in initial_state.serialize()
    assert "(= (fuel-cost ) 0.0)" in initial_state.serialize()
    action = ActionCall(name="drive", grounded_parameters=["truck0", "depot0", "distributor1"])
    expected_predicate_not_in_state = "(at truck0 depot0)"
    expected_numeric_fluent_in_state = "(= (fuel-cost ) 10.0)"

    # Act
    next_state, is_applicable, reward = depot_numeric_agent.observe(initial_state, action)

    # Assert
    assert expected_predicate_in_state in next_state.serialize()
    assert expected_numeric_fluent_in_state in next_state.serialize()
    assert expected_predicate_not_in_state not in next_state.serialize()
    assert is_applicable


def test_observe_on_state_with_inapplicable_action_returns_the_same_state_as_before_on_numeric_domains_and_the_agent_recalls_the_action_was_inapplicable(
    depot_problem: Problem, depot_numeric_agent: IPCAgent
):
    # Arrange
    state_predicates = depot_problem.initial_state_predicates
    state_fluents = depot_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot0"])

    # Act
    next_state, is_applicable, reward = depot_numeric_agent.observe(initial_state, action)

    # Assert
    assert next_state == initial_state
    assert not is_applicable


def test_get_environment_actions_gets_the_correct_number_of_grounded_actions(
    minecraft_agent: IPCAgent, minecraft_large_problem: Problem
):
    # Arrange
    state_predicates = minecraft_large_problem.initial_state_predicates
    state_fluents = minecraft_large_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    num_expected_actions = 1442
    total_grounded_actions = minecraft_agent.get_environment_actions(initial_state)

    assert len(total_grounded_actions) == num_expected_actions


def test_execute_plan_when_plan_is_valid_and_goal_was_reached_returns_trace_of_same_size_as_plan_with_all_transitions_applicable_and_the_goal_was_reached(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM, domain=depot_numeric_domain).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))

    # Act
    trace, goal_reached = depot_numeric_agent.execute_plan(plan)

    # Assert
    assert len(trace) == len(plan)
    assert goal_reached
    assert all([component.is_successful for component in trace.components])


def test_execute_plan_when_plan_is_valid_but_last_action_does_not_reach_goal_returns_trace_with_successful_transitions_but_goal_not_reached(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM, domain=depot_numeric_domain).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))

    # Act
    trace, goal_reached = depot_numeric_agent.execute_plan(plan[:-1])

    # Assert
    assert len(trace) == len(plan) - 1
    assert not goal_reached
    assert all([component.is_successful for component in trace.components])


def test_execute_plan_when_plan_contains_invalid_action_returns_trace_with_length_of_actions_until_failure_and_goal_not_reached(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM, domain=depot_numeric_domain).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))
    test_failing_action = ActionCall(name="lift", grounded_parameters=["hoist5", "crate2", "pallet5", "distributor"])
    plan[9] = test_failing_action

    # Act
    trace, goal_reached = depot_numeric_agent.execute_plan(plan)

    # Assert
    assert len(trace) == 10
    assert not goal_reached
    assert all([component.is_successful for component in trace.components[:-1]])
    assert trace.components[-1].is_successful is False


def test_execute_plan_adds_problem_objects_to_trace_when_trace_created(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM, domain=depot_numeric_domain).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))

    # Act
    trace, goal_reached = depot_numeric_agent.execute_plan(plan)

    # Assert
    assert len(trace.grounded_objects) > 0


def test_goal_reached_when_goal_is_only_numeric_correctly_evaluates_that_goal_was_reached_in_the_observed_state(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(
        problem_path=DEPOT_ONLINE_LEARNING_PROBLEM_WITH_NUMERIC_GOAL, domain=depot_numeric_domain
    ).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    state_predicates = problem.initial_state_predicates
    state_fluents = problem.initial_state_fluents
    state_fluents["(current_load truck0)"].set_value(200.0)
    state_fluents["(current_load truck1)"].set_value(100.0)
    state_containing_goal_fluents = State(predicates=state_predicates, fluents=state_fluents)
    assert depot_numeric_agent.goal_reached(state_containing_goal_fluents)


def test_goal_reached_when_goal_is_only_numeric_and_goal_fluents_not_matching_goal_requirements_returns_goal_not_reached(
    depot_numeric_agent: IPCAgent, depot_numeric_domain: Domain
):
    # Arrange
    problem = ProblemParser(
        problem_path=DEPOT_ONLINE_LEARNING_PROBLEM_WITH_NUMERIC_GOAL, domain=depot_numeric_domain
    ).parse_problem()
    depot_numeric_agent.initialize_problem(problem)
    state_predicates = problem.initial_state_predicates
    state_fluents = problem.initial_state_fluents
    state_fluents["(current_load truck0)"].set_value(99.0)
    state_fluents["(current_load truck1)"].set_value(15.0)
    state_containing_goal_fluents = State(predicates=state_predicates, fluents=state_fluents)
    assert not depot_numeric_agent.goal_reached(state_containing_goal_fluents)

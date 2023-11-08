"""Module test for the IPC active learning agent."""
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Problem, State, ActionCall, GroundedPredicate, NumericalExpressionTree, \
    construct_expression_tree
from pytest import fixture

from experiments.ipc_agent import IPCAgent
from tests.consts import DEPOTS_NUMERIC_DOMAIN_PATH


@fixture()
def depot_domain() -> Domain:
    domain_parser = DomainParser(DEPOTS_NUMERIC_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture
def depot_discrete_agent(depot_discrete_domain: Domain, depot_discrete_problem: Problem) -> IPCAgent:
    return IPCAgent(depot_discrete_domain, depot_discrete_problem)


@fixture
def depot_numeric_agent(depot_domain: Domain, depot_problem: Problem) -> IPCAgent:
    return IPCAgent(depot_domain, depot_problem)


def test_observe_on_state_with_applicable_action_returns_correct_next_state_only_discrete(
        depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)
    expected_predicate_in_state = "(at truck1 distributor1)"
    assert expected_predicate_in_state not in initial_state.serialize()
    action = ActionCall(name="drive", grounded_parameters=["truck1", "depot0", "distributor1"])
    expected_predicate_not_in_state = "(at truck1 depot0)"

    # Act
    next_state, reward = depot_discrete_agent.observe(initial_state, action)

    # Assert
    assert expected_predicate_in_state in next_state.serialize()
    assert expected_predicate_not_in_state not in next_state.serialize()


def test_observe_on_state_with_inapplicable_action_returns_the_same_state_as_before_only_discrete(
        depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck1", "distributor1", "depot0"])

    # Act
    next_state, reward  = depot_discrete_agent.observe(initial_state, action)

    # Assert
    initial_state_predicates = {p.untyped_representation for predicates in state_predicates.values() for p in
                                predicates}
    assert initial_state_predicates == {p.untyped_representation for predicates in next_state.state_predicates.values()
                                        for p in predicates}


def test_observe_on_state_with_applicable_action_returns_correct_next_state_numeric(
        depot_problem: Problem, depot_numeric_agent: IPCAgent):
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
    next_state, reward  = depot_numeric_agent.observe(initial_state, action)

    # Assert
    assert expected_predicate_in_state in next_state.serialize()
    assert expected_numeric_fluent_in_state in next_state.serialize()
    assert expected_predicate_not_in_state not in next_state.serialize()


def test_observe_on_state_with_inapplicable_action_returns_the_same_state_as_before_numeric(
        depot_problem: Problem, depot_numeric_agent: IPCAgent):
    # Arrange
    state_predicates = depot_problem.initial_state_predicates
    state_fluents = depot_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot0"])

    # Act
    next_state, reward  = depot_numeric_agent.observe(initial_state, action)

    # Assert
    initial_state_predicates = {p.untyped_representation for predicates in state_predicates.values() for p in
                                predicates}
    assert initial_state_predicates == {p.untyped_representation for predicates in next_state.state_predicates.values()
                                        for p in predicates}

    initial_state_fluents = {fluent.untyped_representation for fluent in state_fluents.values()}
    assert initial_state_fluents == {fluent.untyped_representation for fluent in next_state.state_fluents.values()}


def test_get_reward_returns_correct_reward_one_only_discrete(
        depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent, depot_discrete_domain: Domain):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)

    goal_predicates = {
        GroundedPredicate(name="on", signature=depot_discrete_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate0", "?y": "pallet2"}),
        GroundedPredicate(name="on", signature=depot_discrete_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate1", "?y": "pallet1"})
    }

    goal_state = initial_state.copy()
    goal_state.state_predicates[depot_discrete_domain.predicates["on"].untyped_representation].update(goal_predicates)

    # Act
    reward = depot_discrete_agent.goal_reached(goal_state)

    # Assert
    assert reward


def test_get_reward_returns_correct_reward_zero_only_discrete(
        depot_discrete_problem: Problem, depot_discrete_agent: IPCAgent, depot_discrete_domain: Domain):
    # Arrange
    state_predicates = depot_discrete_problem.initial_state_predicates
    initial_state = State(predicates=state_predicates, fluents={}, is_init=True)

    goal_predicates = {
        GroundedPredicate(name="on", signature=depot_discrete_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate0", "?y": "pallet2"}),
        GroundedPredicate(name="on", signature=depot_discrete_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate1", "?y": "pallet0"})
    }

    goal_state = initial_state.copy()
    goal_state.state_predicates[depot_discrete_domain.predicates["on"].untyped_representation].update(goal_predicates)

    # Act
    reward = depot_discrete_agent.goal_reached(goal_state)

    # Assert
    assert not reward


def test_get_reward_returns_correct_reward_one_when_goal_includes_numeric_conditions(
        depot_problem: Problem, depot_numeric_agent: IPCAgent, depot_domain: Domain):
    # Arrange
    state_predicates = depot_problem.initial_state_predicates
    state_fluents = depot_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck0", "depot0", "distributor1"])
    numeric_goal_components = [">", ["fuel-cost"], "0.0"]
    numeric_expression = NumericalExpressionTree(
        construct_expression_tree(numeric_goal_components, domain_functions=depot_domain.functions))
    depot_problem.goal_state_fluents.add(numeric_expression)

    assert not depot_numeric_agent.goal_reached(initial_state)
    # Act
    next_state, reward = depot_numeric_agent.observe(initial_state, action)
    goal_predicates = {
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate0", "?y": "pallet2"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate1", "?y": "crate3"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate2", "?y": "pallet0"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate3", "?y": "pallet1"}),
    }

    # Assert
    next_state.state_predicates[depot_domain.predicates["on"].untyped_representation].update(goal_predicates)
    assert depot_numeric_agent.goal_reached(next_state)


def test_get_reward_returns_correct_reward_zero_when_goal_includes_numeric_conditions_but_goal_not_reached(
        depot_problem: Problem, depot_numeric_agent: IPCAgent, depot_domain: Domain):
    # Arrange
    state_predicates = depot_problem.initial_state_predicates
    state_fluents = depot_problem.initial_state_fluents
    initial_state = State(predicates=state_predicates, fluents=state_fluents, is_init=True)
    action = ActionCall(name="drive", grounded_parameters=["truck0", "depot0", "distributor1"])
    numeric_goal_components = [">", ["fuel-cost"], "20.0"]
    numeric_expression = NumericalExpressionTree(
        construct_expression_tree(numeric_goal_components, domain_functions=depot_domain.functions))
    depot_problem.goal_state_fluents.add(numeric_expression)

    assert not depot_numeric_agent.goal_reached(initial_state)

    # Act
    next_state, reward = depot_numeric_agent.observe(initial_state, action)
    goal_predicates = {
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate0", "?y": "pallet2"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate1", "?y": "crate3"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate2", "?y": "pallet0"}),
        GroundedPredicate(name="on", signature=depot_domain.predicates["on"].signature,
                          object_mapping={"?x": "crate3", "?y": "pallet1"}),
    }

    # Assert
    next_state.state_predicates[depot_domain.predicates["on"].untyped_representation].update(goal_predicates)
    assert not depot_numeric_agent.goal_reached(next_state)
    assert next_state.state_fluents["(fuel-cost )"].value == 10.0

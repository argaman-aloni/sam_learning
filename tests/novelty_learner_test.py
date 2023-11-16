"""Module test for the novelty learner module."""

import pytest
from pddl_plus_parser.models import Observation, Domain

from sam_learning.core import NoveltyLearner


@pytest.fixture
def novelty_learner() -> NoveltyLearner:
    """Fixture to initialize the novelty learner."""
    return NoveltyLearner()


def test_add_sample_to_execution_db_successfully_adds_the_new_sample(
        novelty_learner: NoveltyLearner, minecraft_observation: Observation) -> None:
    """Tests adding a sample to the execution DB."""
    first_state = minecraft_observation.components[0].previous_state
    first_action = minecraft_observation.components[0].grounded_action_call

    state_predicates = {predicate.untyped_representation for predicates in
                        first_state.state_predicates.values() for predicate in predicates}

    novelty_learner.add_sample_to_execution_db(first_action.name, first_state, 1)
    assert novelty_learner._execution_db["lifted_action"] == [first_action.name]
    assert novelty_learner._execution_db["state_predicates"] == [state_predicates]
    state_functions = {func.state_representation for func in first_state.state_fluents.values()}
    db_functions = {func.state_representation for func in novelty_learner._execution_db["functions_and_values"][0]}
    assert state_functions == db_functions
    assert novelty_learner._execution_db["execution_result"] == [1]


def test_compute_state_l1_norm_when_states_are_the_same_returns_zero(
        novelty_learner: NoveltyLearner, minecraft_observation: Observation) -> None:
    """Tests computing the L1 norm of the state when the states are the same."""
    first_state = minecraft_observation.components[0].previous_state
    first_action = minecraft_observation.components[0].grounded_action_call

    state_predicates = {predicate.untyped_representation for predicates in
                        first_state.state_predicates.values() for predicate in predicates}

    novelty_learner.add_sample_to_execution_db(first_action.name, first_state, 1)
    assert novelty_learner._compute_state_l1_norm(first_state.state_fluents, state_predicates, 0) == 0


def test_compute_state_l1_norm_when_states_are_different_returns_value_larger_than_zero(
        novelty_learner: NoveltyLearner, minecraft_observation: Observation) -> None:
    """Tests computing the L1 norm of the state when the states are different."""
    first_state = minecraft_observation.components[0].previous_state
    first_action = minecraft_observation.components[0].grounded_action_call
    next_state = minecraft_observation.components[0].next_state

    state_predicates = {predicate.untyped_representation for predicates in
                        next_state.state_predicates.values() for predicate in predicates}

    novelty_learner.add_sample_to_execution_db(first_action.name, first_state, 1)
    assert novelty_learner._compute_state_l1_norm(next_state.state_fluents, state_predicates, 0) > 0


def test_calculate_novelty_when_the_action_was_not_observed_returns_the_default_exploration_rate(
        novelty_learner: NoveltyLearner, minecraft_observation: Observation, minecraft_full_domain: Domain) -> None:
    """Tests calculating the novelty when the action was not observed."""
    first_state = minecraft_observation.components[0].previous_state
    first_action = minecraft_observation.components[0].grounded_action_call
    next_state = minecraft_observation.components[0].next_state

    novelty_learner.add_sample_to_execution_db(first_action.name, first_state, -1)
    assert novelty_learner.calculate_novelty(
        action=first_action, state=next_state, domain=minecraft_full_domain, observed_actions=[]) == 1000


def test_calculate_novelty_when_the_action_has_been_observed_and_is_learned_returns_the_value_of_the_l1_norm_of_the_state_the_action_was_observed_at(
        novelty_learner: NoveltyLearner, minecraft_observation: Observation, minecraft_full_domain: Domain) -> None:
    """Tests calculating the novelty when the action was observed and is learned."""
    first_state = minecraft_observation.components[0].previous_state
    first_action = minecraft_observation.components[0].grounded_action_call

    novelty_learner.add_sample_to_execution_db(first_action.name, first_state, 1)
    assert novelty_learner.calculate_novelty(
        action=first_action, state=first_state, domain=minecraft_full_domain,
        observed_actions=[first_action.name]) < 1000

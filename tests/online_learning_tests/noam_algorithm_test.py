"""Module test for the online_nsam module."""
import logging
import shutil
import time
from pathlib import Path
from queue import PriorityQueue

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, State, ActionCall, Predicate, Precondition
from pytest import fixture

from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners import NumericOnlineActionModelLearner
from sam_learning.learners.noam_algorithm import ExplorationAlgorithmType
from solvers import ENHSPSolver
from tests.consts import DEPOTS_NUMERIC_DOMAIN_PATH, create_plan_actions, DEPOT_ONLINE_LEARNING_PLAN, \
    DEPOT_ONLINE_LEARNING_PROBLEM


@fixture()
def working_directory():
    current_directory = Path(__file__).parent
    workdir = current_directory / "working_directory"
    workdir.mkdir(parents=True, exist_ok=True)
    yield workdir
    # teardown
    shutil.rmtree(workdir, ignore_errors=True)


@fixture()
def depot_numeric_domain() -> Domain:
    domain_parser = DomainParser(DEPOTS_NUMERIC_DOMAIN_PATH, partial_parsing=False)
    return domain_parser.parse_domain()


@fixture()
def depot_problem(depot_numeric_domain: Domain) -> Problem:
    problem_parser = ProblemParser(DEPOT_ONLINE_LEARNING_PROBLEM, depot_numeric_domain)
    return problem_parser.parse_problem()


@fixture
def depot_numeric_agent(depot_numeric_domain: Domain, depot_problem: Problem) -> IPCAgent:
    agent = IPCAgent(depot_numeric_domain)
    agent.initialize_problem(depot_problem)
    return agent


@fixture()
def depot_noam_informative_explorer(depot_domain: Domain, working_directory: Path,
                                    depot_numeric_agent: IPCAgent) -> NumericOnlineActionModelLearner:
    return NumericOnlineActionModelLearner(
        workdir=working_directory,
        partial_domain=depot_domain,
        polynomial_degree=0,
        agent=depot_numeric_agent,
        solver=ENHSPSolver(),
        exploration_type=ExplorationAlgorithmType.informative_explorer,
    )


@fixture()
def depot_noam_goal_oriented(depot_domain: Domain, working_directory: Path,
                             depot_numeric_agent: IPCAgent) -> NumericOnlineActionModelLearner:
    return NumericOnlineActionModelLearner(
        workdir=working_directory,
        partial_domain=depot_domain,
        polynomial_degree=0,
        agent=depot_numeric_agent,
        solver=ENHSPSolver(),
        exploration_type=ExplorationAlgorithmType.goal_oriented,
    )

@fixture()
def depot_noam_goal_combined_explorer(depot_domain: Domain, working_directory: Path, depot_numeric_agent: IPCAgent) -> NumericOnlineActionModelLearner:
    return NumericOnlineActionModelLearner(
        workdir=working_directory,
        partial_domain=depot_domain,
        polynomial_degree=0,
        agent=depot_numeric_agent,
        solver=ENHSPSolver(),
        exploration_type=ExplorationAlgorithmType.combined,
    )


def test_depot_noam_informative_explorer_class_initialization_does_not_fail(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner):
    assert depot_noam_informative_explorer is not None


def test_calculate_state_action_informative_state_when_state_observed_for_the_first_time_returns_true(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
):
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    first_action = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))[0]
    depot_noam_informative_explorer.initialize_learning_algorithms()
    is_informative, action_applicable = depot_noam_informative_explorer._calculate_state_action_informative(
        current_state=initial_state, action_to_test=first_action, problem_objects=depot_problem.objects
    )
    assert is_informative


def test_add_transition_data_when_failure_caused_by_discrete_condition_not_holding_adds_failure_to_discrete_model_and_not_to_numeric_and_undecided(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    first_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "distributor2"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    component = trace.components[0]
    depot_noam_informative_explorer.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=component.grounded_action_call,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_informative_explorer._add_transition_data(action_to_update=first_action, is_transition_successful=True)
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].must_be_preconditions) == 0

    failed_action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot3"])
    depot_noam_informative_explorer.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=failed_action,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_informative_explorer._add_transition_data(action_to_update=failed_action, is_transition_successful=False)
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].must_be_preconditions) == 1
    assert len(depot_noam_informative_explorer._numeric_models_learners[first_action.name]._svm_learner.data) == 1
    assert len(depot_noam_informative_explorer.undecided_failure_observations[first_action.name]) == 0


def test_add_transition_data_when_transition_is_not_successful_adds_data_to_all_model_learners(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    failed_first_action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot3"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[failed_first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    component = trace.components[0]
    depot_noam_informative_explorer.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=component.grounded_action_call,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_informative_explorer._add_transition_data(action_to_update=failed_first_action,
                                                         is_transition_successful=False)
    assert len(depot_noam_informative_explorer._discrete_models_learners[
                   failed_first_action.name].cannot_be_preconditions) == 0
    assert len(
        depot_noam_informative_explorer._numeric_models_learners[failed_first_action.name]._svm_learner.data) == 0
    assert len(depot_noam_informative_explorer._informative_states_learner[failed_first_action.name].combined_data) == 1
    assert len(depot_noam_informative_explorer.undecided_failure_observations[failed_first_action.name]) == 1


def test_add_transition_data_when_transition_is_successful_adds_data_to_all_model_learners_and_not_to_undecided_data_structure(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    first_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "distributor2"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    component = trace.components[0]
    depot_noam_informative_explorer.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=component.grounded_action_call,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_informative_explorer._add_transition_data(action_to_update=first_action, is_transition_successful=True)
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].cannot_be_preconditions) > 0
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].cannot_be_effects) > 0
    assert len(depot_noam_informative_explorer._numeric_models_learners[first_action.name]._svm_learner.data) == 1
    assert len(depot_noam_informative_explorer._informative_states_learner[first_action.name].combined_data) == 1
    assert len(depot_noam_informative_explorer.undecided_failure_observations[first_action.name]) == 0


def test_select_action_and_execute_when_when_there_is_a_single_applicable_action_in_the_frontier_executes_the_action_and_returns_successive_state(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_informative_explorer.initialize_learning_algorithms()
    frontier = [ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "distributor2"])]
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    selected_action, is_successful, next_state = depot_noam_informative_explorer._select_action_and_execute(
        current_state=initial_state, frontier=frontier, problem_objects=depot_problem.objects)
    assert selected_action.name == "drive"
    assert is_successful
    assert next_state is not None
    assert next_state != initial_state


def test_select_action_and_execute_when_when_there_is_a_single_inapplicable_action_in_the_frontier_executes_the_action_and_returns_same_state(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_informative_explorer.initialize_learning_algorithms()
    frontier = [ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot3"])]
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    selected_action, is_successful, next_state = depot_noam_informative_explorer._select_action_and_execute(
        current_state=initial_state, frontier=frontier, problem_objects=depot_problem.objects)
    assert selected_action.name == "drive"
    assert not is_successful
    assert next_state is not None
    assert next_state == initial_state


def test_select_action_and_execute_when_when_there_is_a_single_inapplicable_action_in_the_frontier_executes_the_action_and_reduces_frontier_size(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_informative_explorer.initialize_learning_algorithms()
    frontier = [ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot3"])]
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    selected_action, is_successful, next_state = depot_noam_informative_explorer._select_action_and_execute(
        current_state=initial_state, frontier=frontier, problem_objects=depot_problem.objects)
    assert selected_action.name == "drive"
    assert not is_successful
    assert len(frontier) == 0


def test_select_action_and_execute_when_exploration_policy_is_goal_oriented_selects_random_action_from_frontier_even_though_it_is_not_informative(
        depot_noam_goal_oriented: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_noam_informative_explorer: NumericOnlineActionModelLearner,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_goal_oriented.initialize_learning_algorithms()
    depot_noam_informative_explorer.initialize_learning_algorithms()
    first_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "distributor2"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[first_action])
    component = trace.components[0]
    depot_noam_goal_oriented.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=component.grounded_action_call,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_informative_explorer.triplet_snapshot.create_triplet_snapshot(
        previous_state=component.previous_state,
        next_state=component.next_state,
        current_action=component.grounded_action_call,
        observation_objects=trace.grounded_objects,
    )
    depot_noam_goal_oriented._add_transition_data(action_to_update=first_action, is_transition_successful=True)
    depot_noam_informative_explorer._add_transition_data(action_to_update=first_action, is_transition_successful=True)

    frontier = [ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "distributor2"]),
                ActionCall(name="lift", grounded_parameters=["hoist5", "crate2", "crate0", "distributor1"])]
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    selected_action, _, _ = depot_noam_goal_oriented._select_action_and_execute(
        current_state=initial_state, frontier=frontier, problem_objects=depot_problem.objects)
    informative_selected_action, _, _ = depot_noam_informative_explorer._select_action_and_execute(
        current_state=initial_state, frontier=frontier, problem_objects=depot_problem.objects)
    assert selected_action.name == "drive"
    assert informative_selected_action.name == "lift"


def test_train_models_using_trace_when_given_a_single_action_adds_the_action_data_to_all_model_learners(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    first_action = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))[0]
    depot_numeric_agent.initialize_problem(depot_problem)
    trace, _ = depot_numeric_agent.execute_plan(plan=[first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].cannot_be_effects) > 0
    assert len(depot_noam_informative_explorer._discrete_models_learners[first_action.name].cannot_be_preconditions) > 0
    assert len(
        depot_noam_informative_explorer._numeric_models_learners[first_action.name]._convex_hull_learner.data) > 0


def test_train_models_using_trace_when_given_an_already_observed_state_and_action_makes_the_action_to_be_not_informative(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    first_action = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))[0]
    trace, _ = depot_numeric_agent.execute_plan(plan=[first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    is_informative, action_applicable = depot_noam_informative_explorer._calculate_state_action_informative(
        current_state=initial_state, action_to_test=first_action, problem_objects=depot_problem.objects
    )
    assert is_informative
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    is_informative, action_applicable = depot_noam_informative_explorer._calculate_state_action_informative(
        current_state=initial_state, action_to_test=first_action, problem_objects=depot_problem.objects
    )
    assert not is_informative


def test_train_models_using_trace_when_given_an_inapplicable_action_in_a_state_with_no_prior_observations_adds_observation_to_uncertain_failure_dataset(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)
    failed_first_action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor1", "depot3"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[failed_first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    is_informative, action_applicable = depot_noam_informative_explorer._calculate_state_action_informative(
        current_state=initial_state, action_to_test=failed_first_action, problem_objects=depot_problem.objects
    )
    assert is_informative
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    assert len(depot_noam_informative_explorer.undecided_failure_observations[failed_first_action.name]) > 0


def test_train_models_using_trace_when_given_an_inapplicable_action_and_then_the_action_was_correctly_applied_in_the_state_removes_action_from_undecided_observations(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    failed_first_action = ActionCall(name="drive", grounded_parameters=["truck0", "distributor2", "depot3"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[failed_first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    assert len(depot_noam_informative_explorer.undecided_failure_observations[failed_first_action.name]) == 1
    fixed_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "depot0"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[fixed_action])
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    assert len(depot_noam_informative_explorer.undecided_failure_observations[failed_first_action.name]) == 0


def test_train_models_using_trace_when_given_an_inapplicable_action_and_then_the_action_was_correctly_creates_optimistic_model_with_or_conditions_for_failed_action(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    failed_first_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot2", "depot3"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[failed_first_action])
    depot_noam_informative_explorer.initialize_learning_algorithms()
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    assert len(depot_noam_informative_explorer.undecided_failure_observations[failed_first_action.name]) == 1
    fixed_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "depot0"])
    trace, _ = depot_numeric_agent.execute_plan(plan=[fixed_action])
    depot_noam_informative_explorer.train_models_using_trace(trace=trace)
    optimistic_model = depot_noam_informative_explorer._construct_optimistic_action_model()
    optimistic_conditions = optimistic_model.actions[failed_first_action.name].preconditions.root.operands.pop()
    assert isinstance(optimistic_conditions, Precondition)
    assert optimistic_conditions.binary_operator == "or"
    print(str(optimistic_conditions))


def test_train_models_using_trace_when_given_multiple_successful_transitions_does_not_fail_and_optimistic_model_and_safe_model_can_be_constructed(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    try:
        plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))
        trace, _ = depot_numeric_agent.execute_plan(plan=plan)
        depot_noam_informative_explorer.initialize_learning_algorithms()
        depot_noam_informative_explorer.train_models_using_trace(trace=trace)
        safe_model = depot_noam_informative_explorer._construct_safe_action_model()
        optimistic_model = depot_noam_informative_explorer._construct_optimistic_action_model()
        print("Safe model:\n", safe_model.to_pddl())
        print("Optimistic model:\n", optimistic_model.to_pddl())
    except Exception as e:
        assert False, e


def test_train_models_using_trace_when_given_multiple_successful_transitions_returns_optimistic_model_without_numeric_preconditions_as_only_positive_samples_are_available(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent
):
    try:
        plan = create_plan_actions(Path(DEPOT_ONLINE_LEARNING_PLAN))
        trace, _ = depot_numeric_agent.execute_plan(plan=plan)
        depot_noam_informative_explorer.initialize_learning_algorithms()
        depot_noam_informative_explorer.train_models_using_trace(trace=trace)
        safe_model = depot_noam_informative_explorer._construct_safe_action_model()
        optimistic_model = depot_noam_informative_explorer._construct_optimistic_action_model()
        print("Safe model:\n", safe_model.to_pddl())
        print("Optimistic model:\n", optimistic_model.to_pddl())
    except Exception as e:
        assert False, e


def test_construct_safe_action_model_when_no_observation_given_does_not_fail(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner):
    try:
        depot_noam_informative_explorer.initialize_learning_algorithms()
        depot_noam_informative_explorer._construct_safe_action_model()
    except Exception as e:
        assert False, e


def test_construct_optimistic_action_model_when_no_observation_given_does_not_fail(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner):
    try:
        depot_noam_informative_explorer.initialize_learning_algorithms()
        depot_noam_informative_explorer._construct_optimistic_action_model()
    except Exception as e:
        assert False, e


def test_explore_to_refine_models_changes_the_models_after_short_episode_is_done_and_does_not_take_extremely_long_to_finish(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem, depot_domain: Domain,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)

    depot_noam_informative_explorer.initialize_learning_algorithms()
    start_time = time.time()
    num_steps_done = depot_noam_informative_explorer.explore_to_refine_models(
        init_state=initial_state,
        num_steps_till_episode_end=10,
        problem_objects=depot_problem.objects,
    )
    end_time = time.time()
    assert num_steps_done <= 100
    assert end_time - start_time < 60, "Exploration took too long to finish"
    print("Safe model:\n", depot_noam_informative_explorer._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_informative_explorer._construct_optimistic_action_model().to_pddl())


def test_explore_to_refine_models_changes_the_models_after_long_episode_is_done_updates_models_and_safe_and_optimistic_domains_are_available(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem, depot_domain: Domain,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)

    depot_noam_informative_explorer.initialize_learning_algorithms()
    num_steps_done = depot_noam_informative_explorer.explore_to_refine_models(
        init_state=initial_state,
        num_steps_till_episode_end=10000,
        problem_objects=depot_problem.objects,
    )
    assert num_steps_done <= 10000
    print("Safe model:\n", depot_noam_informative_explorer._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_informative_explorer._construct_optimistic_action_model().to_pddl())


def test_explore_to_refine_models_when_using_goal_oriented_exploration_does_not_fail(
        depot_noam_goal_oriented: NumericOnlineActionModelLearner, depot_problem: Problem, depot_domain: Domain,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    initial_state = State(predicates=depot_problem.initial_state_predicates,
                          fluents=depot_problem.initial_state_fluents)

    depot_noam_goal_oriented.initialize_learning_algorithms()
    num_steps_done = depot_noam_goal_oriented.explore_to_refine_models(
        init_state=initial_state,
        num_steps_till_episode_end=10000,
        problem_objects=depot_problem.objects,
    )
    assert num_steps_done <= 10000
    print("Safe model:\n", depot_noam_goal_oriented._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_goal_oriented._construct_optimistic_action_model().to_pddl())


def test_apply_exploration_policy_when_exploration_policy_is_informative_explorer_applies_policy_and_returns_less_than_max_number_of_steps_for_episode(
        depot_noam_informative_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_informative_explorer.initialize_learning_algorithms()
    num_steps = depot_noam_informative_explorer.apply_exploration_policy(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM)
    assert num_steps <= 5000
    print("Safe model:\n", depot_noam_informative_explorer._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_informative_explorer._construct_optimistic_action_model().to_pddl())

def test_apply_exploration_policy_when_exploration_policy_is_goal_oriented_and_planner_is_none_fails_as_it_tries_to_call_planner_with_optimistic_domain(
        depot_noam_goal_oriented: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_goal_oriented.initialize_learning_algorithms()
    depot_noam_goal_oriented._solver = None  # Set solver to None to simulate the failure case
    with pytest.raises(AttributeError):
        depot_noam_goal_oriented.apply_exploration_policy(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM)

def test_apply_exploration_policy_when_exploration_policy_is_goal_oriented_and_planner_is_set_applies_policy_and_returns_less_than_max_number_of_steps_for_episode(
        depot_noam_goal_oriented: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    # NOTE: This test assumes that the planner is set and that the environment variable `ENHSP_FILE_PATH` is configured correctly.
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_goal_oriented.initialize_learning_algorithms()
    num_steps = depot_noam_goal_oriented.apply_exploration_policy(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM)
    assert num_steps <= 5000
    print("Safe model:\n", depot_noam_goal_oriented._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_goal_oriented._construct_optimistic_action_model().to_pddl())


def test_apply_exploration_policy_when_exploration_policy_combined_and_planner_is_set_applies_policy_and_returns_less_than_max_number_of_steps_for_episode(
        depot_noam_goal_combined_explorer: NumericOnlineActionModelLearner, depot_problem: Problem,
        depot_numeric_agent: IPCAgent):
    # NOTE: This test assumes that the planner is set and that the environment variable `ENHSP_FILE_PATH` is configured correctly.
    depot_numeric_agent.initialize_problem(depot_problem)
    depot_noam_goal_combined_explorer.initialize_learning_algorithms()
    num_steps = depot_noam_goal_combined_explorer.apply_exploration_policy(problem_path=DEPOT_ONLINE_LEARNING_PROBLEM)
    assert num_steps <= 5000
    print("Safe model:\n", depot_noam_goal_combined_explorer._construct_safe_action_model().to_pddl())
    print("Optimistic model:\n", depot_noam_goal_combined_explorer._construct_optimistic_action_model().to_pddl())
import shutil
import time
from pathlib import Path

from pandas import DataFrame
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, State, ActionCall, Observation
from pytest import fixture

from sam_learning.core import EpisodeInfoRecord
from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners.semi_online_learning_algorithm import SemiOnlineNumericAMLearner, MAX_SUCCESSFUL_STEPS_PER_EPISODE
from solvers import ENHSPSolver
from tests.consts import (
    DEPOTS_NUMERIC_DOMAIN_PATH,
    DEPOT_ONLINE_LEARNING_PROBLEM,
    DEPOTS_NUMERIC_EMPTY_DOMAIN_PATH,
    DEPOT_ONLINE_LEARNING_PROD_BUG_PROBLEM,
    DEPOT_ONLINE_LEARNING_PROD_BUG_TRAJECTORY,
)


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
def episode_info_recorder(depot_numeric_domain: Domain, working_directory: Path) -> EpisodeInfoRecord:
    return EpisodeInfoRecord(
        action_names=list(depot_numeric_domain.actions),
        working_directory=working_directory,
    )


@fixture()
def depot_semi_online_learner(
    depot_domain: Domain, working_directory: Path, depot_numeric_agent: IPCAgent, episode_info_recorder: EpisodeInfoRecord
) -> SemiOnlineNumericAMLearner:
    return SemiOnlineNumericAMLearner(
        workdir=working_directory,
        partial_domain=depot_domain,
        polynomial_degree=0,
        agent=depot_numeric_agent,
        solvers=[ENHSPSolver()],
        episode_recorder=episode_info_recorder,
    )


@fixture()
def depot_effects_bug_problem(depot_numeric_domain: Domain) -> Problem:
    return ProblemParser(problem_path=DEPOT_ONLINE_LEARNING_PROD_BUG_PROBLEM, domain=depot_numeric_domain).parse_problem()


@fixture()
def depot_effects_bug_trajectory(depot_numeric_domain: Domain, depot_effects_bug_problem: Problem) -> Observation:
    """Fixture to create a trajectory that reproduces the effects bug."""
    observation = TrajectoryParser(partial_domain=depot_numeric_domain, problem=depot_effects_bug_problem).parse_trajectory(
        trajectory_file_path=DEPOT_ONLINE_LEARNING_PROD_BUG_TRAJECTORY, contain_transitions_status=True
    )
    return observation


def test_sort_ground_actions_based_on_success_rate_does_not_fail_when_no_observations_are_given(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the sort_ground_actions_based_on_success method correctly sorts actions based on success rate."""
    # Create a mock action model with some actions and their success rates
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    grounded_actions = depot_numeric_agent.get_environment_actions(init_state)

    sorted_actions = depot_semi_online_learner.sort_ground_actions_based_on_success_rate(grounded_actions)
    assert sorted_actions is not None


def test_sort_ground_actions_based_on_success_rate_correctly_sorts_actions_by_success_rate(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the sort_ground_actions_based_on_success method correctly sorts actions based on success rate."""
    # Create a mock action model with some actions and their success rates
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    grounded_actions = depot_numeric_agent.get_environment_actions(init_state)

    # setting the success rates for the actions
    episode_info_recorder._action_successful_execution_history = {
        "num_drive_success": 50,
        "num_lift_success": 4,
        "num_drop_success": 10,
        "num_load_success": 15,
        "num_unload_success": 2,
    }
    sorted_actions = depot_semi_online_learner.sort_ground_actions_based_on_success_rate(grounded_actions)
    assert sorted_actions[0].name == "unload"
    assert sorted_actions[-1].name == "drive"


def test_sort_ground_actions_based_on_success_rate_when_updating_a_single_transition_returns_correct_transition_even_with_multiple_executions(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the sort_ground_actions_based_on_success method correctly sorts actions based on success rate."""
    # Create a mock action model with some actions and their success rates
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    grounded_actions = depot_numeric_agent.get_environment_actions(init_state)
    drive_action = next(action for action in grounded_actions if action.name == "drive")

    # setting the success rates for the actions
    episode_info_recorder.record_single_step(action=drive_action, action_applicable=True, previous_state=init_state, next_state=init_state)
    sorted_actions = depot_semi_online_learner.sort_ground_actions_based_on_success_rate(grounded_actions)
    assert sorted_actions[-1].name == "drive"
    sorted_actions = depot_semi_online_learner.sort_ground_actions_based_on_success_rate(grounded_actions)
    assert sorted_actions[-1].name == "drive"


def test_eliminate_undecided_observations_when_no_undecided_observations_does_not_call_model_learners(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    # The learners are not initialized so we can check if the method does not call them by checking that no exception is raised.
    try:
        depot_semi_online_learner._eliminate_undecided_observations("lift")
    except Exception:
        assert False, "The method should not raise an exception when there are no undecided observations."


def test_extract_parameter_bound_state_data_returns_correct_parameter_bound_predicates(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the extract_parameter_bound_state_data method returns the correct parameter bound predicates."""
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    # tested action - (drive truck1 depot3 distributor1)
    ground_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot3", "depot1"])
    state_predicates = {predicate for predicates in init_state.state_predicates.values() for predicate in predicates}
    print(state_predicates)
    pb_predicate, pb_functions = depot_semi_online_learner._extract_parameter_bound_state_data(
        ground_action, state_predicates=state_predicates, state_functions=init_state.state_fluents
    )
    assert len(pb_functions) == 3  # (load_limit ?t - truck), (current_load ?t - truck), (fuel-cost )


def test_execute_selected_action_returns_correct_next_state_and_success_when_transition_is_successful(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the execute_selected_action method returns the correct next state and success."""
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    ground_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot3", "depot1"])

    next_state, is_successful = depot_semi_online_learner._execute_selected_action(
        ground_action, init_state, problem_objects=depot_problem.objects, integrate_in_models=False
    )

    assert isinstance(next_state, State)
    assert is_successful is True


def test_execute_selected_action_returns_correct_next_state_and_failure_when_transition_is_unsuccessful(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the execute_selected_action method returns the correct next state and success."""
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)
    ground_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot2", "depot1"])

    next_state, is_successful = depot_semi_online_learner._execute_selected_action(
        ground_action, init_state, problem_objects=depot_problem.objects, integrate_in_models=False
    )

    assert next_state == init_state
    assert is_successful is False


def test_select_action_and_execute_when_first_action_is_not_applicable_from_frontier_of_two_actions_clear_the_frontier_after_both_executed(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the select_action_and_execute method clears the frontier after both actions are executed."""
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)

    # The first action is not applicable
    first_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot2", "depot1"])
    second_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot3", "depot1"])

    # Execute the first action
    frontier = [first_action, second_action]
    next_state, is_successful, _ = depot_semi_online_learner._select_action_and_execute(
        current_state=init_state, frontier=frontier, problem_objects=depot_problem.objects
    )

    # Check that the frontier is cleared after both actions are executed
    assert len(frontier) == 0


def test_select_action_and_execute_when_first_action_is_applicable_from_frontier_of_two_actions_executes_only_first_action(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_numeric_agent: IPCAgent,
    depot_domain: Domain,
    depot_problem: Problem,
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the select_action_and_execute method clears the frontier after both actions are executed."""
    init_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)

    # The first action is not applicable
    first_action = ActionCall(name="drive", grounded_parameters=["truck1", "depot2", "depot1"])
    second_action = ActionCall(name="drive", grounded_parameters=["truck0", "depot3", "depot1"])

    # Execute the first action
    frontier = [second_action, first_action]
    next_state, is_successful, _ = depot_semi_online_learner._select_action_and_execute(
        current_state=init_state, frontier=frontier, problem_objects=depot_problem.objects
    )

    # Check that the frontier is cleared after both actions are executed
    assert len(frontier) == 1


def test_explore_to_refine_models_changes_the_models_after_short_episode_is_done_and_does_not_take_extremely_long_to_finish(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_problem: Problem,
    depot_domain: Domain,
    depot_numeric_agent: IPCAgent,
):
    depot_numeric_agent.initialize_problem(depot_problem)
    initial_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)

    depot_semi_online_learner.initialize_learning_algorithms()
    start_time = time.time()
    goal_reached, num_steps_done = depot_semi_online_learner.explore_to_refine_models(
        init_state=initial_state,
        num_steps_till_episode_end=10,
        problem_objects=depot_problem.objects,
    )
    end_time = time.time()
    assert not goal_reached, "Goal should not be reached in such a short episode"
    assert num_steps_done <= 100
    assert end_time - start_time < 60, "Exploration took too long to finish"


def test_explore_to_refine_models_executes_the_correct_number_of_successful_actions_in_the_episode(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_problem: Problem,
    depot_domain: Domain,
    depot_numeric_agent: IPCAgent,
):
    depot_numeric_agent.initialize_problem(depot_problem)
    initial_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)

    depot_semi_online_learner.initialize_learning_algorithms()
    depot_semi_online_learner.explore_to_refine_models(
        init_state=initial_state,
        num_steps_till_episode_end=10,
        problem_objects=depot_problem.objects,
    )
    num_successful_actions = depot_semi_online_learner.episode_recorder._episode_info["sum_successful_actions"]
    assert num_successful_actions == 10


def test_use_solvers_to_solve_problem_returns_state_when_no_solution_is_found(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_problem: Problem,
):
    """Test that the use_solvers_to_solve_problem method returns the state when no solution is found."""
    initial_state = State(predicates=depot_problem.initial_state_predicates, fluents=depot_problem.initial_state_fluents)

    # Simulate a situation where no solution is found
    solution_stat, _, result_state = depot_semi_online_learner._use_solvers_to_solve_problem(
        problem_path=DEPOT_ONLINE_LEARNING_PROBLEM,
        domain_path=DEPOTS_NUMERIC_EMPTY_DOMAIN_PATH,
        init_state=initial_state,
    )

    assert isinstance(result_state, State), "The returned state should be an instance of State"


def test_train_models_using_trace_using_trajectory_from_production_bug_returns_correct_effects_to_the_safe_action_model_when_trace_contains_non_injective_transitions(
    depot_semi_online_learner: SemiOnlineNumericAMLearner,
    depot_effects_bug_problem: Problem,
    depot_effects_bug_trajectory: Observation,
):
    """Test that the train_models_using_trace method correctly trains the safe action model."""
    # Train the models using the trajectory
    depot_semi_online_learner.initialize_learning_algorithms()
    depot_semi_online_learner.train_models_using_trace(depot_effects_bug_trajectory)

    # Check that the safe action model has been updated with the correct effects
    discrete_drive_action_model = depot_semi_online_learner._discrete_models_learners["drive"]
    effect_strings = [eff.untyped_representation for eff in discrete_drive_action_model.must_be_effects]
    assert "(not (at ?x ?y))" in effect_strings
    assert "(at ?x ?z)" in effect_strings
    assert len(effect_strings) == 2, "There should be exactly two effects for the drive action model"

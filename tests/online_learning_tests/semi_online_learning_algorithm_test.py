import shutil
from pathlib import Path

from pandas import DataFrame
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, Problem, State
from pytest import fixture

from sam_learning.core import EpisodeInfoRecord
from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners.semi_online_learning_algorithm import SemiOnlineNumericAMLearner
from solvers import ENHSPSolver
from tests.consts import (
    DEPOTS_NUMERIC_DOMAIN_PATH,
    DEPOT_ONLINE_LEARNING_PROBLEM,
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

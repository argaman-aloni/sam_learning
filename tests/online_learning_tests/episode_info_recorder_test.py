"""Module for testing the EpisodeInfoRecorder class."""

from pathlib import Path

from pddl_plus_parser.lisp_parsers import PDDLTokenizer, TrajectoryParser
from pytest import fixture, mark, raises
from pddl_plus_parser.models import State, ActionCall, Domain, Observation

from sam_learning.core import EpisodeInfoRecord


@fixture
def episode_info_recorder() -> EpisodeInfoRecord:
    """Fixture to create an instance of EpisodeInfoRecord for testing."""
    return EpisodeInfoRecord(action_names=["action1", "action2"], working_directory=Path("."))


@fixture
def depot_numeric_episode_info_recorder(depot_domain: Domain) -> EpisodeInfoRecord:
    """Fixture to create an instance of EpisodeInfoRecord for testing."""
    return EpisodeInfoRecord(action_names=list(depot_domain.actions), working_directory=Path("."))


def test_record_single_step_creates_a_transition_with_no_init_state_to_avoid_duplicated_inits(episode_info_recorder: EpisodeInfoRecord):
    """Test that the record_single_step method creates a transition with no init state to avoid duplicated inits."""
    previous_state = State(predicates={}, fluents={})
    next_state = State(predicates={}, fluents={})
    action = ActionCall(name="action1", grounded_parameters=[])

    episode_info_recorder.record_single_step(action=action, action_applicable=True, previous_state=previous_state, next_state=next_state)
    assert len(episode_info_recorder.trajectory.components) == 1
    assert episode_info_recorder.trajectory.components[0].previous_state.is_init is False
    assert episode_info_recorder.trajectory.components[0].next_state.is_init is False


def test_record_single_step_creates_a_transition_with_no_init_state_to_avoid_duplicated_inits_even_when_previous_state_and_next_state_are_init(
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the record_single_step method creates a transition with no init state to avoid duplicated inits."""
    previous_state = State(predicates={}, fluents={}, is_init=True)
    next_state = State(predicates={}, fluents={}, is_init=True)
    action = ActionCall(name="action1", grounded_parameters=[])

    episode_info_recorder.record_single_step(action=action, action_applicable=True, previous_state=previous_state, next_state=next_state)
    assert len(episode_info_recorder.trajectory.components) == 1
    assert episode_info_recorder.trajectory.components[0].previous_state.is_init is False
    assert episode_info_recorder.trajectory.components[0].next_state.is_init is False


def test_record_single_step_creates_a_transition_even_when_the_transition_is_not_successful(
    episode_info_recorder: EpisodeInfoRecord,
):
    """Test that the record_single_step method creates a transition with no init state to avoid duplicated inits."""
    previous_state = State(predicates={}, fluents={}, is_init=True)
    next_state = State(predicates={}, fluents={}, is_init=True)
    action = ActionCall(name="action1", grounded_parameters=[])

    # Record a step where the action is not applicable
    assert len(episode_info_recorder.trajectory.components) == 0
    episode_info_recorder.record_single_step(action=action, action_applicable=False, previous_state=previous_state, next_state=next_state)
    assert len(episode_info_recorder.trajectory.components) == 1


def test_record_single_step_increments_successful_action_count_when_action_is_successful(episode_info_recorder: EpisodeInfoRecord):
    """Test that the record_single_step method increments the successful action count when the action is successful."""
    previous_state = State(predicates={}, fluents={})
    next_state = State(predicates={}, fluents={})
    action = ActionCall(name="action1", grounded_parameters=[])

    episode_info_recorder.record_single_step(action, True, previous_state, next_state)

    assert episode_info_recorder._episode_info["num_action1_success"] == 1
    assert episode_info_recorder._episode_info["sum_successful_actions"] == 1


def test_record_single_step_increments_failed_action_count_when_action_is_not_successful(episode_info_recorder: EpisodeInfoRecord):
    """Test that the record_single_step method increments the failed action count when the action is not successful."""
    previous_state = State(predicates={}, fluents={})
    next_state = State(predicates={}, fluents={})
    action = ActionCall(name="action1", grounded_parameters=[])

    episode_info_recorder.record_single_step(action, False, previous_state, next_state)

    assert episode_info_recorder._episode_info["num_action1_fail"] == 1
    assert episode_info_recorder._episode_info["sum_failed_actions"] == 1


def test_export_episode_trajectory_export_trajectory_in_correct_format(episode_info_recorder: EpisodeInfoRecord):
    """Test that the export_episode_trajectory method exports the trajectory in the correct format."""
    previous_state = State(predicates={}, fluents={})
    next_state = State(predicates={}, fluents={})
    action = ActionCall(name="action1", grounded_parameters=[])

    episode_info_recorder.record_single_step(action, True, previous_state, next_state)

    # Export the trajectory
    trajectory_str = episode_info_recorder.export_episode_trajectory(test_mode=True)

    # Check if the file exists and is not empty
    assert trajectory_str is not None
    tokenizer = PDDLTokenizer(pddl_str=trajectory_str)
    try:
        tokens = tokenizer.tokenize()
        assert tokens is not None

    except Exception:
        raise AssertionError("The trajectory string is not parsable.")


def test_export_episode_trajectory_export_trajectory_in_correct_format_when_the_domain_and_trajectories_are_real(
    depot_numeric_episode_info_recorder: EpisodeInfoRecord, depot_observation: Observation, depot_domain: Domain
):
    """Test that the export_episode_trajectory method exports the trajectory in the correct format."""
    for transition in depot_observation.components:
        depot_numeric_episode_info_recorder.record_single_step(
            transition.grounded_action_call, transition.is_successful, transition.previous_state, transition.next_state
        )

    # Export the trajectory
    trajectory_str = depot_numeric_episode_info_recorder.export_episode_trajectory(test_mode=True)

    # Check if the file exists and is not empty
    assert trajectory_str is not None
    # mocking the _read_trajectory_file of the TrajectoryParser to return the PDDLTokenizer of the trajectory string
    observation = TrajectoryParser(partial_domain=depot_domain).parse_trajectory(
        trajectory_string=trajectory_str, contain_transitions_status=True
    )
    try:
        assert observation is not None

    except Exception:
        raise AssertionError("The trajectory string is not parsable.")

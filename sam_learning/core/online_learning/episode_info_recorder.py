"""class to record the information about each episode of the online learning process."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from pandas import DataFrame
from pddl_plus_parser.models import ActionCall, Observation, State

RECORD_COLUMNS = [
    "num_grounded_actions",
    "sum_failed_actions",
    "sum_successful_actions",
    "goal_reached",
    "num_steps_in_episode",
    "solver_solved_problem",
]


class EpisodeInfoRecord:
    """Records the information about the execution of each episode of the online learning process."""

    def __init__(self, action_names: List[str], working_directory: Path):
        self._episode_number = 0
        self._num_informative_actions_in_step = []
        self._action_names = action_names
        self._episode_info = {
            **{record_name: 0 for record_name in RECORD_COLUMNS},
            **{f"num_{action_name}_success": 0 for action_name in action_names},
            **{f"num_{action_name}_fail": 0 for action_name in action_names},
        }
        self.summarized_info = DataFrame(
            columns=[
                "episode_number",
                *RECORD_COLUMNS,
                *{f"num_{action_name}_success" for action_name in action_names},
                *{f"num_{action_name}_fail" for action_name in action_names},
                *{f"num_unknown_failed_transitions_{action_name}" for action_name in action_names},
            ]
        )
        self.trajectory = Observation()
        self.working_directory = working_directory

    def get_number_successful_action_executions(self, action_name: str) -> int:
        return sum(self.summarized_info[f"num_{action_name}_success"])

    def add_num_grounded_actions(self, num_grounded_actions: int) -> None:
        """Adds the number of grounded actions in the episode.

        :param num_grounded_actions: the number of grounded actions in the episode.
        """
        self._episode_info["num_grounded_actions"] = num_grounded_actions

    def record_single_step(
        self,
        action: ActionCall,
        action_applicable: bool,
        previous_state: State,
        next_state: State,
    ) -> None:
        """Records the result of a single step in the episode.

        :param action: the action that was executed.
        :param action_applicable: whether the action was applicable (i.e., succeeded).
        :param previous_state: the state before the action was executed.
        :param next_state: the state after the action was executed.
        """
        non_init_prev_state = State(predicates=previous_state.state_predicates, fluents=previous_state.state_fluents, is_init=False)
        non_init_next_state = State(predicates=next_state.state_predicates, fluents=next_state.state_fluents, is_init=False)
        self.trajectory.add_component(
            previous_state=non_init_prev_state, next_state=non_init_next_state, call=action, is_successful_transition=action_applicable
        )
        if action_applicable:
            self._episode_info[f"num_{action.name}_success"] += 1
            self._episode_info["sum_successful_actions"] += 1
            return

        self._episode_info[f"num_{action.name}_fail"] += 1
        self._episode_info["sum_failed_actions"] += 1

    def end_episode(
        self,
        undecided_states: Dict[str, List[Tuple]],
        goal_reached: bool,
        num_steps_in_episode: int,
        has_solved_solver_problem: bool = False,
    ) -> None:
        """Ends the episode by recording summary statistics, updating the episode counter,
        and exporting the episode trajectory. This method should be called at the end of each episode
        to ensure all relevant information is saved and the recorder is reset for the next episode.

        :param undecided_states: a dictionary mapping action names to lists of undecided state transitions.
        :param goal_reached: whether the goal was reached in this episode.
        :param num_steps_in_episode: the number of steps taken in this episode.
        :param has_solved_solver_problem: whether the solver successfully solved the problem in this episode.
        """
        self.summarized_info.loc[len(self.summarized_info)] = {
            **{"episode_number": self._episode_number},
            **self._episode_info,
            **{
                f"num_unknown_failed_transitions_{action_name}": len(undecided_states.get(action_name, []))
                for action_name in self._action_names
            },
            **{"goal_reached": goal_reached},
            "num_steps_in_episode": num_steps_in_episode,
            "solver_solved_problem": has_solved_solver_problem,
        }

        self._episode_info = {record_name: 0 for record_name in self._episode_info}
        self._episode_number += 1
        self.export_episode_trajectory()

    def export_statistics(self, path: Path) -> None:
        """Exports the statistics of the episode to a CSV file.

        :param path: the path of the CSV file to export the statistics to.
        """
        self.summarized_info.to_csv(path, index=False)

    def export_episode_trajectory(self, test_mode: bool = False) -> Optional[str]:
        """Exports the trajectory of the episode to a CSV file.

        :param test_mode: whether the function is called in test mode. If True, the trajectory will not be saved to a file.
        """
        if len(self.trajectory) == 0:
            return None

        trajectory_path = Path(f"trajectory_episode_{self._episode_number}.trajectory")
        trajectory_str = "("
        for index, component in enumerate(self.trajectory.components):
            if index == 0:
                trajectory_str += f"{component.previous_state.serialize()}"

            trajectory_str += (
                f"(operator: {str(component.grounded_action_call)})\n"
                f"(:transition_status ({'success' if component.is_successful else 'failure'}))\n"
                f"{component.next_state.serialize()}"
            )

        trajectory_str += ")"
        if test_mode:
            return trajectory_str

        with open(self.working_directory / trajectory_path, "wt") as trajectory_file:
            trajectory_file.write(trajectory_str)

        return trajectory_str

    def clear_trajectory(self):
        """Clears the trajectory of the episode."""
        self.trajectory = Observation()

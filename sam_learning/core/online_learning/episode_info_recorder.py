"""class to record the information about each episode of the online learning process."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from pandas import DataFrame
from pddl_plus_parser.models import ActionCall, Observation, State

RECORD_COLUMNS = [
    "problem_name",
    "num_grounded_actions",
    "exploration_time",
    "model_load_time",
    "sum_failed_actions",
    "sum_successful_actions",
    "goal_reached",
    "num_steps_in_episode",
    "solver_solved_problem",
    "safe_model_solution_status",
    "optimistic_model_solution_status",
]

DISCRETE = "discrete"
NUMERIC = "numeric"
UNKNOWN = "unknown"

not_used_for_solving = "irrelevant"
MAX_STEPS_IN_FILE = 5000  # Maximum number of steps to record in a single trajectory file


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
            **{f"num_failed_due_to_discrete_condition_{action_name}": 0 for action_name in action_names},
            **{f"num_failed_due_to_numeric_condition_{action_name}": 0 for action_name in action_names},
            **{f"num_unknown_failed_transitions_{action_name}": 0 for action_name in action_names},
        }
        self.summarized_info = DataFrame(
            columns=[
                "episode_number",
                *RECORD_COLUMNS,
                *{f"num_{action_name}_success" for action_name in action_names},
                *{f"num_{action_name}_fail" for action_name in action_names},
                *{f"num_unknown_failed_transitions_{action_name}" for action_name in action_names},
                *{f"num_failed_due_to_discrete_condition_{action_name}" for action_name in action_names},
                *{f"num_failed_due_to_numeric_condition_{action_name}" for action_name in action_names},
            ]
        )
        self.trajectory = Observation()
        self._action_successful_execution_history = {f"num_{action}_success": 0 for action in action_names}
        self.working_directory = working_directory
        self._trajectory_paths = []
        self._model_loading_time = -1

    @property
    def trajectory_paths(self) -> List[Path]:
        """Returns the path to the trajectory file for the current episode."""
        return self._trajectory_paths

    def get_number_successful_action_executions(self, action_name: str) -> int:
        return self._action_successful_execution_history[f"num_{action_name}_success"]

    def add_num_grounded_actions(self, num_grounded_actions: int) -> None:
        """Adds the number of grounded actions in the episode.

        :param num_grounded_actions: the number of grounded actions in the episode.
        """
        self._episode_info["num_grounded_actions"] = num_grounded_actions

    def add_model_loading_time(self, model_loading_time: float) -> None:
        """Adds the time taken to load the model in the episode.

        :param model_loading_time: the time taken to load the model in seconds.
        """
        self._model_loading_time = model_loading_time

    def record_failure_reason(self, action_name: str, failure_reason: str) -> None:
        """Records the reason for a failure of an action in the episode.

        :param action_name: the name of the action that failed.
        :param failure_reason: the reason for the failure of the action.
        """
        if failure_reason == DISCRETE:
            self._episode_info[f"num_failed_due_to_discrete_condition_{action_name}"] += 1

        elif failure_reason == NUMERIC:
            self._episode_info[f"num_failed_due_to_numeric_condition_{action_name}"] += 1

        elif failure_reason == UNKNOWN:
            self._episode_info[f"num_unknown_failed_transitions_{action_name}"] += 1

        else:
            raise ValueError(f"Unknown failure reason: {failure_reason}")

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
            self._action_successful_execution_history[f"num_{action.name}_success"] += 1
            return

        self._episode_info[f"num_{action.name}_fail"] += 1
        self._episode_info["sum_failed_actions"] += 1

    def end_episode(
        self,
        problem_name: str,
        goal_reached: bool,
        has_solved_solver_problem: bool = False,
        safe_model_solution_stat: str = not_used_for_solving,
        optimistic_model_solution_stat: str = not_used_for_solving,
        exploration_time: float = 0.0,
    ) -> None:
        """Ends the episode by recording summary statistics, updating the episode counter,
        and exporting the episode trajectory. This method should be called at the end of each episode
        to ensure all relevant information is saved and the recorder is reset for the next episode.

        :param problem_name: the name of the problem that was solved in this episode.
        :param goal_reached: whether the goal was reached in this episode.
        :param has_solved_solver_problem: whether the solver successfully solved the problem in this episode.
        :param safe_model_solution_stat: the solution status of the safe model in this episode.
        :param optimistic_model_solution_stat: the solution status of the optimistic model in this episode.
        """
        self.summarized_info.loc[len(self.summarized_info)] = {
            **{"episode_number": self._episode_number},
            **self._episode_info,
            **{"goal_reached": goal_reached},
            "problem_name": problem_name,
            "num_steps_in_episode": self._episode_info["sum_successful_actions"] + self._episode_info["sum_failed_actions"],
            "solver_solved_problem": has_solved_solver_problem,
            "safe_model_solution_status": safe_model_solution_stat,
            "optimistic_model_solution_status": optimistic_model_solution_stat,
            "exploration_time": exploration_time,
            "model_load_time": self._model_loading_time,
        }

        self._episode_info = {record_name: 0 for record_name in self._episode_info}
        self._episode_number += 1
        self._model_loading_time = -1
        self.export_episode_trajectory(problem_name)

    def export_statistics(self, path: Path) -> None:
        """Exports the statistics of the episode to a CSV file.

        :param path: the path of the CSV file to export the statistics to.
        """
        self.summarized_info.to_csv(path, index=False)

    def export_episode_trajectory(self, problem_name: str, test_mode: bool = False) -> Optional[str]:
        """Exports the trajectory of the episode to a CSV file.

        :param problem_name: the name of the problem that was solved in this episode.
        :param test_mode: whether the function is called in test mode. If True, the trajectory will not be saved to a file.
        """
        if len(self.trajectory) == 0:
            return None

        # splitting the observation to multiple files and writing them separately
        for i in range(0, len(self.trajectory.components), MAX_STEPS_IN_FILE):
            trajectory_slice = self.trajectory.components[i : i + MAX_STEPS_IN_FILE]
            if not trajectory_slice:
                continue

            trajectory_str = "("
            for index, component in enumerate(trajectory_slice):
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

            trajectory_path = Path(
                self.working_directory
                / f"trajectory_{problem_name}_episode_{self._episode_number}_part_{i // MAX_STEPS_IN_FILE}.trajectory"
            )
            self._trajectory_paths.append(trajectory_path)
            with open(trajectory_path, "wt") as trajectory_file:
                trajectory_file.write(trajectory_str)

        return None

    def clear_trajectory(self):
        """Clears the trajectory of the episode."""
        self.trajectory = Observation()

    def clear_paths(self):
        """Clears the paths of the trajectory files."""
        self._trajectory_paths = []

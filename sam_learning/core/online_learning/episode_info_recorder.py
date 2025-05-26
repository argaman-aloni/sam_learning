"""class to record the information about each episode of the online learning process."""

from pathlib import Path
from typing import List, Tuple, Dict

from pandas import DataFrame

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

    def __init__(self, action_names: List[str]):
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

    def add_num_grounded_actions(self, num_grounded_actions: int) -> None:
        """Adds the number of grounded actions in the episode.

        :param num_grounded_actions: the number of grounded actions in the episode.
        """
        self._episode_info["num_grounded_actions"] = num_grounded_actions

    def record_single_step(self, action_name: str, action_applicable: bool) -> None:
        """

        :param action_name:
        :param action_applicable:
        :return:
        """
        if action_applicable:
            self._episode_info[f"num_{action_name}_success"] += 1
            self._episode_info["sum_successful_actions"] += 1
            return

        self._episode_info[f"num_{action_name}_fail"] += 1
        self._episode_info["sum_failed_actions"] += 1

    def end_episode(
        self,
        undecided_states: Dict[str, List[Tuple]],
        goal_reached: bool,
        num_steps_in_episode: int,
        has_solved_solver_problem: bool = False,
    ) -> None:
        """Ends the episode."""
        self.summarized_info.loc[len(self.summarized_info)] = {
            **{"episode_number": self._episode_number},
            **self._episode_info,
            **{f"num_unknown_failed_transitions_{action_name}": len(undecided_states[action_name]) for action_name in self._action_names},
            **{"goal_reached": goal_reached},
            "num_steps_in_episode": num_steps_in_episode,
            "solver_solved_problem": has_solved_solver_problem,
        }

        self._episode_info = {record_name: 0 for record_name in self._episode_info}
        self._episode_number += 1

    def export_statistics(self, path: Path) -> None:
        """Exports the statistics of the episode to a CSV file.

        :param path: the path of the CSV file to export the statistics to.
        """
        self.summarized_info.to_csv(path, index=False)

"""class to record the information about each episode of the online learning process."""
from pathlib import Path
from typing import List

from pandas import DataFrame

RECORD_COLUMNS = ["#grounded_actions", "avg_informative_actions", "sum_failed_actions", "sum_successful_actions"]


class EpisodeInfoRecord:
    """Records the information about the execution of each episode of the online learning process."""

    def __init__(self, action_names: List[str]):
        self._episode_number = 0
        self._num_informative_actions_in_step = []
        self._episode_info = {
            **{record_name: 0 for record_name in RECORD_COLUMNS},
            **{f"#{action_name}_success": 0 for action_name in action_names},
            **{f"#{action_name}_fail": 0 for action_name in action_names}
        }
        self.summarized_info = DataFrame(columns=[
            "episode_number", *RECORD_COLUMNS, *{f"#{action_name}_success" for action_name in action_names},
            *{f"#{action_name}_fail" for action_name in action_names}])

    def add_num_grounded_actions(self, num_grounded_actions: int) -> None:
        """Adds the number of grounded actions in the episode.

        :param num_grounded_actions: the number of grounded actions in the episode.
        """
        self._episode_info["#grounded_actions"] = num_grounded_actions

    def add_num_informative_actions_in_step(self, num_informative_actions: int) -> None:
        """Adds the number of informative actions in the episode.

        :param num_informative_actions: the number of informative actions in the episode.
        """
        self._num_informative_actions_in_step.append(num_informative_actions)

    def add_step_data(self, action_name: str, step_result: int) -> None:
        """

        :param action_name:
        :param step_result:
        :return:
        """
        if step_result == 1:
            self._episode_info[f"#{action_name}_success"] += 1
            self._episode_info["sum_successful_actions"] += 1
            return

        self._episode_info[f"#{action_name}_fail"] += 1
        self._episode_info["sum_failed_actions"] += 1

    def end_episode(self) -> None:
        """Ends the episode."""
        self._episode_info["avg_informative_actions"] = sum(self._num_informative_actions_in_step) / \
                                                        len(self._num_informative_actions_in_step)
        self.summarized_info.loc[len(self.summarized_info)] = {
            **{"episode_number": self._episode_number},
            **self._episode_info}

        self._episode_info = {record_name: 0 for record_name in self._episode_info}
        self._num_informative_actions_in_step = []
        self._episode_number += 1

    def export_statistics(self, path: Path) -> None:
        """Exports the statistics of the episode to a CSV file.

        :param path: the path of the CSV file to export the statistics to.
        """
        self.summarized_info.to_csv(path, index=False)

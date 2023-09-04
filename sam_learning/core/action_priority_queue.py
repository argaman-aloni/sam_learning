"""Module to create a smart priority queue for the actions in the domain."""
import random
from collections import defaultdict

from typing import List, Tuple, Dict, Any


class PriorityQueue:
    """Class that represents a priority queue for the grounded actions in the online learning setting."""

    _prioritized_queue: Dict[float, List[Tuple[Any, float]]]

    def __init__(self):
        self._prioritized_queue = defaultdict(list)

    def get_item(self) -> Any:
        """Returns the item with the highest priority.

        Note:
            This action removes the selected item from the priority queue.

        :return: the item with the highest priority. If there are multiple items with the same priority, one of them
            is selected randomly based on the weights set in insert.
        """
        highest_priority = max(list(self._prioritized_queue.keys()))
        highest_priority_queue = self._prioritized_queue[highest_priority]
        if len(highest_priority_queue) == 1:
            selected_action = highest_priority_queue[0][0]
            self._prioritized_queue.pop(highest_priority)
            return selected_action

        action_probabilities = [queue_item[1] for queue_item in highest_priority_queue]
        selected_queue_item = random.choices(highest_priority_queue, weights=action_probabilities, k=1)[0]
        selected_action = selected_queue_item[0]
        highest_priority_queue.pop(highest_priority_queue.index(selected_queue_item))
        return selected_action

    def insert(self, item: Any, priority: float, selection_probability: float) -> None:
        """Adds an action to the priority queue.

        :param item: the action to add to the priority queue.
        :param priority: the priority of the action.
        :param selection_probability: the probability of selecting the action in case of tie-breaking.
        """
        normalized_priority = round(priority, 4)
        self._prioritized_queue[normalized_priority].append((item, selection_probability))

    def __len__(self) -> int:
        """Returns the number of actions in the priority queue.

        :return: the number of actions in the priority queue.
        """
        return sum([len(queue) for queue in self._prioritized_queue.values()])

    def clear(self) -> None:
        """Clears the priority queue."""
        self._prioritized_queue.clear()

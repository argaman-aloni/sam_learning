"""Module representing an abstract agent that can be used in active learning"""

from abc import ABC, abstractmethod

from pddl_plus_parser.models import State, ActionCall, Observation
from typing import Tuple, Set, List, Optional


class AbstractAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_environment_actions(self, state: State) -> Set[ActionCall]:
        """Returns the set of the actions that are legal in the environment in the current state.

        Note:
            This function was added to handle cases where the modeling does not accurately represent the environment.
            In these cases the agent will have to inform the online learning of what actions are actually legal in the
            environment.

        :param state:  the state that the agent is currently in.
        :return: the set of actions that are legal in the environment in the current state.
        """
        pass

    @abstractmethod
    def observe(self, state: State, action: ActionCall) -> Tuple[State, bool, int]:
        """Observes an action being executed on the state and the resulting new state of the environment.

        :param state: the state before the action was executed.
        :param action: the action that was executed.
        :return: the state after the action was executed, whether the execution of the action in the state was
            successful and the reward received.
        """
        pass

    @abstractmethod
    def goal_reached(self, state: State) -> bool:
        """Indicates whether the goal of the problem has been reached (we assume episodic tasks).

        :param state: the state that the agent is currently in.
        :return: whether the goal has been reached.
        """
        pass

    @abstractmethod
    def execute_plan(self, plan: List[ActionCall]) -> Tuple[Observation, bool]:
        """Executes a plan in the environment and returns the trace created from the plan, and whether using the plan's
        actions the goal was reached.

        :param plan: the plan to be executed.
        :return: The trace created from the plan and whether the goal was reached.
        """
        pass

"""Module representing an abstract agent that can be used in active learning"""

from abc import ABC, abstractmethod

from pddl_plus_parser.models import State, ActionCall
from typing import Tuple


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def observe(self, state: State, action: ActionCall) -> Tuple[State, int]:
        pass

    @abstractmethod
    def goal_reached(self, state: State) -> bool:
        pass

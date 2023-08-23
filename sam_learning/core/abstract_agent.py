"""Module representing an abstract agent that can be used in active learning"""

from abc import ABC, abstractmethod

from pddl_plus_parser.models import State, ActionCall


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def observe(self, state: State, action: ActionCall) -> State:
        pass

    @abstractmethod
    def get_reward(self, state: State) -> float:
        pass

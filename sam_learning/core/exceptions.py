"""Module for exceptions that will be used in the algorithms."""
from enum import Enum


class EquationSolutionType(Enum):
    no_solution_found = 1
    not_enough_data = 2
    convex_hull_not_found = 3
    ok = 4


class NotSafeActionError(Exception):
    action_name: str
    reason: str
    solution_type: EquationSolutionType

    def __init__(self, name: str, reason: str, solution_type: EquationSolutionType):
        self.action_name = name
        self.reason = reason
        self.solution_type = solution_type

    def __str__(self):
        return f"The action {self.action_name} is not safe to use! The reason - {self.reason}"

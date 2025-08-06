"""Abstract class for all solvers."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any


class SolutionOutputTypes(Enum):
    ok = 1
    no_solution = 2
    timeout = 3
    not_applicable = 4
    goal_not_achieved = 5
    solver_error = 6
    irrelevant = 7


class AbstractSolver(ABC):

    name: str = "AbstractSolver"

    @abstractmethod
    def __init__(self):
        """Initialize the solver."""
        pass

    @abstractmethod
    def solve_problem(
        self, domain_file_path: Path, problem_file_path: Path, problems_directory_path: Path, solving_timeout: int, **kwargs: Any
    ) -> SolutionOutputTypes:
        """Solve the problem."""
        pass

    @abstractmethod
    def execute_solver(self, *args, **kwargs):
        """Solves a directory of problems."""
        pass

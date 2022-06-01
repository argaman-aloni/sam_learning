"""Module responsible for running the Expressive Numeric Heuristic Planner (ENHSP)."""
import logging
import subprocess
import sys
from pathlib import Path

from jdk4py import JAVA
from typing import Dict

ENHSP_FILE_PATH = "/sise/home/mordocha/numeric_planning/ENHSP/enhsp.jar"
MAX_RUNNING_TIME = 60  # seconds

TIMEOUT_ERROR_CODE = b"Timeout has been reached"
PROBLEM_SOLVED = b"Problem Solved"


class ENHSPSolver:
    """Class designated to use to activate the metric-FF solver on the cluster and parse its result."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write_batch_and_execute_solver(self, problems_directory_path: Path, domain_file_path: Path) -> Dict[str, str]:
        """Solves numeric and PDDL+ problems using the ENHSP algorithm, automatically outputs the solution into a file.

        :param problems_directory_path: the path to the problems directory.
        :param domain_file_path: the path to the domain file.
        :return: dictionary mapping the problem file name to its solution status.
        """
        solving_stats = {}
        self.logger.info("Starting to solve the input problems using ENHSP solver.")
        for problem_file_path in problems_directory_path.glob("pfile*.pddl"):
            self.logger.debug(f"Starting to work on solving problem - {problem_file_path.stem}")
            solution_path = problems_directory_path / f"{problem_file_path.stem}.solution"
            running_options = ["-o", str(domain_file_path.absolute()),
                               "-f", str(problem_file_path.absolute()),
                               "-planner", "sat-hmrphj",
                               "-sp", str(solution_path.absolute())]
            try:
                process = subprocess.run([str(JAVA), "-jar", ENHSP_FILE_PATH, *running_options], capture_output=True,
                                         timeout=MAX_RUNNING_TIME + 1)
                self.logger.info("ENHSP finished its execution!")
                if PROBLEM_SOLVED in process.stdout:
                    self.logger.debug(f"Solver succeeded in solving problem - {problem_file_path.stem}")
                    solving_stats[problem_file_path.stem] = "ok"

                else:
                    self.logger.debug(f"Solver could not solve problem - {problem_file_path.stem}")
                    solving_stats[problem_file_path.stem] = "no_solution"

            except subprocess.TimeoutExpired:
                self.logger.debug(f"Learning algorithm did not finish in time so was killed "
                                  f"while trying to solve - {problem_file_path.stem}")
                solving_stats[problem_file_path.stem] = "timeout"
                continue

        return solving_stats


if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    solver = ENHSPSolver()
    solver.write_batch_and_execute_solver(problems_directory_path=Path(args[1]),
                                          domain_file_path=Path(args[2]))

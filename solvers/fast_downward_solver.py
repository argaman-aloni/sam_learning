"""Module responsible for running the Expressive Numeric Heuristic Planner (ENHSP)."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

FAST_DOWNWARD_DIR_PATH = os.environ["FAST_DOWNWARD_DIR_PATH"]


class FastDownwardSolver:
    """Class designated to use to activate the metric-FF solver on the cluster and parse its result."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _remove_cost_from_file(solution_path: Path) -> None:
        """Removes the line that contains the plan cost from the plan file because the framework does not support it.

        :param solution_path: the path to the solution file.
        """
        with open(solution_path, "r") as solution_file:
            solution_lines = solution_file.readlines()

        with open(solution_path, "w") as solution_file:
            solution_file.writelines(solution_lines[:-1])

    def execute_solver(self, problems_directory_path: Path, domain_file_path: Path) -> Dict[str, str]:
        """Runs the Fast Downward solver on all the problems in the given directory.

        :param problems_directory_path:
        :param domain_file_path:
        :return:
        """
        solving_stats = {}
        os.chdir(FAST_DOWNWARD_DIR_PATH)
        self.logger.info("Starting to solve the input problems using Fast-Downward solver.")
        for problem_file_path in problems_directory_path.glob("pfile*.pddl"):
            self.logger.debug(f"Starting to work on solving problem - {problem_file_path.stem}")
            solution_path = problems_directory_path / f"{problem_file_path.stem}.solution"
            running_options = ["--overall-time-limit", "60s",
                               "--plan-file", str(solution_path.absolute()),
                               "--sas-file", f"{domain_file_path.stem}_output.sas",
                               str(domain_file_path.absolute()),
                               str(problem_file_path.absolute()),
                               "--evaluator", "'hcea=cea()'",
                               "--search", "'lazy_greedy([hcea], preferred=[hcea])'"]
            run_command = f"./fast-downward.py {' '.join(running_options)}"
            try:
                subprocess.check_output(run_command, shell=True)
                self.logger.info(f"Solver succeeded in solving problem - {problem_file_path.stem}")
                solving_stats[problem_file_path.stem] = "ok"
                self._remove_cost_from_file(solution_path)

            except subprocess.CalledProcessError as e:
                if e.returncode == 23:
                    self.logger.warning(f"Fast Downward returned status code 23 - timeout on problem {problem_file_path.stem}.")
                    solving_stats[problem_file_path.stem] = "timeout"
                elif e.returncode == 11 or e.returncode == 12:
                    self.logger.warning(f"Fast Downward returned status code {e.returncode} - plan unsolvable for problem {problem_file_path.stem}.")
                    solving_stats[problem_file_path.stem] = "no_solution"
                else:
                    self.logger.critical(f"Fast Downward returned status code {e.returncode} - unknown error.")
                    solving_stats[problem_file_path.stem] = "no_solution"

        return solving_stats


if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)
    solver = FastDownwardSolver()
    solver.execute_solver(problems_directory_path=Path(args[1]),
                          domain_file_path=Path(args[2]))

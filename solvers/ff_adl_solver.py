import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict

from pddl_plus_parser.exporters import MetricFFParser

FF_DIRECTORY = os.environ["FF_DIRECTORY"]

MAX_RUNNING_TIME = 60  # seconds


class FFADLSolver:
    """Class designated to use to activate the metric-FF solver on the cluster and parse its result."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = MetricFFParser()

    def _run_ff_process(self, run_command: str, solution_path: Path,
                        problem_file_path: Path, solving_stats: Dict[str, str]) -> None:
        """Runs the metric-ff process."""
        self.logger.info(f"FF solver is working on - {problem_file_path.stem}")
        process = subprocess.Popen(run_command, shell=True)
        try:
            process.wait(timeout=MAX_RUNNING_TIME)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"FF solver took more than {MAX_RUNNING_TIME} seconds to finish.")
            os.kill(process.pid, signal.SIGTERM)
            os.system("pkill -f ./ff")
            solution_path.unlink(missing_ok=True)
            solving_stats[problem_file_path.stem] = "timeout"
            return

        if process.returncode is None:
            solving_stats[problem_file_path.stem] = "timeout"
            solution_path.unlink(missing_ok=True)
            return

        if process.returncode != 0:
            self.logger.warning(f"Solver returned status code {process.returncode}.")
            solving_stats[problem_file_path.stem] = "solver_error"
            solution_path.unlink(missing_ok=True)
            return

        self.logger.info("Solver finished its execution!")
        solving_status = self.parser.get_solving_status(solution_path)[0]
        if solving_status == "ok":
            self.logger.info(f"Solver succeeded in solving problem - {problem_file_path.stem}")
            solving_stats[problem_file_path.stem] = solving_status
            self.parser.parse_plan(solution_path, solution_path)

        elif solving_status == "no-solution":
            self.logger.warning(f"Solver could not solve problem - {problem_file_path.stem}")
            solving_stats[problem_file_path.stem] = "no_solution"

    def execute_solver(self, problems_directory_path: Path, domain_file_path: Path) -> Dict[str, str]:
        """Solves numeric and PDDL+ problems using the FF algorithm and outputs the solution into a file.

        :param problems_directory_path: the path to the problems directory.
        :param domain_file_path: the path to the domain file.
        """
        solving_stats = {}
        os.chdir(FF_DIRECTORY)
        self.logger.info("Starting to solve the input problems using FF solver.")
        for problem_file_path in problems_directory_path.glob("pfile*.pddl"):
            self.logger.debug(f"Starting to work on solving problem - {problem_file_path.stem}")
            solution_path = problems_directory_path / f"{problem_file_path.stem}.solution"
            run_command = f"./ff -o {domain_file_path} -f {problem_file_path} -i 114 > {solution_path}"
            self._run_ff_process(run_command, solution_path, problem_file_path, solving_stats)

        return solving_stats


if __name__ == '__main__':
    args = sys.argv
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    solver = FFADLSolver()
    solver.execute_solver(problems_directory_path=Path(args[1]), domain_file_path=Path(args[2]))

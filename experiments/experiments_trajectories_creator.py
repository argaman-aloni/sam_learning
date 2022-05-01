"""Creates the trajectories that will be used in the trajectory"""
import logging
import sys
from pathlib import Path
from typing import NoReturn

from pddl_plus_parser.exporters import MetricFFParser, TrajectoryExporter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser


class ExperimentTrajectoriesCreator:
    domain_file_name: str
    working_directory_path: Path
    logger: logging.Logger

    def __init__(self, domain_file_name: str, working_directory_path: Path):
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.logger = logging.getLogger(__name__)

    def fix_solution_files(self) -> NoReturn:
        """Fixes the format of the solution files."""
        solution_parser = MetricFFParser()
        for solution_file_path in self.working_directory_path.glob("*.solution"):
            self.logger.debug(f"Fixing the solution file - {solution_file_path.stem}")
            solution_parser.parse_plan(solution_file_path, solution_file_path)

    def create_domain_trajectories(self) -> NoReturn:
        """Creates the domain trajectory files."""
        domain_file_path = self.working_directory_path / self.domain_file_name
        domain = DomainParser(domain_file_path).parse_domain()
        trajectory_exporter = TrajectoryExporter(domain=domain)
        for solution_file_path in self.working_directory_path.glob("*.solution"):
            problem_file_path = self.working_directory_path / f"{solution_file_path.stem}.pddl"
            trajectory_file_path = self.working_directory_path / f"{solution_file_path.stem}.trajectory"
            if trajectory_file_path.exists():
                continue

            problem = ProblemParser(problem_path=problem_file_path, domain=domain).parse_problem()
            try:
                triplets = trajectory_exporter.parse_plan(problem, solution_file_path)
                trajectory_exporter.export_to_file(triplets, trajectory_file_path)

            except Exception:
                continue


if __name__ == '__main__':
    trajectory_creator = ExperimentTrajectoriesCreator(sys.argv[1], Path(sys.argv[2]))
    trajectory_creator.fix_solution_files()
    trajectory_creator.create_domain_trajectories()

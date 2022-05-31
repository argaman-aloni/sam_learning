"""Converts regular trajectories to the type of trajectories that PlanMiner algorithm accepts."""
import logging
import sys
from pathlib import Path
from typing import NoReturn

from pddl_plus_parser.exporters import MetricFFParser, TrajectoryExporter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser


class PlanMinerTrajectoriesCreator:
    """Class that transforms our trajectory file data into the type of trajectories that plan miner accepts."""
    domain_file_name: str
    working_directory_path: Path
    logger: logging.Logger

    def __init__(self, domain_file_name: str, working_directory_path: Path):
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.logger = logging.getLogger(__name__)

    def create_plan_miner_trajectories(self) -> NoReturn:
        """Creates the domain trajectory files."""
        domain_file_path = self.working_directory_path / self.domain_file_name
        domain = DomainParser(domain_file_path).parse_domain()
        for trajectory_file_path in self.working_directory_path.glob("*.trajectory"):
            problem_file_path = self.working_directory_path / f"{trajectory_file_path.stem}.pddl"
            trajectory_file_path = self.working_directory_path / f"{trajectory_file_path.stem}.trajectory"
            if trajectory_file_path.exists():
                continue

            problem = ProblemParser(problem_path=problem_file_path, domain=domain).parse_problem()
            try:
                triplets = trajectory_exporter.parse_plan(problem, trajectory_file_path)
                trajectory_exporter.export_to_file(triplets, trajectory_file_path)

            except (ValueError, IndexError):
                continue


if __name__ == '__main__':
    trajectory_creator = PlanMinerTrajectoriesCreator(sys.argv[1], Path(sys.argv[2]))
    trajectory_creator.create_domain_trajectories()

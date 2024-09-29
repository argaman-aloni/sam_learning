"""Creates the trajectories that will be used in the trajectory"""
import logging
import shutil
import sys
from pathlib import Path
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.multi_agent import PlanConverter, MultiAgentTrajectoryExporter


class MATrajectoriesCreator:
    """Class responsible for creating the trajectories that will be used in the experiments."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_domain_trajectories(self, domain_path: Path, problems_directory: Path, agent_names: List[str]) -> None:
        """Creates the domain trajectory files."""
        for solution_path in problems_directory.glob("*.solution"):
            if (problems_directory / f"{solution_path.stem}.trajectory").exists():
                continue

            self.logger.info(f"Creating trajectory for {solution_path.stem}")
            problem_path = problems_directory / f"{solution_path.stem}.pddl"
            single_agent_plan_path = problems_directory / f"{solution_path.stem}.solution"
            combined_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
            combined_problem = ProblemParser(problem_path=problem_path, domain=combined_domain).parse_problem()
            if not single_agent_plan_path.exists():
                shutil.move(solution_path, single_agent_plan_path)

            plan_converter = PlanConverter(ma_domain=combined_domain)
            try:
                plan_sequence = plan_converter.convert_plan(problem=combined_problem, plan_file_path=single_agent_plan_path, agent_names=agent_names)
                plan_converter.export_plan(plan_file_path=solution_path, plan_actions=plan_sequence)
                trajectory_exporter = MultiAgentTrajectoryExporter(combined_domain)
                triplets = trajectory_exporter.parse_plan(problem=combined_problem, plan_path=solution_path)

            except ValueError as e:
                self.logger.error(f"Error while parsing the plan for {solution_path.stem}: {e}")
                continue

            trajectory_exporter.export_to_file(triplets, problems_directory / f"{solution_path.stem}.trajectory")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    trajectory_creator = MATrajectoriesCreator()
    executing_agents = sys.argv[3].replace("[", "").replace("]", "").split(",")
    trajectory_creator.create_domain_trajectories(domain_path=Path(sys.argv[1]), problems_directory=Path(sys.argv[2]), agent_names=executing_agents)

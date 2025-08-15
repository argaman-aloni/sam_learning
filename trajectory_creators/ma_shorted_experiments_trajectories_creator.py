"""Creates the trajectories that will be used in the trajectory"""

import logging
import shutil
import sys
from pathlib import Path
from typing import List

from pddl_plus_parser.exporters import ProblemExporter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain
from pddl_plus_parser.multi_agent import PlanConverter, MultiAgentTrajectoryExporter

from trajectory_creators.ma_common_functions import create_dataset_folders, insert_dummy_predicate_to_goal, insert_dummy_actions_to_plan


class MATrajectoriesCreator:
    """Class responsible for creating the trajectories that will be used in the experiments."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _create_multi_agent_trajectory(
        self,
        problem_path: Path,
        plan_file_path: Path,
        domain: Domain,
        agent_names: List[str],
        trajectories_directory: Path,
        should_insert_dummy_predicate: bool = True,
    ) -> None:
        """Creates a single multi-agent trajectory.

        :param problem_path: the path to the problem file.
        :param plan_file_path: the path to the plan file.
        :param domain: the domain to use for the trajectory creation.
        :param agent_names: the names of the agents interacting with the environment.
        :param trajectories_directory: the directory to save the trajectories to.
        """
        self.logger.info(f"Creating the multi-agent trajectories for {problem_path.stem}")
        problem = ProblemParser(problem_path=problem_path, domain=domain).parse_problem()
        plan_converter = PlanConverter(ma_domain=domain)
        plan_sequence = plan_converter.convert_plan(problem=problem, plan_file_path=plan_file_path, agent_names=agent_names)
        if should_insert_dummy_predicate:
            self.logger.debug("Adding the dummy predicate to increase the level of difficulty of the trajectory")
            predicate_added = insert_dummy_predicate_to_goal(problem=problem)
            insert_dummy_actions_to_plan(plan_sequence=plan_sequence, agent_names=agent_names, dummy_in_goal=predicate_added)

        ProblemExporter().export_problem(problem, trajectories_directory / f"{problem_path.stem}.pddl")

        combined_plan_path = trajectories_directory / plan_file_path.name
        plan_converter.export_plan(plan_file_path=combined_plan_path, plan_actions=plan_sequence)

        trajectory_exporter = MultiAgentTrajectoryExporter(domain)
        triplets = trajectory_exporter.parse_plan(problem=problem, plan_path=combined_plan_path)
        trajectory_exporter.export_to_file(triplets, trajectories_directory / f"{plan_file_path.stem}.trajectory")

    def create_domain_trajectories(
        self, vanilla_domain_path: Path, enhanced_domain_path: Path, working_directory: Path, agent_names: List[str]
    ) -> None:
        """Creates the trajectories from the multi-agent plans.

        :param vanilla_domain_path: the path to the vanilla domain file.
        :param enhanced_domain_path: the path to the enhanced domain file.
        :param working_directory: the directory containing the problems and the plans.
        :param agent_names: the names of the agents interacting with the environment.
        """
        enhanced_problems_dir, vanilla_problems_dir = create_dataset_folders(working_directory)
        vanilla_domain = DomainParser(domain_path=vanilla_domain_path, partial_parsing=False).parse_domain()
        enhanced_domain = DomainParser(domain_path=enhanced_domain_path, partial_parsing=False).parse_domain()

        for solution_path in working_directory.glob("*.solution"):
            if (enhanced_problems_dir / f"{solution_path.stem}.trajectory").exists() and (
                vanilla_problems_dir / f"{solution_path.stem}.trajectory"
            ).exists():
                continue

            self.logger.info(f"Creating the multi-agent trajectories for {solution_path.stem}")
            problem_path = working_directory / f"{solution_path.stem}.pddl"
            single_agent_plan_path = working_directory / f"{solution_path.stem}.solution"
            try:
                # converting the single-agent plan to a multi-agent plan
                self._create_multi_agent_trajectory(
                    problem_path=problem_path,
                    plan_file_path=single_agent_plan_path,
                    domain=enhanced_domain,
                    agent_names=agent_names,
                    trajectories_directory=enhanced_problems_dir,
                    should_insert_dummy_predicate=True,
                )
                self._create_multi_agent_trajectory(
                    problem_path=problem_path,
                    plan_file_path=single_agent_plan_path,
                    domain=vanilla_domain,
                    agent_names=agent_names,
                    trajectories_directory=vanilla_problems_dir,
                    should_insert_dummy_predicate=False,
                )

            except ValueError as e:
                self.logger.error(f"Error while parsing the plan for {solution_path.stem}: {e}")
                continue

        self.logger.info("Trajectories creation finished!")
        shutil.copy(vanilla_domain_path, vanilla_problems_dir)
        shutil.copy(enhanced_domain_path, enhanced_problems_dir)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    trajectory_creator = MATrajectoriesCreator()
    executing_agents = sys.argv[4].replace("[", "").replace("]", "").split(",")
    trajectory_creator.create_domain_trajectories(
        vanilla_domain_path=Path(sys.argv[1]),
        enhanced_domain_path=Path(sys.argv[2]),
        working_directory=Path(sys.argv[3]),
        agent_names=executing_agents,
    )

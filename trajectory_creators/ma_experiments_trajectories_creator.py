"""Creates the trajectories that will be used in the trajectory"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

from pddl_plus_parser.exporters import ProblemExporter
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.multi_agent import (
    MultiAgentDomainsConverter,
    MultiAgentProblemsConverter,
    PlanConverter,
    MultiAgentTrajectoryExporter,
)

from trajectory_creators.ma_common_functions import insert_dummy_predicate_to_goal, insert_dummy_actions_to_plan

REGULAR_DATASET_FOLDER = "regular_dataset"
ENHANCED_DATASET_FOLDER = "enhanced_dataset"


class MAExperimentTrajectoriesCreator:
    """Class responsible for creating the trajectories that will be used in the experiments."""

    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _export_domains(problems_directory: Path, output_folder: Path) -> Tuple[Path, Path]:
        """Exports the domains to the enhanced and the regular dataset folders.

        :param problems_directory: the directory containing the MA problems.
        :param output_folder: the folder containing the regular and enhanced dataset folders.
        :return: paths to the created domains.
        """
        regular_trajectories_folder = output_folder / REGULAR_DATASET_FOLDER
        enhanced_trajectories_folder = output_folder / ENHANCED_DATASET_FOLDER
        selected_problem_folder = list(problems_directory.glob("*"))[0]
        domain_converter = MultiAgentDomainsConverter(working_directory_path=selected_problem_folder)
        regular_domain_path = domain_converter.export_combined_domain(output_folder=regular_trajectories_folder, add_dummy_actions=False)
        enhanced_domain_path = domain_converter.export_combined_domain(output_folder=enhanced_trajectories_folder, add_dummy_actions=True)
        return regular_domain_path, enhanced_domain_path

    def create_domain_trajectories(
        self, problems_directory: Path, plans_directory: Path, output_folder: Path, agent_names: List[str], planner_prefix: str
    ) -> None:
        """Creates the domain trajectory files.

        :param problems_directory: the directory containing the MA problems.
        :param plans_directory: the directory containing the MA plans.
        :param output_folder: the folder that the script will output the files to.
        :param agent_names: the names of the agents interacting with the environment.
        :param planner_prefix: the prefix of the plans to extract the correct plan names.
        """
        # will create directory for the enhanced trajectories as well as the regular ones.
        regular_trajectories_folder = output_folder / REGULAR_DATASET_FOLDER
        enhanced_trajectories_folder = output_folder / ENHANCED_DATASET_FOLDER
        regular_trajectories_folder.mkdir(parents=True, exist_ok=True)
        enhanced_trajectories_folder.mkdir(parents=True, exist_ok=True)
        regular_domain_path, enhanced_domain_path = self._export_domains(problems_directory=problems_directory, output_folder=output_folder)
        for problem_folder in problems_directory.glob("*"):
            self.logger.info(f"Creating trajectories for {problem_folder.stem}")
            problem_converter = MultiAgentProblemsConverter(working_directory_path=problem_folder, problem_file_prefix="problem")
            problem_folder_name = problem_folder.stem
            combined_problem = problem_converter.combine_problems(combined_domain_path=regular_domain_path)
            ProblemExporter().export_problem(combined_problem, regular_trajectories_folder / f"{problem_folder_name}.pddl")
            combined_domain = DomainParser(domain_path=enhanced_domain_path, partial_parsing=False).parse_domain()
            plan_converter = PlanConverter(ma_domain=combined_domain)
            plan_folder_path = plans_directory / f"{planner_prefix}_{problem_folder_name}"
            plan_sequence = plan_converter.convert_plan(
                problem=combined_problem,
                plan_file_path=plan_folder_path / "Plan.txt",
                agent_names=agent_names,
                should_validate_concurrency_constraint=True,
            )
            combined_regular_plan_path = regular_trajectories_folder / f"{problem_folder_name}.solution"
            plan_converter.export_plan(plan_file_path=combined_regular_plan_path, plan_actions=plan_sequence)

            added_dummy_to_goal = insert_dummy_predicate_to_goal(problem=combined_problem)
            ProblemExporter().export_problem(combined_problem, enhanced_trajectories_folder / f"{problem_folder_name}.pddl")
            insert_dummy_actions_to_plan(plan_sequence=plan_sequence, agent_names=agent_names, dummy_in_goal=added_dummy_to_goal)
            combined_enhanced_plan_path = enhanced_trajectories_folder / f"{problem_folder_name}.solution"
            plan_converter.export_plan(plan_file_path=combined_enhanced_plan_path, plan_actions=plan_sequence)

            trajectory_exporter = MultiAgentTrajectoryExporter(combined_domain)
            triplets = trajectory_exporter.parse_plan(problem=combined_problem, plan_path=combined_regular_plan_path)
            trajectory_exporter.export_to_file(triplets, regular_trajectories_folder / f"{problem_folder_name}.trajectory")
            triplets = trajectory_exporter.parse_plan(problem=combined_problem, plan_path=combined_enhanced_plan_path)
            trajectory_exporter.export_to_file(triplets, enhanced_trajectories_folder / f"{problem_folder_name}.trajectory")


if __name__ == "__main__":
    trajectory_creator = MAExperimentTrajectoriesCreator()
    input_agent_names = sys.argv[4].replace("[", "").replace("]", "").split(",")
    trajectory_creator.create_domain_trajectories(
        problems_directory=Path(sys.argv[1]),
        plans_directory=Path(sys.argv[2]),
        output_folder=Path(sys.argv[3]),
        agent_names=input_agent_names,
        planner_prefix=sys.argv[5],
    )

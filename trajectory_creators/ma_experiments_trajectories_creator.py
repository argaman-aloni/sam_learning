"""Creates the trajectories that will be used in the trajectory"""
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import JointActionCall, ActionCall, NOP_ACTION
from pddl_plus_parser.multi_agent import MultiAgentDomainsConverter, MultiAgentProblemsConverter, PlanConverter, \
    MultiAgentTrajectoryExporter

DUMMY_ACTION_CALL = ActionCall(name="dummy-action", grounded_parameters=[])


class MAExperimentTrajectoriesCreator:
    """Class responsible for creating the trajectories that will be used in the experiments."""
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _copy_domain_files(self, problem_folder: Path, output_folder: Path) -> None:
        """Copies the domain files to the output folder.

        :param problem_folder: the folder containing the problems and the domains.
        :param output_folder: the folder to copy the files to once the trajectories are created.
        """
        self.logger.info("Copying domain related files...")
        combined_domain_file_name = "combined_domain.pddl"
        combined_problem_file_name = "combined_problem.pddl"
        problem_folder_name = problem_folder.stem
        domain_file_path = problem_folder / combined_domain_file_name
        combined_plan_path = problem_folder / f"{problem_folder_name}.solution"
        shutil.copy(domain_file_path, output_folder)
        shutil.copy(problem_folder / combined_problem_file_name, output_folder / f"pfile_{problem_folder_name}.pddl")
        shutil.copy(combined_plan_path, output_folder / f"pfile_{problem_folder_name}.solution")
        shutil.copy(problem_folder / f"{problem_folder_name}.trajectory",
                    output_folder / f"pfile_{problem_folder_name}.trajectory")

    @staticmethod
    def insert_dummy_actions_to_plan(plan_sequence: List[JointActionCall]) -> List[JointActionCall]:
        """Inserts the dummy action to the plan sequence.

        :param plan_sequence: the joint action sequence to insert the dummy action to.
        :return: the joint action sequence with the dummy action inserted in multiple places.
        """
        new_plan_sequence = []
        for joint_action in plan_sequence:
            add_dummy_action = bool(random.randint(0, 1))
            if not add_dummy_action:
                continue

            new_action_index = random.randint(0, len(joint_action.actions) - 1)
            while joint_action.actions[new_action_index].name != NOP_ACTION:
                new_action_index = random.randint(0, len(joint_action.actions) - 1)

            joint_action.actions.insert(new_action_index, DUMMY_ACTION_CALL)
            new_plan_sequence.append(joint_action)

        return new_plan_sequence

    def create_domain_trajectories(self, problems_directory: Path, plans_directory: Path, output_folder: Path,
                                   agent_names: List[str], planner_prefix: str) -> None:
        """Creates the domain trajectory files.

        :param problems_directory: the directory containing the MA problems.
        :param plans_directory: the directory containing the MA plans.
        :param output_folder: the folder that the script will output the files to.
        :param agent_names: the names of the agents interacting with the environment.
        :param planner_prefix: the prefix of the plans to extract the correct plan names.
        """
        for problem_folder in problems_directory.glob("*"):
            self.logger.info(f"Creating trajectories for {problem_folder.stem}")
            domain_converter = MultiAgentDomainsConverter(working_directory_path=problem_folder)
            problem_converter = MultiAgentProblemsConverter(working_directory_path=problem_folder,
                                                            problem_file_prefix="problem")

            problem_folder_name = problem_folder.stem
            combined_domain_file_name = "combined_domain.pddl"
            domain_file_path = problem_folder / combined_domain_file_name
            domain_converter.export_combined_domain()
            problem_converter.export_combined_problem(combined_domain_path=domain_file_path)
            combined_domain = DomainParser(domain_path=domain_file_path, partial_parsing=False).parse_domain()
            combined_problem = ProblemParser(problem_path=problem_folder / "combined_problem.pddl",
                                             domain=combined_domain).parse_problem()
            plan_converter = PlanConverter(ma_domain=combined_domain)
            plan_folder_path = plans_directory / f"{planner_prefix}_{problem_folder_name}"
            plan_sequence = plan_converter.convert_plan(problem=combined_problem,
                                                        plan_file_path=plan_folder_path / "Plan.txt",
                                                        agent_names=agent_names,
                                                        should_validate_concurrency_constraint=False)
            self.insert_dummy_actions_to_plan(plan_sequence=plan_sequence)
            combined_plan_path = problem_folder / f"{problem_folder_name}.solution"
            plan_converter.export_plan(plan_file_path=combined_plan_path, plan_actions=plan_sequence)
            trajectory_exporter = MultiAgentTrajectoryExporter(combined_domain)
            triplets = trajectory_exporter.parse_plan(problem=combined_problem, plan_path=combined_plan_path)
            trajectory_exporter.export_to_file(triplets, problem_folder / f"{problem_folder_name}.trajectory")
            self._copy_domain_files(problem_folder, output_folder)


if __name__ == '__main__':
    trajectory_creator = MAExperimentTrajectoriesCreator()
    input_agent_names = sys.argv[4].replace("[", "").replace("]", "").split(",")
    trajectory_creator.create_domain_trajectories(problems_directory=Path(sys.argv[1]),
                                                  plans_directory=Path(sys.argv[2]),
                                                  output_folder=Path(sys.argv[3]),
                                                  agent_names=input_agent_names,
                                                  planner_prefix=sys.argv[5])

"""Converts regular trajectories to the type of trajectories that PlanMiner algorithm accepts."""
import logging
import sys
from pathlib import Path
from typing import NoReturn

from pddl_plus_parser.exporters import MetricFFParser, TrajectoryExporter
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Operator


class PlanMinerTrajectoriesCreator:
    """Class that transforms our trajectory file data into the type of trajectories that plan miner accepts.

    Note:
        This assumes that there are regular trajectories to be converted to the correct format.
    """
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
        pm_trajectories = []
        for trajectory_file_path in self.working_directory_path.glob("*.trajectory"):
            plan_miner_trajectory_file_path = self.working_directory_path / f"{trajectory_file_path.stem}.pts"
            if plan_miner_trajectory_file_path.exists():
                continue

            problem_file_path = self.working_directory_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_file_path, domain).parse_problem()
            observation = TrajectoryParser(domain, problem).parse_trajectory(trajectory_file_path)
            plan_sequence = ["##Tasks##"]
            state_sequence = ["##States##"]
            for index, action_triplet in enumerate(observation.components):
                action = action_triplet.grounded_action_call
                prev_state = action_triplet.previous_state
                next_state = action_triplet.next_state
                op = Operator(action=domain.actions[action.name], domain=domain, grounded_action_call=action.parameters)
                plan_sequence.append(f"[{index}, {index + 1}]: {op.typed_action_call}")
                if index == 0:
                    state_sequence.append(f"[{index}]: {prev_state.typed_serialize()}")

                state_sequence.append(f"[{index + 1}]: {next_state.typed_serialize()}")

            action_trace = "\n".join(plan_sequence)
            state_trace = "\n".join(state_sequence)
            plan_miner_trajectory = f"New plan!!!\n\n" \
                                    f"{action_trace}\n\n" \
                                    f"{state_trace}"

            pm_trajectories.append(plan_miner_trajectory)


if __name__ == '__main__':
    trajectory_creator = PlanMinerTrajectoriesCreator(sys.argv[1], Path(sys.argv[2]))
    trajectory_creator.create_domain_trajectories()

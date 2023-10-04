"""Creates the trajectories that will be used in the trajectory"""
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple

from pddl_plus_parser.exporters import TrajectoryExporter, TrajectoryTriplet
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import PDDLObject, Domain, ActionCall, Problem, State

from sam_learning.core import VocabularyCreator
from utilities import SolverType

MAX_NUM_STEPS_IN_TRAJECTORY = 100


class RandomWalkTrajectoriesCreator:
    """Class that creates trajectories from executing random walks in the domain."""
    domain_file_name: str
    working_directory_path: Path
    vocabulary_creator: VocabularyCreator
    logger: logging.Logger

    def __init__(self, domain_file_name: str, working_directory_path: Path):
        self.domain_file_name = domain_file_name
        self.working_directory_path = working_directory_path
        self.vocabulary_creator = VocabularyCreator()
        self.logger = logging.getLogger(__name__)

    def create_random_trajectory(self, domain: Domain, problem: Problem, grounded_actions: Set[ActionCall],
                                 trajectory_exporter: TrajectoryExporter) -> Tuple[List[TrajectoryTriplet], List[str]]:
        """

        :param domain:
        :param problem:
        :param grounded_actions:
        :param trajectory_exporter:
        :return:
        """
        action_triplets = []
        plan = []
        init_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
        current_state = init_state.copy()
        for _ in range(MAX_NUM_STEPS_IN_TRAJECTORY):
            random_action_name = random.choice([action for action in domain.actions.keys()])
            action = random.choice([action for action in grounded_actions if action.name == random_action_name])
            triplet = trajectory_exporter.create_single_triplet(
                previous_state=current_state, action_call=str(action), problem_objects=problem.objects)
            plan.append(str(action))
            action_triplets.append(triplet)
            current_state = triplet.next_state
            if len(grounded_actions) == 0:
                break

        return action_triplets, plan

    def create_all_grounded_actions(
            self, observed_objects: Dict[str, PDDLObject], domain: Domain) -> Set[ActionCall]:
        """Creates all the grounded actions for the domain given the current possible objects.

        :param observed_objects: the objects that the learner has observed so far.
        :param domain: the domain of which the actions were taken from.
        :return: a set of all the possible grounded actions.
        """
        self.logger.info("Creating all the grounded actions for the domain given the current possible objects.")
        grounded_action_calls = self.vocabulary_creator.create_grounded_actions_vocabulary(
            domain=domain, observed_objects=observed_objects)
        return grounded_action_calls

    def create_domain_trajectories(self, problems_prefix: str = "pfile") -> None:
        """Creates the domain trajectory files."""
        domain_file_path = self.working_directory_path / self.domain_file_name
        domain = DomainParser(domain_file_path).parse_domain()
        trajectory_exporter = TrajectoryExporter(domain=domain)
        for problem_file_path in self.working_directory_path.glob(f"{problems_prefix}*.pddl"):
            self.logger.info(f"Creating the trajectory for the problem - {problem_file_path.stem}")
            problem = ProblemParser(problem_path=problem_file_path, domain=domain).parse_problem()
            grounded_actions = self.create_all_grounded_actions(observed_objects=problem.objects, domain=domain)
            random_walk_triplets, plan = self.create_random_trajectory(
                domain, problem, grounded_actions, trajectory_exporter)
            self.logger.debug("Creating a copy of the problem file with the trajectory as well as a solution_file.")
            with open(
                    self.working_directory_path / f"{problem_file_path.stem}_random_walk.solution", "wt") as plan_file:
                plan_file.write("\n".join(plan))

            shutil.copy(problem_file_path, self.working_directory_path / f"{problem_file_path.stem}_random_walk.pddl")
            trajectory_exporter.export_to_file(
                random_walk_triplets, self.working_directory_path / f"{problem_file_path.stem}_random_walk.trajectory")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)

    trajectory_creator = RandomWalkTrajectoriesCreator(
        domain_file_name=sys.argv[1],
        working_directory_path=Path(sys.argv[2]))
    selected_solver = SolverType.enhsp
    trajectory_creator.create_domain_trajectories()

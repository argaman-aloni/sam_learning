"""Creates the trajectories that will be used in the trajectory"""
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

from pddl_plus_parser.exporters import TrajectoryExporter, TrajectoryTriplet
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import PDDLObject, Domain, ActionCall, Problem, State, GroundedPredicate, PDDLFunction, Operator

from sam_learning.core import VocabularyCreator
from utilities import SolverType

MAX_NUM_STEPS_IN_TRAJECTORY = 100
random.seed(42)


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

    def create_random_plan(
        self, domain: Domain, problem: Problem, grounded_actions: Set[ActionCall], trajectory_exporter: TrajectoryExporter
    ) -> Tuple[List[TrajectoryTriplet], List[str]]:
        """Creates a random trajectory from the input domain and problem by randomly executing actions.

        :param domain: The domain that contain the actions to execute.
        :param problem: the problem containing the initial state and the objects.
        :param grounded_actions: a list of all the possible grounded actions that match the problem.
        :param trajectory_exporter: the exporter that is used to apply the action
        :return:
        """
        self.logger.info(f"Starting to create a random trajectory for the problem - {problem.name}")
        plan = []
        for _ in range(MAX_NUM_STEPS_IN_TRAJECTORY):
            random_action_name = random.choice([action for action in domain.actions.keys()])
            action = random.choice([action for action in grounded_actions if action.name == random_action_name])
            plan.append(str(action))

        action_triplets = trajectory_exporter.parse_plan(problem=problem, action_sequence=plan)
        return action_triplets, plan

    def create_all_grounded_actions(
        self, observed_objects: Dict[str, PDDLObject], domain: Domain, initial_state_fluents: List[str]
    ) -> Set[ActionCall]:
        """Creates all the grounded actions for the domain given the current possible objects.

        :param observed_objects: the objects that the learner has observed so far.
        :param domain: the domain of which the actions were taken from.
        :param initial_state_fluents: the initial state fluents to validate that all random actions do not interact with illegal fluents.
        :return: a set of all the possible grounded actions.
        """
        self.logger.info("Creating all the grounded actions for the domain given the current possible objects.")
        grounded_action_calls = self.vocabulary_creator.create_grounded_actions_vocabulary(domain=domain, observed_objects=observed_objects)
        legal_actions = set()

        for action in grounded_action_calls:
            operator = Operator(
                action=domain.actions[action.name], domain=domain, grounded_action_call=action.parameters, problem_objects=observed_objects
            )
            operator.ground()
            # validating that the preconditions do not contain any numeric fluents that do not exist in the problem.
            grounded_effect_fluents = set()
            for effect in operator.grounded_effects:
                grounded_effect_fluents.update(effect.grounded_numeric_fluents)

            if not (
                operator.grounded_preconditions.grounded_numeric_fluents.issubset(set(initial_state_fluents))
                and grounded_effect_fluents.issubset(set(initial_state_fluents))
            ):
                continue

            legal_actions.add(action)

        return legal_actions

    def create_domain_trajectories(self, problems_prefix: str = "pfile", output_directory_path: Optional[Path] = None) -> None:
        """Creates the domain trajectory files."""
        domain_file_path = self.working_directory_path / self.domain_file_name
        domain = DomainParser(domain_file_path).parse_domain()
        trajectory_exporter = TrajectoryExporter(domain=domain, allow_invalid_actions=False)
        for problem_file_path in self.working_directory_path.glob(f"{problems_prefix}*.pddl"):
            self.logger.info(f"Creating the trajectory for the problem - {problem_file_path.stem}")
            problem = ProblemParser(problem_path=problem_file_path, domain=domain).parse_problem()
            grounded_actions = self.create_all_grounded_actions(
                observed_objects=problem.objects, domain=domain, initial_state_fluents=list(problem.initial_state_fluents.keys())
            )
            random_walk_triplets, plan = self.create_random_plan(domain, problem, grounded_actions, trajectory_exporter)
            self.logger.debug("Creating a copy of the problem file with the trajectory as well as a solution_file.")
            output_dir = output_directory_path or self.working_directory_path
            with open(output_dir / f"{problem_file_path.stem}_random_walk.solution", "wt") as plan_file:
                plan_file.write("\n".join(plan))

            shutil.copy(problem_file_path, output_dir / f"{problem_file_path.stem}_random_walk.pddl")
            trajectory_exporter.export_to_file(random_walk_triplets, output_dir / f"{problem_file_path.stem}_random_walk.trajectory")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    trajectory_creator = RandomWalkTrajectoriesCreator(domain_file_name=sys.argv[1], working_directory_path=Path(sys.argv[2]))
    selected_solver = SolverType.enhsp
    trajectory_creator.create_domain_trajectories()

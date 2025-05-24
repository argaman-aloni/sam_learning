"""The PIL main framework - Compile, Learn and Plan."""
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import State

from sam_learning.core.online_learning_agents.minecraft_agent import MinecraftAgent
from sam_learning.core import LearnerDomain, EpisodeInfoRecord
from sam_learning.learners import NumericOnlineActionModelLearner
from solvers import MetricFFSolver
from utilities import LearningAlgorithmType, SolverType
from validators import OnlineLearningDomainValidator

DEFAULT_SPLIT = 10
DEFAULT_NUMERIC_TOLERANCE = 0.1

MAX_SIZE_MB = 10


class PIL:
    """Class that represents the PIL framework."""

    logger: logging.Logger
    working_directory_path: Path
    domain_file_name: str
    domain_validator: OnlineLearningDomainValidator
    problems_prefix: str

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        solver_type: SolverType,
        learning_algorithm: LearningAlgorithmType,
        problem_prefix: str = "pfile",
        polynomial_degree: int = 0,
    ):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problems_prefix = problem_prefix
        self.domain_validator = OnlineLearningDomainValidator(
            self.working_directory_path,
            learning_algorithm,
            self.working_directory_path / domain_file_name,
            solver_type=solver_type,
            problem_prefix=problem_prefix,
        )
        self._learning_algorithm = learning_algorithm
        self._polynomial_degree = polynomial_degree

    def export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path, file_name: Optional[str] = None) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        :param file_name: the name of the file to export the domain to.
        """
        domain_file_name = file_name or self.domain_file_name
        domain_path = test_set_path / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl())

        return domain_path

    def learn_model_online(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        train_set_dir_path = self.working_directory_path / "train" / f"fold_{fold_num}_{self._learning_algorithm.value}"
        test_set_dir_path = self.working_directory_path / "test" / f"fold_{fold_num}_{self._learning_algorithm.value}"
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        episode_info_recorder = EpisodeInfoRecord(action_names=[action for action in complete_domain.actions])
        online_learner = NumericOnlineActionModelLearner(
            partial_domain=partial_domain, polynomial_degree=self._polynomial_degree, episode_recorder=episode_info_recorder
        )
        num_training_goal_achieved = 0
        for problem_index, problem_path in enumerate(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl")):
            self.logger.info(f"Starting episode number {problem_index + 1}!")
            problem = ProblemParser(problem_path, complete_domain).parse_problem()
            init_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
            agent = MinecraftAgent(domain=complete_domain, problem=problem)
            online_learner.update_agent(agent)
            learned_model, num_steps_in_episode, goal_achieved = online_learner.explore_to_refine_models(init_state)
            self.logger.info(
                f"Finished episode number {problem_index + 1}! " f"The current goal was {'achieved' if goal_achieved else 'not achieved'}."
            )
            num_training_goal_achieved += 1 if goal_achieved else 0
            episode_info_recorder.end_episode()
            if goal_achieved:
                self.logger.info("The agent successfully solved the current task!")

            if (problem_index + 1) % 100 == 0:
                solved_all_test_problems = self.validate_learned_domain(
                    learned_model,
                    test_set_dir_path,
                    episode_number=problem_index + 1,
                    num_steps_in_episode=num_steps_in_episode,
                    fold_num=fold_num,
                    num_goal_achieved=num_training_goal_achieved,
                )
                episode_stats_path = (
                    self.working_directory_path
                    / "results_directory"
                    / (f"fold_{self._learning_algorithm.name}_{fold_num}" f"_episode_{problem_index + 1}_info.csv")
                )
                episode_info_recorder.export_statistics(episode_stats_path)
                num_training_goal_achieved = 0
                if solved_all_test_problems:
                    self.domain_validator.write_statistics(fold_num)
                    self.logger.info(f"All test set problem were solved after {problem_index + 1} episodes!")
                    return

        self.domain_validator.write_statistics(fold_num)
        self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

    def validate_learned_domain(
        self,
        learned_model: LearnerDomain,
        test_set_dir_path: Path,
        episode_number: int,
        num_steps_in_episode: int,
        fold_num: int,
        num_goal_achieved: int,
    ) -> bool:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param episode_number: the number of the current episode.
        :param num_steps_in_episode: the number of steps that were taken in the current episode.
        :param fold_num: the index of the current folder that is currently running.
        :param num_goal_achieved: the number of times the training goal was achieved in the batch.
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        domains_backup_dir_path = self.working_directory_path / "results_directory" / "domains_backup"
        domains_backup_dir_path.mkdir(exist_ok=True)
        self.export_learned_domain(
            learned_model,
            domains_backup_dir_path,
            f"online_nsam_{self._learning_algorithm.name}_fold_{fold_num}_" f"{learned_model.name}_episode_{episode_number}.pddl",
        )
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        all_possible_solution_types = [solution_type.name for solution_type in SolutionOutputTypes]

        self.domain_validator.solver = MetricFFSolver()
        self.domain_validator._solver_name = "metric_ff"
        self.domain_validator.validate_domain(
            tested_domain_file_path=domain_file_path,
            test_set_directory_path=test_set_dir_path,
            episode_number=episode_number,
            num_steps=num_steps_in_episode,
            num_training_goal_achieved=num_goal_achieved,
            tolerance=DEFAULT_NUMERIC_TOLERANCE,
        )
        metric_ff_solved_problems = sum([self.domain_validator.solving_stats[-1][problem_type] for problem_type in all_possible_solution_types])
        metric_ff_ok_problems = self.domain_validator.solving_stats[-1][SolutionOutputTypes.ok.name]

        if metric_ff_solved_problems == metric_ff_ok_problems:
            self.logger.info("All the test set problems were solved using the learned domain!")
            return True

        return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the PIL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--solver_type",
        required=False,
        type=int,
        choices=[1, 2, 3],
        help="The solver that should be used for the sake of validation.\n FD - 1, Metric-FF - 2, ENHSP - 3.",
        default=3,
    )
    parser.add_argument(
        "--learning_algorithm",
        required=True,
        type=int,
        choices=[3, 4, 6, 14],
        help="The type of learning algorithm. " "\n3: numeric_sam\n4: raw_numeric_sam\n 6: polynomial_sam\n ",
    )
    parser.add_argument("--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' " "fluents", default=None)
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)

    args = parser.parse_args()
    return args


def configure_logger(args: argparse.Namespace):
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    working_directory_path = Path(args.working_directory_path)
    logs_directory_path = working_directory_path / "logs"
    logs_directory_path.mkdir(exist_ok=True)
    # Create a rotating file handler
    max_bytes = MAX_SIZE_MB * 1024 * 1024  # Convert megabytes to bytes
    file_handler = RotatingFileHandler(
        logs_directory_path / f"log_{args.domain_file_name}_fold_{learning_algorithm.name}_{args.fold_number}", maxBytes=max_bytes, backupCount=1
    )

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[file_handler])


def main():
    args = parse_arguments()
    configure_logger(args)
    learner = PIL(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        solver_type=SolverType(args.solver_type),
        learning_algorithm=LearningAlgorithmType(args.learning_algorithm),
        fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
        problem_prefix=args.problems_prefix,
    )
    learner.learn_model_online(fold_num=args.fold_number)


if __name__ == "__main__":
    main()

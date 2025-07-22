"""The PIL main framework - Compile, Learn and Plan."""

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import State

from sam_learning.core import EpisodeInfoRecord
from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners import NumericOnlineActionModelLearner, InformativeExplorer, GoalOrientedExplorer, OptimisticExplorer
from sam_learning.learners.noam_algorithm import InformativeSVM
from sam_learning.learners.semi_online_learning_algorithm import SemiOnlineNumericAMLearner
from solvers import ENHSPSolver, MetricFFSolver
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, NegativePreconditionPolicy, SolverType
from validators import DomainValidator

MAX_SIZE_MB = 5


LEARNING_ALGORITHMS = {
    LearningAlgorithmType.noam_learning: NumericOnlineActionModelLearner,
    LearningAlgorithmType.semi_online: SemiOnlineNumericAMLearner,
    LearningAlgorithmType.informative_explorer: InformativeExplorer,
    LearningAlgorithmType.goal_oriented_explorer: GoalOrientedExplorer,
    LearningAlgorithmType.optimistic_explorer: OptimisticExplorer,
    LearningAlgorithmType.informative_svm: InformativeSVM,
}


class PIL:
    """Class that represents the PIL framework."""

    logger: logging.Logger
    working_directory_path: Path
    domain_file_name: str
    problems_prefix: str

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        problem_prefix: str = "pfile",
        polynomial_degree: int = 0,
        exploration_type: LearningAlgorithmType = LearningAlgorithmType.semi_online,
    ):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problems_prefix = problem_prefix
        self._polynomial_degree = polynomial_degree
        self._exploration_type = exploration_type
        self._agent = None
        self._learning_algorithm = exploration_type
        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path,
            exploration_type,
            self.working_directory_path / domain_file_name,
            problem_prefix=problem_prefix,
        )

    def _init_semantic_performance_calculator(self, fold_num: int) -> None:
        """Initializes the algorithm of the semantic precision - recall calculator."""
        self.semantic_performance_calc = init_semantic_performance_calculator(
            working_directory_path=self.working_directory_path,
            domain_file_name=self.domain_file_name,
            learning_algorithm=self._learning_algorithm,
            test_set_dir_path=self.working_directory_path / "performance_evaluation_trajectories" / f"fold_{fold_num}",
            problem_prefix=self.problems_prefix,
        )

    def validate_online_model(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self._init_semantic_performance_calculator(fold_num)
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        train_set_dir_path = self.working_directory_path / "train" / f"fold_{fold_num}_{self._learning_algorithm.value}"
        test_set_dir_path = self.working_directory_path / "test" / f"fold_{fold_num}_{self._learning_algorithm.value}"
        partial_domain_path = train_set_dir_path / self.domain_file_name
        num_training_episodes = len(list(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl")))
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        safe_domain_path = train_set_dir_path / f"{partial_domain.name}_safe_learned_domain.pddl"
        if not safe_domain_path.exists():
            self.logger.error(f"The safe domain file {safe_domain_path} does not exist. Cannot validate performance.")
            return

        self.logger.info("Validating the model performance of the safe model.")
        self.semantic_performance_calc.calculate_performance(
            safe_domain_path, num_training_episodes, policy=NegativePreconditionPolicy.no_remove
        )
        self.domain_validator.validate_domain(
            tested_domain_file_path=safe_domain_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=list(train_set_dir_path.glob(f"{self.problems_prefix}*.trajectory")),
            tolerance=0.1,
            timeout=300,
            learning_time=0,
            solvers_portfolio=[SolverType.metric_ff, SolverType.enhsp],
            preconditions_removal_policy=NegativePreconditionPolicy.no_remove,
        )
        optimistic_domain_path = train_set_dir_path / f"{partial_domain.name}_optimistic_learned_domain.pddl"
        if not optimistic_domain_path.exists():
            self.logger.error(f"The optimistic domain file {optimistic_domain_path} does not exist. Cannot validate performance.")
            return

        self.logger.info("Validating the model performance of the safe model.")
        self.semantic_performance_calc.calculate_performance(
            optimistic_domain_path, num_training_episodes, policy=NegativePreconditionPolicy.hard
        )

        for solution_file_path in test_set_dir_path.glob("*.solution"):
            solution_file_path.unlink()

        self.domain_validator.validate_domain(
            tested_domain_file_path=optimistic_domain_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=list(train_set_dir_path.glob(f"{self.problems_prefix}*.trajectory")),
            tolerance=0.1,
            timeout=300,
            learning_time=0,
            solvers_portfolio=[SolverType.metric_ff, SolverType.enhsp],
            preconditions_removal_policy=NegativePreconditionPolicy.no_remove,
        )

        self.semantic_performance_calc.export_semantic_performance(fold_num, num_training_episodes)
        self.domain_validator.write_statistics(fold_num, num_training_episodes)

    def learn_model_online(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        train_set_dir_path = self.working_directory_path / "train" / f"fold_{fold_num}_{self._exploration_type.value}"
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        self._agent = IPCAgent(complete_domain)
        episode_recorder = EpisodeInfoRecord(
            action_names=list(partial_domain.actions),
            working_directory=train_set_dir_path,
            fold_number=fold_num,
            algorithm_type=self._learning_algorithm,
        )
        online_learner = LEARNING_ALGORITHMS[self._learning_algorithm](
            workdir=train_set_dir_path,
            partial_domain=partial_domain,
            polynomial_degree=self._polynomial_degree,
            solvers=[MetricFFSolver(), ENHSPSolver()],
            exploration_type=self._exploration_type,
            agent=self._agent,
            episode_recorder=episode_recorder,
        )

        online_learner.initialize_learning_algorithms()
        if self._learning_algorithm == LearningAlgorithmType.semi_online:
            problems_to_solve = sorted(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl"))
            online_learner.try_to_solve_problems(problems_to_solve)

        else:
            num_training_goal_achieved = 0
            for problem_index, problem_path in enumerate(sorted(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl"))):
                self.logger.info(f"Starting episode number {problem_index + 1}!")
                problem = ProblemParser(problem_path, complete_domain).parse_problem()
                self._agent.initialize_problem(problem)
                initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
                num_grounded_actions = len(self._agent.get_environment_actions(initial_state))
                episode_recorder.add_num_grounded_actions(num_grounded_actions)
                goal_achieved, num_steps_in_episode = online_learner.try_to_solve_problem(problem_path)
                episode_recorder.clear_trajectory()
                episode_recorder.export_statistics(train_set_dir_path / f"{self._learning_algorithm.name}_exploration_statistics.csv")
                self.logger.info(
                    f"Finished episode number {problem_index + 1}! "
                    f"The current goal was {'achieved' if goal_achieved else 'not achieved'}."
                )
                num_training_goal_achieved += 1 if goal_achieved else 0
                if goal_achieved:
                    self.logger.info("The agent successfully solved the current task!")

        self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")
        episode_recorder.export_statistics(
            self.working_directory_path
            / "results_directory"
            / f"{self._learning_algorithm.name}_exploration_statistics_fold_{fold_num}.csv"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the PIL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument(
        "--learning_algorithm",
        required=False,
        help="The type of learning algorithm to use for the numeric action model learning.",
        type=int,
        choices=[20, 14, 17, 18, 21, 22],
        default=20,
    )
    parser.add_argument("--debug", required=False, help="Whether in debug mode.", type=bool, default=False)
    args = parser.parse_args()
    return args


def configure_logger(args: argparse.Namespace):
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    local_logs_parent_path = os.environ.get("LOCAL_LOGS_PATH", args.working_directory_path)
    working_directory_path = Path(local_logs_parent_path)
    logs_directory_path = working_directory_path / "logs"
    try:
        logs_directory_path.mkdir(exist_ok=True)
    except PermissionError:
        # This is a hack to not fail and just avoid logging in case the directory cannot be created
        return

    # Create a rotating file handler
    max_bytes = MAX_SIZE_MB * 1024 * 1024  # Convert megabytes to bytes
    file_handler = RotatingFileHandler(
        logs_directory_path / f"log_{args.domain_file_name}_fold_{learning_algorithm.name}_{args.fold_number}.log",
        maxBytes=max_bytes,
        backupCount=1,
    )
    stream_handler = logging.StreamHandler()
    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    if args.debug:
        logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG, handlers=[file_handler, stream_handler])

    else:
        logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[file_handler])


def main():
    args = parse_arguments()
    configure_logger(args)
    learner = PIL(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        problem_prefix=args.problems_prefix,
        polynomial_degree=0,  # Assuming linear models for simplicity
        exploration_type=LearningAlgorithmType(args.learning_algorithm),
    )

    learner.learn_model_online(fold_num=args.fold_number)
    learner.validate_online_model(fold_num=args.fold_number)


if __name__ == "__main__":
    main()

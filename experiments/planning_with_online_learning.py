"""The PIL main framework - Compile, Learn and Plan."""

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser

from sam_learning.core import LearnerDomain
from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners import NumericOnlineActionModelLearner
from sam_learning.learners.noam_algorithm import ExplorationAlgorithmType
from solvers import ENHSPSolver
from utilities import LearningAlgorithmType


MAX_SIZE_MB = 10


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
        exploration_type: ExplorationAlgorithmType = ExplorationAlgorithmType.combined,
    ):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problems_prefix = problem_prefix
        self._polynomial_degree = polynomial_degree
        self._exploration_type = exploration_type
        self._agent = None

    def export_learned_domain(
        self, learned_domain: LearnerDomain, test_set_path: Path, file_name: Optional[str] = None
    ) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        :param file_name: the name of the file to export the domain to.
        """
        domain_file_name = file_name or self.domain_file_name
        domain_path = test_set_path / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl(decimal_digits=4))

        return domain_path

    def learn_model_online(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        train_set_dir_path = (
            self.working_directory_path / "train" / f"fold_{fold_num}_{LearningAlgorithmType.noam_learning.value}"
        )
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        self._agent = IPCAgent(complete_domain)
        online_learner = NumericOnlineActionModelLearner(
            workdir=train_set_dir_path,
            partial_domain=partial_domain,
            polynomial_degree=self._polynomial_degree,
            solver=ENHSPSolver(),
            exploration_type=self._exploration_type,
            agent=self._agent,
        )
        online_learner.initialize_learning_algorithms()
        num_training_goal_achieved = 0
        for problem_index, problem_path in enumerate(sorted(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl"))):
            self.logger.info(f"Starting episode number {problem_index + 1}!")
            problem = ProblemParser(problem_path, complete_domain).parse_problem()
            self._agent.initialize_problem(problem)
            goal_achieved, num_steps_in_episode = online_learner.try_to_solve_problem(problem_path)
            self.logger.info(
                f"Finished episode number {problem_index + 1}! "
                f"The current goal was {'achieved' if goal_achieved else 'not achieved'}."
            )
            num_training_goal_achieved += 1 if goal_achieved else 0
            if goal_achieved:
                self.logger.info("The agent successfully solved the current task!")

        self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the PIL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile"
    )
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)

    args = parser.parse_args()
    return args


def configure_logger(args: argparse.Namespace):
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
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
        logs_directory_path
        / f"log_{args.domain_file_name}_fold_{LearningAlgorithmType.noam_learning.name}_{args.fold_number}.log",
        maxBytes=max_bytes,
        backupCount=1,
    )
    stream_handler = logging.StreamHandler()
    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[stream_handler])


def main():
    args = parse_arguments()
    configure_logger(args)
    learner = PIL(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        problem_prefix=args.problems_prefix,
        polynomial_degree=0,  # Assuming linear models for simplicity
        exploration_type=ExplorationAlgorithmType.combined,  # Using combined exploration strategy
    )
    learner.learn_model_online(fold_num=args.fold_number)


if __name__ == "__main__":
    main()

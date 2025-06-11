"""The PIL main framework - Compile, Learn and Plan."""

import argparse
import logging
import shutil
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, State

from sam_learning.core import EpisodeInfoRecord
from sam_learning.core.online_learning_agents import IPCAgent
from sam_learning.learners import NumericOnlineActionModelLearner
from sam_learning.learners.noam_algorithm import ExplorationAlgorithmType
from sam_learning.learners.semi_online_learning_algorithm import SemiOnlineNumericAMLearner
from solvers import ENHSPSolver, MetricFFSolver
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType

MAX_SIZE_MB = 10
MAX_EPISODE_NUM_STEPS = 5000

ONLINE_LEARNING_ALGORITHMS = {
    ExplorationAlgorithmType.combined: LearningAlgorithmType.noam_learning,
    ExplorationAlgorithmType.informative_explorer: LearningAlgorithmType.noam_informative_explorer,
    ExplorationAlgorithmType.goal_oriented: LearningAlgorithmType.noam_goal_oriented_explorer,
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
        exploration_type: ExplorationAlgorithmType = ExplorationAlgorithmType.combined,
    ):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.domain_file_name = domain_file_name
        self.problems_prefix = problem_prefix
        self._polynomial_degree = polynomial_degree
        self._exploration_type = exploration_type
        self._agent = None
        self._learning_algorithm = ONLINE_LEARNING_ALGORITHMS[exploration_type]

    def _init_semantic_performance_calculator(self, fold_num: int) -> None:
        """Initializes the algorithm of the semantic precision - recall calculator."""
        self.semantic_performance_calc = init_semantic_performance_calculator(
            working_directory_path=self.working_directory_path,
            domain_file_name=self.domain_file_name,
            learning_algorithm=self._learning_algorithm,
            test_set_dir_path=self.working_directory_path / "performance_evaluation_trajectories" / f"fold_{fold_num}",
            problem_prefix=self.problems_prefix,
        )

    def _export_domain_and_backup(
        self,
        episode_number: int,
        fold_number: int,
        learned_model: Domain,
        is_safe_model: bool = False,
    ) -> None:
        """Exports the domain to the test set directory and another backup to the results directory.

        :param episode_number: the number of the episode that is currently running.
        :param fold_number: the number of the fold that is currently running.
        :param learned_model: the domain learned by the action model learning algorithm.
        :param is_safe_model: whether the learned model is a safe model or an optimistic one.
        :return: the path of the PDDL domain in the test set directory.
        """
        domains_backup_dir_path = self.working_directory_path / "results_directory" / "domains_backup"
        domains_backup_dir_path.mkdir(exist_ok=True)
        domain_file_name = learned_model.name + f"_{'safe' if is_safe_model else 'optimistic'}_learned_domain.pddl"
        backup_domain_name = (
            f"{self._learning_algorithm.name}_fold_{fold_number}_{learned_model.name}"
            f"_episode_{episode_number}_{'safe' if is_safe_model else 'optimistic'}_model.pddl"
        )
        shutil.copy(self.working_directory_path / domain_file_name, domains_backup_dir_path / backup_domain_name)

    def learn_model_online(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        train_set_dir_path = self.working_directory_path / "train" / f"fold_{fold_num}_{LearningAlgorithmType.noam_learning.value}"
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        self._agent = IPCAgent(complete_domain)
        episode_recorder = EpisodeInfoRecord(action_names=list(partial_domain.actions), working_directory=train_set_dir_path)
        online_learner = NumericOnlineActionModelLearner(
            workdir=train_set_dir_path,
            partial_domain=partial_domain,
            polynomial_degree=self._polynomial_degree,
            solvers=[MetricFFSolver(), ENHSPSolver()],
            exploration_type=self._exploration_type,
            agent=self._agent,
            episode_recorder=episode_recorder,
        )
        online_learner.initialize_learning_algorithms()
        num_training_goal_achieved = 0
        for problem_index, problem_path in enumerate(sorted(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl"))):
            self.logger.info(f"Starting episode number {problem_index + 1}!")
            problem = ProblemParser(problem_path, complete_domain).parse_problem()
            self._agent.initialize_problem(problem)
            initial_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
            num_grounded_actions = len(self._agent.get_environment_actions(initial_state))
            episode_recorder.add_num_grounded_actions(num_grounded_actions)
            goal_achieved, num_steps_in_episode = online_learner.try_to_solve_problem(
                problem_path,
                num_steps_till_episode_end=MAX_EPISODE_NUM_STEPS,
            )
            episode_recorder.clear_trajectory()
            self.logger.info(
                f"Finished episode number {problem_index + 1}! " f"The current goal was {'achieved' if goal_achieved else 'not achieved'}."
            )
            num_training_goal_achieved += 1 if goal_achieved else 0
            self._export_domain_and_backup(problem_index, fold_num, partial_domain, is_safe_model=True)
            self._export_domain_and_backup(problem_index, fold_num, partial_domain, is_safe_model=False)

            if goal_achieved:
                self.logger.info("The agent successfully solved the current task!")

        self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        episode_recorder.export_statistics(
            self.working_directory_path
            / "results_directory"
            / f"{LearningAlgorithmType.noam_learning.name}_episode_info_fold_{fold_num}.csv"
        )

    def learn_model_semi_online(self, fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        # TODO: Change the path to the train set directory of the correct algorithm.
        train_set_dir_path = self.working_directory_path / "train" / f"fold_{fold_num}_{LearningAlgorithmType.noam_learning.value}"
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        self._agent = IPCAgent(complete_domain)
        episode_recorder = EpisodeInfoRecord(action_names=list(partial_domain.actions), working_directory=train_set_dir_path)
        online_learner = SemiOnlineNumericAMLearner(
            workdir=train_set_dir_path,
            partial_domain=partial_domain,
            polynomial_degree=self._polynomial_degree,
            solvers=[MetricFFSolver(), ENHSPSolver()],
            agent=self._agent,
            episode_recorder=episode_recorder,
        )
        online_learner.initialize_learning_algorithms()
        problems_to_solve = sorted(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl"))
        online_learner.try_to_solve_problems(problems_to_solve)
        self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")
        episode_recorder.export_statistics(
            self.working_directory_path
            / "results_directory"
            / f"{LearningAlgorithmType.noam_learning.name}_episode_info_fold_{fold_num}.csv"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the PIL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument(
        "--exploration_policy",
        required=False,
        help="The policy of the online learning algorithm being tested",
        type=str,
        choices=["informative_explorer", "goal_oriented", "combined"],
        default="combined",
    )
    args = parser.parse_args()
    return args


def configure_logger():
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
    stream_handler = logging.StreamHandler()
    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[stream_handler])


def main():
    args = parse_arguments()
    configure_logger()
    learner = PIL(
        working_directory_path=Path(args.working_directory_path),
        domain_file_name=args.domain_file_name,
        problem_prefix=args.problems_prefix,
        polynomial_degree=0,  # Assuming linear models for simplicity
        exploration_type=ExplorationAlgorithmType.combined,  # Using combined exploration strategy
    )
    learner.learn_model_semi_online(fold_num=args.fold_number)


if __name__ == "__main__":
    main()

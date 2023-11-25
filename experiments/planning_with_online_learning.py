"""The PIL main framework - Compile, Learn and Plan."""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import State

from experiments.ipc_agent import IPCAgent
from experiments.minecraft_agent import MinecraftAgent
from sam_learning.core import LearnerDomain, EpisodeInfoRecord
from sam_learning.learners import OnlineNSAMLearner
from solvers import MetricFFSolver
from utilities import LearningAlgorithmType, SolverType, SolutionOutputTypes
from utilities.k_fold_split import KFoldSplit
from validators import OnlineLearningDomainValidator

DEFAULT_SPLIT = 10
DEFAULT_NUMERIC_TOLERANCE = 0.1

NO_INSIGHT_NUMERIC_ALGORITHMS = [
    LearningAlgorithmType.raw_numeric_sam.value,
    LearningAlgorithmType.raw_polynomial_nam.value,
]


class PIL:
    """Class that represents the PIL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    domain_validator: OnlineLearningDomainValidator
    problems_prefix: str

    def __init__(
            self, working_directory_path: Path, domain_file_name: str, solver_type: SolverType,
            learning_algorithm: LearningAlgorithmType, problem_prefix: str = "pfile",
            polynomial_degree: int = 0, fluents_map_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path, domain_file_name=domain_file_name,
                                 n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self.problems_prefix = problem_prefix
        self.domain_validator = OnlineLearningDomainValidator(
            self.working_directory_path, LearningAlgorithmType.online_nsam,
            self.working_directory_path / domain_file_name,
            solver_type=solver_type, problem_prefix=problem_prefix)
        self._learning_algorithm = learning_algorithm
        self._polynomial_degree = polynomial_degree
        if learning_algorithm.value in NO_INSIGHT_NUMERIC_ALGORITHMS:
            self._fluents_map = None

        else:
            with open(fluents_map_path, "rt") as json_file:
                self._fluents_map = json.load(json_file)

    def export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path,
                              file_name: Optional[str] = None) -> Path:
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

    def learn_model_online(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        complete_domain = DomainParser(domain_path=partial_domain_path).parse_domain()
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        episode_info_recorder = EpisodeInfoRecord(action_names=[action for action in complete_domain.actions])
        online_learner = OnlineNSAMLearner(partial_domain=partial_domain, fluents_map=self._fluents_map,
                                           polynomial_degree=self._polynomial_degree,
                                           episode_recorder=episode_info_recorder)
        num_training_goal_achieved = 0
        for problem_index, problem_path in enumerate(train_set_dir_path.glob(f"{self.problems_prefix}*.pddl")):
            self.logger.info(f"Starting episode number {problem_index + 1}!")
            problem = ProblemParser(problem_path, complete_domain).parse_problem()
            init_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
            agent = MinecraftAgent(domain=complete_domain, problem=problem)
            online_learner.update_agent(agent)
            learned_model, num_steps_in_episode, goal_achieved = online_learner.search_to_learn_action_model(init_state)
            self.logger.info(f"Finished episode number {problem_index + 1}! "
                             f"The current goal was {'achieved' if goal_achieved else 'not achieved'}.")
            num_training_goal_achieved += 1 if goal_achieved else 0
            episode_info_recorder.end_episode()
            if goal_achieved:
                self.logger.info("The agent successfully solved the current task!")

            if (problem_index + 1) % 100 == 0:
                solved_all_test_problems = self.validate_learned_domain(
                    learned_model, test_set_dir_path, episode_number=problem_index + 1,
                    num_steps_in_episode=num_steps_in_episode, fold_num=fold_num,
                    num_goal_achieved=num_training_goal_achieved)
                episode_stats_path = self.working_directory_path / "results_directory" / \
                                     f"fold_{fold_num}_episode_{problem_index + 1}_info.csv"
                episode_info_recorder.export_statistics(episode_stats_path)
                num_training_goal_achieved = 0
                if solved_all_test_problems:
                    self.domain_validator.write_statistics(fold_num)
                    return

        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(
            self, learned_model: LearnerDomain, test_set_dir_path: Path,
            episode_number: int, num_steps_in_episode: int, fold_num: int, num_goal_achieved: int) -> bool:
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
        self.export_learned_domain(learned_model, self.working_directory_path / "results_directory",
                                   f"online_nsam_{self._learning_algorithm.name}_fold_{fold_num}_{learned_model.name}_episode_{episode_number}.pddl")
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        all_possible_solution_types = [solution_type.name for solution_type in SolutionOutputTypes]

        self.domain_validator.solver = MetricFFSolver()
        self.domain_validator._solver_name = "metric_ff"
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path,
                                              episode_number=episode_number,
                                              num_steps=num_steps_in_episode,
                                              num_training_goal_achieved=num_goal_achieved,
                                              tolerance=DEFAULT_NUMERIC_TOLERANCE)
        metric_ff_solved_problems = sum([self.domain_validator.solving_stats[-1][problem_type]
                                         for problem_type in all_possible_solution_types])
        metric_ff_ok_problems = self.domain_validator.solving_stats[-1][SolutionOutputTypes.ok.name]

        if metric_ff_solved_problems == metric_ff_ok_problems:
            self.logger.info("All the test set problems were solved using the learned domain!")
            return True

        return False

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_online(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the PIL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--solver_type", required=False, type=int, choices=[1, 2, 3],
                        help="The solver that should be used for the sake of validation.\n FD - 1, Metric-FF - 2, ENHSP - 3.",
                        default=3)
    parser.add_argument("--learning_algorithm", required=True, type=int, choices=[3, 4, 6, 14],
                        help="The type of learning algorithm. "
                             "\n3: numeric_sam\n4: raw_numeric_sam\n 6: polynomial_sam\n ")
    parser.add_argument("--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' "
                                                                   "fluents", default=None)
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names",
                        type=str, default="pfile")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    offline_learner = PIL(working_directory_path=Path(args.working_directory_path),
                          domain_file_name=args.domain_file_name,
                          solver_type=SolverType(args.solver_type),
                          learning_algorithm=LearningAlgorithmType(args.learning_algorithm),
                          fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
                          problem_prefix=args.problems_prefix)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

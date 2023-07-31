"""The PIL main framework - Compile, Learn and Plan."""
import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Generator

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, ActionCall, PDDLObject, Domain, Operator, State

from experiments.k_fold_split import KFoldSplit
from experiments.learning_statistics_manager import LearningStatisticsManager
from experiments.numeric_performance_calculator import NumericPerformanceCalculator
from experiments.semantic_performance_calculator import SemanticPerformanceCalculator
from experiments.utils import init_semantic_performance_calculator
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner, NumericSAMLearner, PolynomialSAMLearning, ConditionalSAM, \
    UniversallyConditionalSAM, OnlineNSAMLearner
from utilities import LearningAlgorithmType, SolverType, SolutionOutputTypes
from validators import DomainValidator

DEFAULT_SPLIT = 5


class PIL:
    """Class that represents the PIL framework."""
    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    domain_validator: DomainValidator

    def __init__(self, working_directory_path: Path, domain_file_name: str, solver_type: SolverType):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path, domain_file_name=domain_file_name,
                                 n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path, domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=LearningAlgorithmType.online_nsam)

        self.domain_validator = DomainValidator(
            self.working_directory_path, LearningAlgorithmType.online_nsam,
            self.working_directory_path / domain_file_name,
            solver_type=solver_type)

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

    def calculate_information_gain_and_execute(
            self, current_state: State, online_learner: OnlineNSAMLearner,
            possible_grounded_actions: List[Operator]) -> Generator[LearnerDomain, None, None]:
        """

        :param current_state:
        :param online_learner:
        :param possible_grounded_actions:
        :return:
        """
        tried_actions_in_state = 0
        while tried_actions_in_state < len(possible_grounded_actions):
            op_to_execute: Operator = random.choice(possible_grounded_actions)
            op_action_call = ActionCall(op_to_execute.name, op_to_execute.grounded_call_objects)

            while online_learner.calculate_state_information_gain(state=current_state, action=op_action_call) == 0:
                if tried_actions_in_state == len(possible_grounded_actions):
                    return

                tried_actions_in_state += 1
                op_to_execute = random.choice(possible_grounded_actions)
                op_action_call = ActionCall(op_to_execute.name, op_to_execute.grounded_call_objects)

            try:
                next_state = op_to_execute.apply(current_state)
                tried_actions_in_state = 0

            except ValueError:
                self.logger.debug(f"Could not apply the action {op_to_execute.name} to the state.")
                next_state = current_state.copy()
                tried_actions_in_state += 1

            online_learner.execute_action(
                action_to_execute=op_action_call, previous_state=current_state, next_state=next_state)
            yield online_learner.create_safe_model()

        return

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
        online_learner = OnlineNSAMLearner(partial_domain=partial_domain)
        online_learner.init_online_learning()
        for index, problem_path in enumerate(train_set_dir_path.glob("pfile*.pddl")):
            problem = ProblemParser(problem_path, complete_domain).parse_problem()
            observed_objects = problem.objects
            all_ground_actions = self.create_all_grounded_actions(complete_domain, observed_objects)
            init_state = State(predicates=problem.initial_state_predicates, fluents=problem.initial_state_fluents)
            for domain in self.calculate_information_gain_and_execute(init_state, online_learner, all_ground_actions):
                solved_all_test_problems = self.validate_learned_domain(domain, test_set_dir_path)
                if solved_all_test_problems:
                    return

        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(self, learned_model: LearnerDomain,
                                test_set_dir_path: Path) -> bool:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        self.export_learned_domain(learned_model, self.working_directory_path / "results_directory",
                                   f"{learned_model.name}.pddl")
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        self.domain_validator.validate_domain(tested_domain_file_path=domain_file_path,
                                              test_set_directory_path=test_set_dir_path)
        all_possible_solution_types = [solution_type.name for solution_type in SolutionOutputTypes]
        sum_problem_types = sum([self.domain_validator.solving_stats[-1][problem_type]
                                 for problem_type in all_possible_solution_types])
        solved_ok_problems = self.domain_validator.solving_stats[-1][SolutionOutputTypes.ok.name]
        if sum_problem_types == solved_ok_problems:
            self.logger.info("All the test set problems were solved using the learned domain!")
            return True

        return False

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.learn_model_online(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self.domain_validator.write_complete_joint_statistics()

    def create_all_grounded_actions(self, complete_domain: Domain,
                                    observed_objects: Dict[str, PDDLObject]) -> List[Operator]:
        pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the POL algorithm on the input domain.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--solver_type", required=False, type=int, choices=[1, 2, 3],
                        help="The solver that should be used for the sake of validation.\n FD - 1, Metric-FF - 2, ENHSP - 3.",
                        default=3)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    offline_learner = PIL(working_directory_path=Path(args.working_directory_path),
                          domain_file_name=args.domain_file_name,
                          solver_type=SolverType(args.solver_type))
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

"""The POL main framework - Compile, Learn and Plan."""
import argparse
import logging
from pathlib import Path
from typing import List, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import MultiAgentObservation, Observation, Domain

from experiments.multi_agent_experiment_runner import MultiAgentExperimentRunner
from sam_learning.learners import MASAMPlus
from sam_learning.core import LearnerDomain
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, SolverType, NegativePreconditionPolicy, MappingElement
from validators import MacroDomainValidator

DEFAULT_SPLIT = 5
INDEXES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 30, 40, 50, 60, 70, 80]


class MultiAgentPlusExperimentRunner(MultiAgentExperimentRunner):
    """Class that represents the POL framework for multi-agent plus problems."""
    executing_agents: List[str]
    ma_domain_path: Path
    negative_preconditions_policy: NegativePreconditionPolicy
    max_iter: int

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 problem_prefix: str = "pfile", executing_agents: List[str] = None, max_traj: int = 20):
        super().__init__(working_directory_path=working_directory_path, domain_file_name=domain_file_name,
                         learning_algorithm=LearningAlgorithmType.ma_sam_plus,
                         problem_prefix=problem_prefix)
        self.executing_agents = executing_agents
        self.ma_domain_path = None
        self.domain_validator = MacroDomainValidator(
            self.working_directory_path, LearningAlgorithmType.ma_sam_plus,
            self.working_directory_path / domain_file_name, problem_prefix=problem_prefix,
        )
        self.max_iter = max_traj

    def learn_ma_plus_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_ma_plus_observations = []
        observed_objects = {}
        for index, trajectory_file_path in enumerate(train_set_dir_path.glob("*.trajectory")):
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            observed_objects.update(problem.objects)
            complete_observation: MultiAgentObservation = TrajectoryParser(partial_domain, problem).parse_trajectory(
                trajectory_file_path, self.executing_agents)

            allowed_ma_plus_observations.append(complete_observation)
            if index + 1 not in INDEXES:
                continue

            self.logger.info(f"Learning the action model using {len(allowed_ma_plus_observations)} trajectories!")
            for policy in NegativePreconditionPolicy:
                self.negative_preconditions_policy = policy
                self.learn_ma_plus_action_model(allowed_ma_plus_observations, partial_domain, test_set_dir_path, fold_num)

        self.domain_validator.write_statistics(fold_num)

    def validate_learned_macro_domain(
        self, allowed_observations: List[Observation], learned_model: LearnerDomain, test_set_dir_path: Path,
            fold_number: int, learning_time: float, mapping: Dict[str, MappingElement]) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param fold_number: the number of the fold that is currently running.
        :param learning_time: the time it took to learn the domain (in seconds).
        :param mapping: the mapping of the learner+, between macro name and binding.
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        domains_backup_dir_path = self.working_directory_path / "results_directory" / "domains_backup"
        domains_backup_dir_path.mkdir(exist_ok=True)
        self.export_learned_domain(
            learned_model,
            domains_backup_dir_path,
            f"{self._learning_algorithm.name}_fold_{fold_number}_{learned_model.name}_{self.negative_preconditions_policy.name}" f"_{len(allowed_observations)}_trajectories.pddl",
        )

        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        portfolio = (
            [SolverType.fast_forward, SolverType.fast_downward]
        )

        self.domain_validator.validate_domain_macro(
            fold=fold_number,
            policy=self.negative_preconditions_policy,
            tested_domain_file_path=domain_file_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=allowed_observations,
            tolerance=0.1,
            timeout=60,
            learning_time=learning_time,
            solvers_portfolio=portfolio,
            mapping=mapping
        )

        return domain_file_path

    def learn_ma_plus_action_model(
            self, allowed_observations: List[MultiAgentObservation], partial_domain: Domain,
            test_set_dir_path: Path, fold_num: int) -> None:
        """Learns the action model using the ma sam plus algorithm.

        :param allowed_observations: the list of observations that are allowed to be used for learning.
        :param partial_domain: the domain will be learned from the observations.
        :param test_set_dir_path: the path to the test set directory where the learned domain would be validated on.
        :param fold_num: the index of the current fold in the cross validation process.
        """
        learner = MASAMPlus(partial_domain=partial_domain, negative_precondition_policy=self.negative_preconditions_policy)
        self._learning_algorithm = LearningAlgorithmType.ma_sam_plus
        self.domain_validator.learning_algorithm = LearningAlgorithmType.ma_sam_plus
        self.learning_statistics_manager.learning_algorithm = LearningAlgorithmType.ma_sam_plus
        learned_model, learning_report, mapping = learner.learn_combined_action_model_with_macro_actions(allowed_observations)

        self.ma_domain_path = self.generate_path_pattern(LearningAlgorithmType.ma_sam_plus,
                                                         self.negative_preconditions_policy,
                                                         len(allowed_observations),
                                                         fold_num)
        self.export_learned_domain(
            learned_model, self.ma_domain_path.parent,
            self.ma_domain_path.name)
        self.validate_learned_macro_domain(allowed_observations, learned_model, test_set_dir_path,
                                           fold_num, float(learning_report["learning_time"]), learner.mapping)

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold(max_items=self.max_iter)):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.semantic_performance_calc = init_semantic_performance_calculator(
                working_directory_path=self.working_directory_path,
                domain_file_name=self.domain_file_name,
                learning_algorithm=LearningAlgorithmType.ma_sam_plus,
                executing_agents=self.executing_agents,
                test_set_dir_path=test_dir_path)
            self.learn_ma_plus_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self._learning_algorithm = LearningAlgorithmType.ma_sam_plus
        self.domain_validator.write_complete_joint_statistics()
        self.learning_statistics_manager.export_all_folds_action_stats()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the multi-agent model learning experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--executing_agents", required=False, default=None,
                        help="In case of a multi-agent action model learning, the names of the agents that "
                             "are executing the actions")
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names",
                        type=str, default="pfile")
    parser.add_argument("--logs_directory_path", required=False, help="The path to the directory where the logs is")
    parser.add_argument("--num_traj", required=False, help="The max number of trajectories",
                        type=int, default=16)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    executing_agents = args.executing_agents.replace("[", "").replace("]", "").split(",") \
        if args.executing_agents is not None else None

    offline_learner = MultiAgentPlusExperimentRunner(working_directory_path=Path(args.working_directory_path),
                                                     domain_file_name=args.domain_file_name,
                                                     executing_agents=executing_agents,
                                                     problem_prefix=args.problems_prefix,
                                                     max_traj=args.num_traj)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

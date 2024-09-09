"""The POL main framework - Compile, Learn and Plan."""
import argparse
import logging
from pathlib import Path
from typing import List

from logging.handlers import RotatingFileHandler
from experiments.experiments_consts import MAX_SIZE_MB

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import MultiAgentObservation, Observation, Domain

from experiments.basic_experiment_runner import OfflineBasicExperimentRunner
from sam_learning.learners import MultiAgentSAM, SAMLearner
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, SolverType, NegativePreconditionPolicy

DEFAULT_SPLIT = 5

def configure_logger(args: argparse.Namespace):
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
    working_directory_path = Path(args.working_directory_path)
    logs_dir = Path(args.working_directory_path) if not None else working_directory_path
    logs_directory_path = logs_dir / "logs"
    logs_directory_path.mkdir(exist_ok=True)
    # Create a rotating file handler
    max_bytes = MAX_SIZE_MB * 1024 * 1024  # Convert megabytes to bytes
    file_handler = RotatingFileHandler(
        logs_directory_path / f"log_{args.domain_file_name}", maxBytes=max_bytes, backupCount=1,
    )
    stream_handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[file_handler, stream_handler])


class MultiAgentExperimentRunner(OfflineBasicExperimentRunner):
    """Class that represents the POL framework for multi-agent problems."""
    executing_agents: List[str]
    ma_domain_path: Path

    def __init__(self, working_directory_path: Path, domain_file_name: str,
                 problem_prefix: str = "pfile", executing_agents: List[str] = None,
                 negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.normal):
        super().__init__(working_directory_path=working_directory_path, domain_file_name=domain_file_name,
                         learning_algorithm=LearningAlgorithmType.ma_sam,
                         problem_prefix=problem_prefix,
                         negative_precondition_policy=negative_preconditions_policy)
        self.executing_agents = executing_agents
        self.ma_domain_path = None

    def _filter_baseline_single_agent_trajectory(self, complete_observation: MultiAgentObservation) -> Observation:
        """Create a single agent observation from a multi-agent observation.

        :param complete_observation: the multi-agent observation to filter.
        :return: the filtered single agent observation.
        """
        filtered_observation = Observation()
        filtered_observation.add_problem_objects(complete_observation.grounded_objects)
        for component in complete_observation.components:
            if component.grounded_joint_action.action_count > 1:
                self.logger.debug(f"Skipping the joint action - {component.grounded_joint_action} "
                                  f"since it contains multiple agents executing at once.!")
                continue

            filtered_observation.add_component(component.previous_state,
                                               component.grounded_joint_action.operational_actions[0],
                                               component.next_state)
        return filtered_observation

    def learn_ma_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_ma_observations = []
        allowed_sa_observations = []
        observed_objects = {}
        for index, trajectory_file_path in enumerate(train_set_dir_path.glob("*.trajectory")):
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            observed_objects.update(problem.objects)
            complete_observation: MultiAgentObservation = TrajectoryParser(partial_domain, problem).parse_trajectory(
                trajectory_file_path, self.executing_agents)

            filtered_observation = self._filter_baseline_single_agent_trajectory(complete_observation)
            allowed_ma_observations.append(complete_observation)
            allowed_sa_observations.append(filtered_observation)
            self.logger.info(f"Learning the action model using {len(allowed_ma_observations)} trajectories!")
            self.negative_preconditions_policy = NegativePreconditionPolicy.normal
            self.learn_ma_action_model(allowed_ma_observations, partial_domain, test_set_dir_path, fold_num)
            self.learn_baseline_action_model(allowed_sa_observations, partial_domain, test_set_dir_path, fold_num)
            self.negative_preconditions_policy = NegativePreconditionPolicy.soft
            self.learn_ma_action_model(allowed_ma_observations, partial_domain, test_set_dir_path, fold_num)
            self.learn_baseline_action_model(allowed_sa_observations, partial_domain, test_set_dir_path, fold_num)
            self.negative_preconditions_policy = NegativePreconditionPolicy.hard
            self.learn_ma_action_model(allowed_ma_observations, partial_domain, test_set_dir_path, fold_num)
            self.learn_baseline_action_model(allowed_sa_observations, partial_domain, test_set_dir_path, fold_num)

        self.semantic_performance_calc.calculate_performance(self.ma_domain_path, len(allowed_ma_observations))
        self.semantic_performance_calc.export_semantic_performance(fold_num)
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num)
        self.domain_validator.write_statistics(fold_num)

    def validate_learned_domain(
        self, allowed_observations: List[Observation], learned_model, test_set_dir_path: Path, fold_number: int, learning_time: float
    ) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param fold_number: the number of the fold that is currently running.
        :param learning_time: the time it took to learn the domain (in seconds).
        :return: the path for the learned domain.
        """
        domain_file_path = self.export_learned_domain(learned_model, test_set_dir_path)
        domains_backup_dir_path = self.working_directory_path / "results_directory" / "domains_backup"
        domains_backup_dir_path.mkdir(exist_ok=True)
        self.export_learned_domain(
            learned_model,
            domains_backup_dir_path,
            f"{self._learning_algorithm.name}_fold_{fold_number}_{learned_model.name}" f"_{len(allowed_observations)}_trajectories.pddl",
        )

        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        portfolio = (
            [SolverType.fast_forward, SolverType.fast_downward]
        )
        self.domain_validator.validate_domain(
            tested_domain_file_path=domain_file_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=allowed_observations,
            tolerance=0.1,
            timeout=60,
            learning_time=learning_time,
            solvers_portfolio=portfolio,
            policy=self.negative_preconditions_policy
        )

        return domain_file_path

    def learn_baseline_action_model(
            self, allowed_filtered_observations: List[Observation], partial_domain: Domain,
            test_set_dir_path: Path, fold_num: int) -> None:
        """Learns the action model using the baseline algorithm.

        :param allowed_filtered_observations: the list of observations that are allowed to be used for learning.
        :param partial_domain: the domain will be learned from the observations.
        :param test_set_dir_path: the path to the test set directory where the learned domain would be validated on.
        :param fold_num: the index of the current fold in the cross validation process.
        """
        learner = SAMLearner(partial_domain=partial_domain,
                             negative_preconditions_policy=self.negative_preconditions_policy)
        self._learning_algorithm = LearningAlgorithmType.sam_learning
        self.domain_validator.learning_algorithm = LearningAlgorithmType.sam_learning
        self.learning_statistics_manager.learning_algorithm = LearningAlgorithmType.sam_learning
        learned_model, learning_report = learner.learn_action_model(allowed_filtered_observations)
        self.learning_statistics_manager.add_to_action_stats(allowed_filtered_observations, learned_model,
                                                             learning_report)
        self.export_learned_domain(
            learned_model, self.working_directory_path / "results_directory",
            f"ma_baseline_domain_{self.negative_preconditions_policy}_{len(allowed_filtered_observations)}_trajectories_fold_{fold_num}.pddl")
        self.validate_learned_domain(allowed_filtered_observations, learned_model,
                                     test_set_dir_path, fold_num, float(learning_report["learning_time"]))

    def learn_ma_action_model(
            self, allowed_complete_observations: List[MultiAgentObservation],
            partial_domain: Domain, test_set_dir_path: Path, fold_num: int) -> None:
        """Learns the action model using the multi-agent action model learning algorithm.

        :param allowed_complete_observations: the list of observations that are allowed to be used for learning.
        :param partial_domain: the domain will be learned from the observations.
        :param test_set_dir_path: the path to the test set directory where the learned domain would be validated on.
        :param fold_num: the index of the current fold in the cross validation process.
        """
        learner = MultiAgentSAM(partial_domain=partial_domain,
                                negative_precondition_policy=self.negative_preconditions_policy)
        self._learning_algorithm = LearningAlgorithmType.ma_sam
        self.learning_statistics_manager.learning_algorithm = LearningAlgorithmType.ma_sam
        self.domain_validator.learning_algorithm = LearningAlgorithmType.ma_sam
        learned_model, learning_report = learner.learn_combined_action_model(allowed_complete_observations)
        self.learning_statistics_manager.add_to_action_stats(allowed_complete_observations, learned_model,
                                                             learning_report)
        self.ma_domain_path = self.working_directory_path / "results_directory" / \
                              f"ma_sam_domain_{self.negative_preconditions_policy}_{len(allowed_complete_observations)}_trajectories_fold_{fold_num}.pddl"
        self.export_learned_domain(learned_model, self.ma_domain_path.parent, self.ma_domain_path.name)
        self.validate_learned_domain(allowed_complete_observations, learned_model,
                                     test_set_dir_path, fold_num, float(learning_report["learning_time"]))

    def run_cross_validation(self) -> None:
        """Runs that cross validation process on the domain's working directory and validates the results."""
        self.learning_statistics_manager.create_results_directory()
        for fold_num, (train_dir_path, test_dir_path) in enumerate(self.k_fold.create_k_fold()):
            self.logger.info(f"Starting to test the algorithm using cross validation. Fold number {fold_num + 1}")
            self.semantic_performance_calc = init_semantic_performance_calculator(
                working_directory_path=self.working_directory_path,
                domain_file_name=self.domain_file_name,
                learning_algorithm=LearningAlgorithmType.ma_sam,
                executing_agents=self.executing_agents,
                test_set_dir_path=test_dir_path)
            self.learn_ma_model_offline(fold_num, train_dir_path, test_dir_path)
            self.domain_validator.clear_statistics()
            self.learning_statistics_manager.clear_statistics()
            self.logger.info(f"Finished learning the action models for the fold {fold_num + 1}.")

        self._learning_algorithm = LearningAlgorithmType.ma_sam
        self.domain_validator.write_complete_joint_statistics()
        self.semantic_performance_calc.export_combined_semantic_performance()
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
    parser.add_argument("--negative_preconditions_policy", required=False, type=int, choices=[1, 2, 3],
                        help="The negative preconditions policy of the domain",
                        default=1)
    parser.add_argument("--logs_directory_path", required=False, help="The path to the directory where the logs is")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    executing_agents = args.executing_agents.replace("[", "").replace("]", "").split(",") \
        if args.executing_agents is not None else None


    configure_logger(args)

    offline_learner = MultiAgentExperimentRunner(working_directory_path=Path(args.working_directory_path),
                                                 domain_file_name=args.domain_file_name,
                                                 executing_agents=executing_agents,
                                                 problem_prefix=args.problems_prefix,
                                                 negative_preconditions_policy=args.negative_preconditions_policy)
    offline_learner.run_cross_validation()


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)
    main()

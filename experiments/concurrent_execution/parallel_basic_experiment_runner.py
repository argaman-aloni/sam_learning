"""The POL main framework - Compile, Learn and Plan."""
import argparse
import logging
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser
from pddl_plus_parser.models import Observation, Domain, MultiAgentObservation

from experiments.experiments_consts import MAX_SIZE_MB, DEFAULT_SPLIT, DEFAULT_NUMERIC_TOLERANCE, NUMERIC_ALGORITHMS
from sam_learning.core import LearnerDomain
from statistics.learning_statistics_manager import LearningStatisticsManager
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, SolverType, NegativePreconditionPolicy
from utilities.k_fold_split import KFoldSplit
from validators import DomainValidator

PLANNER_EXECUTION_TIMEOUT = 1800  # 30 minutes


def configure_iteration_logger(args: argparse.Namespace):
    """Configures the logger for the numeric action model learning algorithms evaluation experiments."""
    iteration_number = int(args.iteration_number)
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    local_logs_parent_path = os.environ.get("LOCAL_LOGS_PATH", args.working_directory_path)
    working_directory_path = Path(local_logs_parent_path)
    logs_directory_path = working_directory_path / "logs"
    logs_directory_path.mkdir(exist_ok=True)
    # Create a rotating file handler
    max_bytes = MAX_SIZE_MB * 1024 * 1024  # Convert megabytes to bytes
    file_handler = RotatingFileHandler(
        logs_directory_path / f"log_{args.domain_file_name}_fold_{learning_algorithm.name}_{args.fold_number}_{iteration_number}.log",
        maxBytes=max_bytes,
        backupCount=1,
    )
    stream_handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    if args.debug:
        logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[file_handler, stream_handler])
    else:
        logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[file_handler])


class ParallelExperimentRunner:
    """Class that represents the POL framework."""

    logger: logging.Logger
    working_directory_path: Path
    k_fold: KFoldSplit
    domain_file_name: str
    learning_statistics_manager: LearningStatisticsManager
    _learning_algorithm: LearningAlgorithmType
    domain_validator: DomainValidator
    fluents_map: Dict[str, List[str]]

    def __init__(
        self, working_directory_path: Path, domain_file_name: str, learning_algorithm: LearningAlgorithmType, problem_prefix: str = "pfile",
    ):
        self.logger = logging.getLogger(__name__)
        self.working_directory_path = working_directory_path
        self.k_fold = KFoldSplit(working_directory_path=working_directory_path, domain_file_name=domain_file_name, n_split=DEFAULT_SPLIT)
        self.domain_file_name = domain_file_name
        self._learning_algorithm = learning_algorithm
        self.semantic_performance_calc = None
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm,
        )
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name, problem_prefix=problem_prefix,
        )
        self.problem_prefix = problem_prefix

    def _init_semantic_performance_calculator(self, fold_num: int) -> None:
        """Initializes the algorithm of the semantic precision - recall calculator."""
        self.semantic_performance_calc = init_semantic_performance_calculator(
            self.working_directory_path,
            self.domain_file_name,
            self._learning_algorithm,
            test_set_dir_path=self.working_directory_path / "performance_evaluation_trajectories" / f"fold_{fold_num}",
            is_numeric=self._learning_algorithm in NUMERIC_ALGORITHMS,
            problem_prefix=self.problem_prefix,
        )

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, Any]]:
        raise NotImplementedError

    def _export_learned_domain(self, learned_domain: LearnerDomain, test_set_path: Path, file_name: Optional[str] = None) -> Path:
        """Exports the learned domain into a file so that it will be used to solve the test set problems.

        :param learned_domain: the domain that was learned by the action model learning algorithm.
        :param test_set_path: the path to the test set directory where the domain would be exported to.
        :param file_name: the name of the file to export the domain to.
        """
        domain_file_name = file_name or self.domain_file_name
        domain_path = test_set_path / domain_file_name
        with open(domain_path, "wt") as domain_file:
            domain_file.write(learned_domain.to_pddl(should_simplify=False))

        return domain_path

    def _export_domain_and_backup(self, allowed_observations, fold_number, learned_model, negative_preconditions_policy, test_set_dir_path) -> Path:
        """Exports the domain to the test set directory and another backup to the results directory.

        :param allowed_observations: the observations that were used in the learning process.
        :param fold_number: the number of the fold that is currently running.
        :param learned_model: the domain learned by the action model learning algorithm.
        :param negative_preconditions_policy: the policy to use for the negative preconditions.
        :param test_set_dir_path: the path to the test set directory.
        :return: the path of the PDDL domain in the test set directory.
        """
        domain_file_path = self._export_learned_domain(learned_model, test_set_dir_path)
        domains_backup_dir_path = self.working_directory_path / "results_directory" / "domains_backup"
        domains_backup_dir_path.mkdir(exist_ok=True)
        shutil.copy(
            domain_file_path,
            domains_backup_dir_path / f"{self._learning_algorithm.name}_fold_{fold_number}_{learned_model.name}"
            f"_{len(allowed_observations)}_trajectories_{negative_preconditions_policy.name}_policy.pddl",
        )
        return domain_file_path

    def learn_model_offline(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path, iteration_number: int = 0) -> None:
        """Learns the model of the environment by learning from the input trajectories.

        :param fold_num: the index of the current folder that is currently running.
        :param train_set_dir_path: the directory containing the trajectories in which the algorithm is learning from.
        :param test_set_dir_path: the directory containing the test set problems in which the learned model should be
            used to solve.
        :param iteration_number: the number of the iteration that is currently running.
        """
        self.logger.info(f"Starting the learning phase for the fold - {fold_num}!")
        partial_domain_path = train_set_dir_path / self.domain_file_name
        partial_domain = DomainParser(domain_path=partial_domain_path, partial_parsing=True).parse_domain()
        allowed_observations = []
        sorted_trajectory_paths = sorted(train_set_dir_path.glob("*.trajectory"))
        for index, trajectory_file_path in enumerate(sorted_trajectory_paths):
            # assuming that the folders were created so that each folder contains only the correct number of trajectories, i.e., iteration_number
            problem_path = train_set_dir_path / f"{trajectory_file_path.stem}.pddl"
            problem = ProblemParser(problem_path, partial_domain).parse_problem()
            new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
            allowed_observations.append(new_observation)

        self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
        learned_model, learning_report = self._apply_learning_algorithm(partial_domain, allowed_observations, test_set_dir_path)
        self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_model, learning_report)
        learned_domain_path = self.validate_learned_domain(
            allowed_observations, learned_model, test_set_dir_path, fold_num, learning_report["learning_time"]
        )
        self.semantic_performance_calc.calculate_performance(learned_domain_path, len(allowed_observations))
        self.domain_validator.write_statistics(fold_num, iteration_number)
        self.semantic_performance_calc.export_semantic_performance(fold_num, iteration_number)
        self.learning_statistics_manager.export_action_learning_statistics(fold_number=fold_num, iteration_num=iteration_number)

    def validate_learned_domain(
        self,
        allowed_observations: Union[List[Observation], List[MultiAgentObservation]],
        learned_model: LearnerDomain,
        test_set_dir_path: Path,
        fold_number: int,
        learning_time: float,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.no_remove,
    ) -> Path:
        """Validates that using the learned domain both the used and the test set problems can be solved.

        :param allowed_observations: the observations that were used in the learning process.
        :param learned_model: the domain that was learned using POL.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param fold_number: the number of the fold that is currently running.
        :param learning_time: the time it took to learn the domain (in seconds).
        :param negative_preconditions_policy: the policy to use for the negative preconditions.
        :return: the path for the learned domain.
        """
        domain_file_path = self._export_domain_and_backup(
            allowed_observations, fold_number, learned_model, negative_preconditions_policy, test_set_dir_path
        )
        self.logger.debug("Checking that the test set problems can be solved using the learned domain.")
        portfolio = [SolverType.metric_ff, SolverType.enhsp] if self._learning_algorithm in NUMERIC_ALGORITHMS else [SolverType.fast_downward]
        self.domain_validator.validate_domain(
            tested_domain_file_path=domain_file_path,
            test_set_directory_path=test_set_dir_path,
            used_observations=allowed_observations,
            tolerance=DEFAULT_NUMERIC_TOLERANCE,
            timeout=os.environ.get("PLANNER_EXECUTION_TIMEOUT", PLANNER_EXECUTION_TIMEOUT),
            learning_time=learning_time,
            solvers_portfolio=portfolio,
            preconditions_removal_policy=negative_preconditions_policy,
        )

        return domain_file_path

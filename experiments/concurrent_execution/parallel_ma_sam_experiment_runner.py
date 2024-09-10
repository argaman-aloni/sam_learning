"""Runs experiments for the numeric model learning algorithms."""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from pddl_plus_parser.models import Observation, Domain
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser, ProblemParser

from experiments.concurrent_execution.parallel_basic_experiment_runner import (
    ParallelExperimentRunner,
    configure_iteration_logger,
)
from experiments.experiments_consts import MA_SAM_ALGORITHM_VERSIONS, MA_SAM_POLICIES_VERSIONS, SAM_ALGORITHM_VERSIONS
from sam_learning.core import LearnerDomain
from utilities import LearningAlgorithmType
from validators import DomainValidator


class SingleIterationMASAMExperimentRunner(ParallelExperimentRunner):
    """Class to conduct offline numeric action model learning experiments."""
    executing_agents: List[str]

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learning_algorithm: LearningAlgorithmType,
        fluents_map_path: Optional[Path],
        executing_agents: List[str],
        problem_prefix: str = "pfile",
    ):
        super().__init__(
            working_directory_path=working_directory_path,
            domain_file_name=domain_file_name,
            learning_algorithm=learning_algorithm,
            problem_prefix=problem_prefix,
        )
        self.fluents_map = None
        if fluents_map_path is not None:
            with open(fluents_map_path, "rt") as json_file:
                self.fluents_map = json.load(json_file)

        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path,
            learning_algorithm,
            self.working_directory_path / domain_file_name,
            problem_prefix=problem_prefix,
        )
        self.executing_agents = executing_agents

    def _filter_baseline_single_agent_trajectory(self, complete_observations) -> List[Observation]:
        """Create a single agent observation from a multi-agent observation.

        :param complete_observation: the multi-agent observation to filter.
        :return: the filtered single agent observation.
        """
        filtered_observations = []
        for complete_observation in complete_observations:
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
            filtered_observations.append(filtered_observation)

        return filtered_observations

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learns the action model using the numeric action model learning algorithms.

        :param partial_domain: the partial domain without the actions' preconditions and effects.
        :param allowed_observations: the allowed observations.
        :param test_set_dir_path: the path to the directory containing the test problems.
        :return: the learned action model and the learned action model's learning statistics.
        """

        if self._learning_algorithm in MA_SAM_ALGORITHM_VERSIONS:
            learner = MA_SAM_ALGORITHM_VERSIONS[self._learning_algorithm](
                partial_domain=partial_domain, preconditions_fluent_map=self.fluents_map,
                negative_precondition_policy=MA_SAM_POLICIES_VERSIONS[self._learning_algorithm]
            )
            return learner.learn_combined_action_model(allowed_observations)
        elif self._learning_algorithm in SAM_ALGORITHM_VERSIONS:
            filtered_observation = self._filter_baseline_single_agent_trajectory(allowed_observations)
            learner = SAM_ALGORITHM_VERSIONS[self._learning_algorithm](
                partial_domain=partial_domain,
                negative_preconditions_policy=MA_SAM_POLICIES_VERSIONS[self._learning_algorithm]
            )
            return learner.learn_action_model(filtered_observation)

        # TODO add ma sam plus and it should be enough to work for ma sam plus experiments as well

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
            new_observation = (TrajectoryParser(partial_domain, problem)
                               .parse_trajectory(trajectory_file_path, executing_agents=self.executing_agents))
            allowed_observations.append(new_observation)

        self.logger.info(f"Learning the action model using {len(allowed_observations)} trajectories!")
        learned_model, learning_report = self._apply_learning_algorithm(partial_domain, allowed_observations, test_set_dir_path)

        learned_domain_path = self.validate_learned_domain(
            allowed_observations, learned_model, test_set_dir_path, fold_num, learning_report["learning_time"]
        )
        # self.semantic_performance_calc.calculate_performance(learned_domain_path, len(allowed_observations))
        self.domain_validator.write_statistics(fold_num, iteration_number)
        # self.semantic_performance_calc.export_semantic_performance(fold_num, iteration_number)




    def run_fold_iteration(self, fold_num: int, train_set_dir_path: Path, test_set_dir_path: Path, iteration_number: int) -> None:
        """Runs the numeric action model learning algorithms on the input fold.

        :param fold_num: the number of the fold to run.
        :param train_set_dir_path: the path to the directory containing the training set problems.
        :param test_set_dir_path: the path to the directory containing the test set problems.
        :param iteration_number: the current iteration number.
        """
        self.logger.info(f"Running fold {fold_num} iteration {iteration_number}")
        self._init_semantic_performance_calculator(fold_num)
        self.learn_model_offline(fold_num, train_set_dir_path, test_set_dir_path, iteration_number)
        self.domain_validator.clear_statistics()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the numeric action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument("--executing_agents", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--learning_algorithm",
        required=True,
        type=int,
        choices=[1, 7, 31, 32, 33, 34],
    )
    parser.add_argument(
        "--fluents_map_path", required=False, help="The path to the file mapping to the preconditions' " "fluents", default=None,
    )
    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument("--iteration_number", required=True, help="The current iteration to execute", type=int)
    parser.add_argument("--debug", required=False, help="Whether in debug mode.", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    executing_agents = args.executing_agents.replace("[", "").replace("]", "").split(",") \
        if args.executing_agents is not None else None
    # configure_iteration_logger(args)
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    working_directory_path = Path(args.working_directory_path)
    iteration_number = int(args.iteration_number)
    offline_learner = SingleIterationMASAMExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        learning_algorithm=learning_algorithm,
        fluents_map_path=Path(args.fluents_map_path) if args.fluents_map_path else None,
        problem_prefix=args.problems_prefix,
        executing_agents=executing_agents
    )
    offline_learner.run_fold_iteration(
        fold_num=args.fold_number,
        train_set_dir_path=(working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}_{iteration_number}",
        test_set_dir_path=(working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}_{iteration_number}",
        iteration_number=int(args.iteration_number),
    )


if __name__ == "__main__":
    main()

"""The POL main framework - Compile, Learn and Plan."""

from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse

from pddl_plus_parser.models import Observation, Domain

from experiments.concurrent_execution.parallel_basic_experiment_runner import (ParallelExperimentRunner,
                                                                               configure_iteration_logger)
from sam_learning.core import LearnerDomain
from sam_learning.learners import ExtendedSamLearner, SAMLearner
from statistics.learning_statistics_manager import LearningStatisticsManager
from statistics.utils import init_semantic_performance_calculator
from utilities import LearningAlgorithmType, NegativePreconditionPolicy
from validators import DomainValidator

PLANNER_EXECUTION_TIMEOUT = 1800  # 30 minutes
NUM_TRIPLETS_PER_TESTING = 10


class ParallelDiscreteExperimentRunner(ParallelExperimentRunner):
    """Class that represents the POL framework."""

    def __init__(
        self,
        working_directory_path: Path,
        domain_file_name: str,
        learning_algorithm: LearningAlgorithmType,
        problem_prefix: str = "pfile",
        running_triplets_experiment: bool = True,
        executing_agents: List[str] = None,
    ):
        super().__init__(working_directory_path=working_directory_path,
                         domain_file_name=domain_file_name,
                         learning_algorithm=learning_algorithm,
                         problem_prefix=problem_prefix,
                         running_triplets_experiment=running_triplets_experiment,
                         executing_agents=executing_agents)
        self.semantic_performance_calc = None
        self.domain_validator = DomainValidator(
            self.working_directory_path, learning_algorithm, self.working_directory_path / domain_file_name, problem_prefix=problem_prefix,
        )
        self.learning_statistics_manager = LearningStatisticsManager(
            working_directory_path=working_directory_path,
            domain_path=self.working_directory_path / domain_file_name,
            learning_algorithm=learning_algorithm,
        )
        self.learner = None


    def _init_semantic_performance_calculator(self, fold_num: int) -> None:
        """Initializes the algorithm of the semantic precision - recall calculator."""
        self.semantic_performance_calc = init_semantic_performance_calculator(
            working_directory_path=self.working_directory_path,
            domain_file_name=self.domain_file_name,
            learning_algorithm=self._learning_algorithm,
            test_set_dir_path=self.working_directory_path / "performance_evaluation_trajectories" / f"fold_{fold_num}",
            problem_prefix=self.problem_prefix,
            executing_agents=self.executing_agents,
        )

    def _apply_learning_algorithm(
        self, partial_domain: Domain, allowed_observations: List[Observation], test_set_dir_path: Path
    ) -> Tuple[LearnerDomain, Dict[str, Any]]:

        if self._learning_algorithm == LearningAlgorithmType.esam_learning:
            esam_learner = ExtendedSamLearner(partial_domain)
            learner_domain , report = esam_learner.learn_action_model(allowed_observations)

            self.semantic_performance_calc.encode = esam_learner.get_encoder()
            self.semantic_performance_calc.decode = esam_learner.get_decoder()


        elif self._learning_algorithm == LearningAlgorithmType.sam_learning:
            sam_learner = SAMLearner(partial_domain)
            learner_domain , report = sam_learner.learn_action_model(allowed_observations)

            self.semantic_performance_calc.encode = lambda call: [call] if call.name in learner_domain.actions else []
            self.semantic_performance_calc.decode = lambda call: call

        else:
            learner_domain , report = LearnerDomain(partial_domain), {}


        return learner_domain, report


    def _learn_model_offline(
            self, allowed_observations: List[Observation], partial_domain: Domain, test_set_dir_path: Path,
            fold_num: int) -> None:
        """Learns the model of the environment by learning from the input trajectories.
            used to solve.
        """
        policy = NegativePreconditionPolicy.no_remove
        learned_domain, learning_report = self._apply_learning_algorithm(partial_domain, allowed_observations, test_set_dir_path)
        self.learning_statistics_manager.add_to_action_stats(allowed_observations, learned_domain, learning_report, policy=policy)
        learned_domain_path = self.validate_learned_domain(
            allowed_observations, learned_domain, test_set_dir_path, fold_num, float(learning_report["learning_time"]), policy
        )

        self.semantic_performance_calc.calculate_performance(learned_domain_path, sum([len(observation) for observation in allowed_observations]))
        self.logger.info(f"Finished the learning phase for the fold - {fold_num} and {len(allowed_observations)} observations!")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs action model learning algorithms evaluation experiments.")
    parser.add_argument("--working_directory_path", required=True, help="The path to the directory where the domain is")
    parser.add_argument("--domain_file_name", required=True, help="the domain file name including the extension")
    parser.add_argument(
        "--learning_algorithm",
        required=True,
        type=int,
        choices=[1,2],
        help="the type of action model learning algorithm to use.\n SAM- 1, Extended-SAM- 2.",
    )
    parser.add_argument("--iteration_number", required=False, help="The current iteration to execute", type=int)
    parser.add_argument(
        "--solver_type",
        required=False,
        type=int,
        choices=[2],
        help="The solver that should be used for the sake of validation.\nMetric-FF - 2",
        default=2,
    )


    parser.add_argument("--problems_prefix", required=False, help="The prefix of the problems' file names", type=str, default="pfile")
    parser.add_argument("--fold_number", required=True, help="The number of the fold to run", type=int)
    parser.add_argument("--debug", required=False, help="Whether in debug mode.", type=bool, default=False)
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    configure_iteration_logger(args)
    learning_algorithm = LearningAlgorithmType(args.learning_algorithm)
    working_directory_path = Path(args.working_directory_path)
    offline_learner = ParallelDiscreteExperimentRunner(
        working_directory_path=working_directory_path,
        domain_file_name=args.domain_file_name,
        learning_algorithm=learning_algorithm,
        problem_prefix=args.problems_prefix,
    )
    offline_learner.run_action_triplets_experiment(
        fold_num=args.fold_number,
        train_set_dir_path=(
                                       working_directory_path / "train") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
        test_set_dir_path=(
                                      working_directory_path / "test") / f"fold_{args.fold_number}_{args.learning_algorithm}_triplets",
    )

if __name__ == "__main__":
    main()


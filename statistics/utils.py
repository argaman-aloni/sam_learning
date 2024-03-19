"""Utility functionality for the experiments"""
import random
from pathlib import Path
from typing import List, Union
from typing import Optional

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Observation

from statistics.semantic_performance_calculator import SemanticPerformanceCalculator
from statistics.numeric_performance_calculator import NumericPerformanceCalculator
from utilities import LearningAlgorithmType

DEFAULT_SIZE = 10


def init_semantic_performance_calculator(
    working_directory_path: Path,
    domain_file_name: str,
    learning_algorithm: LearningAlgorithmType,
    executing_agents: Optional[List[str]] = None,
    test_set_dir_path: Path = None,
    is_numeric: bool = False,
) -> Union[NumericPerformanceCalculator, SemanticPerformanceCalculator]:
    """Initializes a numeric performance calculator object.

    :param working_directory_path: the directory path where the domain and problem files are located.
    :param domain_file_name: the name of the domain file.
    :param learning_algorithm: the type of learning algorithm to use.
    :param executing_agents: the agents that are executing the domain.
    :param test_set_dir_path: the path to the directory containing the test set.
    :param is_numeric: whether the performance calculator is numeric or not.
    :return: the initialized numeric performance calculator object.
    """
    domain_path = working_directory_path / domain_file_name
    model_domain = partial_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
    observations = []
    problem_files = list(test_set_dir_path.glob("pfile*.pddl"))
    selected_paths = random.sample(problem_files, min(DEFAULT_SIZE, len(problem_files)))
    for test_problem_path in selected_paths:
        trajectory_file_path = working_directory_path / f"{test_problem_path.stem}.trajectory"
        problem = ProblemParser(test_problem_path, partial_domain).parse_problem()
        trajectory = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path, executing_agents=executing_agents)
        sampled_observation = Observation()
        sampled_observation.grounded_objects = trajectory.grounded_objects
        sampled_observation.components = random.sample(trajectory.components, k=min(DEFAULT_SIZE, len(trajectory.components)))
        observations.append(sampled_observation)

    if is_numeric:
        return NumericPerformanceCalculator(
            model_domain=model_domain,
            observations=observations,
            model_domain_path=domain_path,
            working_directory_path=working_directory_path,
            learning_algorithm=learning_algorithm,
        )

    return SemanticPerformanceCalculator(
        model_domain=model_domain,
        model_domain_path=domain_path,
        observations=observations,
        working_directory_path=working_directory_path,
        learning_algorithm=learning_algorithm,
    )

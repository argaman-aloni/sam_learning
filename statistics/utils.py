"""Utility functionality for the experiments"""
from pathlib import Path
from typing import List, Union
from typing import Optional

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser

from statistics.ma_performance_calculator import MASamPerformanceCalculator
from statistics.numeric_performance_calculator import NumericPerformanceCalculator
from statistics.semantic_performance_calculator import SemanticPerformanceCalculator
from utilities import LearningAlgorithmType

DEFAULT_SIZE = 10


def init_semantic_performance_calculator(
    working_directory_path: Path,
    domain_file_name: str,
    learning_algorithm: LearningAlgorithmType,
    executing_agents: Optional[List[str]] = None,
    test_set_dir_path: Path = None,
    is_numeric: bool = False,
    problem_prefix: str = "pfile",
) -> Union[NumericPerformanceCalculator, SemanticPerformanceCalculator]:
    """Initializes a numeric performance calculator object.

    :param working_directory_path: the directory path where the domain and problem files are located.
    :param domain_file_name: the name of the domain file.
    :param learning_algorithm: the type of learning algorithm to use.
    :param executing_agents: the agents that are executing the domain.
    :param test_set_dir_path: the path to the directory containing the test set.
    :param is_numeric: whether the performance calculator is numeric or not.
    :param problem_prefix: the prefix of the problem files.
    :return: the initialized numeric performance calculator object.
    """
    domain_path = working_directory_path / domain_file_name
    model_domain = partial_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
    observations = []
    problem_files = list(test_set_dir_path.glob(f"{problem_prefix}*.pddl"))
    for test_problem_path in problem_files:
        trajectory_file_path = test_set_dir_path / f"{test_problem_path.stem}.trajectory"
        problem = ProblemParser(test_problem_path, partial_domain).parse_problem()
        observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path, executing_agents=executing_agents)
        observations.append(observation)

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


def init_semantic_performance_calculator_for_ma_experiments(
    working_directory_path: Path,
    domain_file_name: str,
    learning_algorithm: LearningAlgorithmType,
    executing_agents: Optional[List[str]] = None,
    test_set_dir_path: Path = None,
) ->MASamPerformanceCalculator:
    """Initializes a numeric performance calculator object.

    :param working_directory_path: the directory path where the domain and problem files are located.
    :param domain_file_name: the name of the domain file.
    :param learning_algorithm: the type of learning algorithm to use.
    :param executing_agents: the agents that are executing the domain.
    :param test_set_dir_path: the path to the directory containing the test set.
    :return: the initialized numeric performance calculator object.
    """
    domain_path = working_directory_path / domain_file_name
    model_domain = partial_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
    observations = []
    problem_files = list(test_set_dir_path.glob("pfile*.pddl"))
    for test_problem_path in problem_files:
        trajectory_file_path = working_directory_path / f"{test_problem_path.stem}.trajectory"
        problem = ProblemParser(test_problem_path, partial_domain).parse_problem()
        observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path, executing_agents=executing_agents)
        observations.append(observation)

    return MASamPerformanceCalculator(
        model_domain=model_domain,
        observations=observations,
        model_domain_path=domain_path,
        working_directory_path=working_directory_path,
        learning_algorithm=learning_algorithm,
    )




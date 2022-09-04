"""Utility functionality for the experiments"""
from pathlib import Path

from pddl_plus_parser.lisp_parsers import ProblemParser, DomainParser, TrajectoryParser

from experiments import NumericPerformanceCalculator
from utilities import LearningAlgorithmType


def init_numeric_performance_calculator(working_directory_path: Path, domain_file_name: str,
                                        learning_algorithm: LearningAlgorithmType) -> NumericPerformanceCalculator:
    """Initializes a numeric performance calculator objecy.

    :param working_directory_path: the directory path where the domain and problem files are located.
    :param domain_file_name: the name of the domain file.
    :param learning_algorithm: the type of learning algorithm to use.
    :return: the initialized numeric performance calculator object.
    """
    domain_path = working_directory_path / domain_file_name
    model_domain = partial_domain = DomainParser(domain_path=domain_path, partial_parsing=False).parse_domain()
    observations = []
    for trajectory_file_path in working_directory_path.glob("*.trajectory"):
        problem_path = working_directory_path / f"{trajectory_file_path.stem}.pddl"
        problem = ProblemParser(problem_path, partial_domain).parse_problem()
        new_observation = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_file_path)
        observations.append(new_observation)

    return NumericPerformanceCalculator(model_domain=model_domain,
                                        observations=observations,
                                        working_directory_path=working_directory_path,
                                        learning_algorithm=learning_algorithm)

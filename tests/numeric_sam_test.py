"""module tests for the Numeric SAM learning algorithm"""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation
from pytest import fixture

from sam_learning.learners.numeric_sam import NumericSAMLearner
from tests.consts import NUMERIC_DOMAIN_PATH, \
    NUMERIC_PROBLEM_PATH, DEPOT_NUMERIC_TRAJECTORY_PATH


@fixture()
def depot_domain() -> Domain:
    domain_parser = DomainParser(NUMERIC_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def depot_problem(depot_domain: Domain) -> Problem:
    return ProblemParser(problem_path=NUMERIC_PROBLEM_PATH, domain=depot_domain).parse_problem()


@fixture()
def numeric_observation(depot_domain: Domain, depot_problem: Problem) -> Observation:
    return TrajectoryParser(depot_domain, depot_problem).parse_trajectory(DEPOT_NUMERIC_TRAJECTORY_PATH)


@fixture()
def numeric_sam_learning(depot_domain: Domain) -> NumericSAMLearner:
    return NumericSAMLearner(depot_domain)


def test_learn_action_model_returns_learned_model(numeric_sam_learning: NumericSAMLearner,
                                                  numeric_observation: Observation):
    learned_model = numeric_sam_learning.learn_action_model([numeric_observation])
    print()
    print(learned_model.to_pddl())

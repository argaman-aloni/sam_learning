"""module tests for the SAM learning algorithm"""

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import GroundedPredicate, Domain, ActionCall, Predicate, Problem, Observation
from pytest import fixture

from sam_learning.learners import SAMLearner
from tests.consts import ELEVATORS_DOMAIN_PATH, ELEVATORS_PROBLEM_PATH, ELEVATORS_TRAJECTORY_PATH


@fixture()
def elevators_domain() -> Domain:
    domain_parser = DomainParser(ELEVATORS_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def elevators_problem(elevators_domain: Domain) -> Problem:
    return ProblemParser(problem_path=ELEVATORS_PROBLEM_PATH, domain=elevators_domain).parse_problem()


@fixture()
def elevators_observation(elevators_domain: Domain, elevators_problem: Problem) -> Observation:
    return TrajectoryParser(elevators_domain, elevators_problem).parse_trajectory(ELEVATORS_TRAJECTORY_PATH)


@fixture()
def sam_learning(elevators_domain: Domain) -> SAMLearner:
    return SAMLearner(elevators_domain)


def test_add_new_action_with_single_trajectory_component_adds_action_data_to_leaned_domain(
        sam_learning: SAMLearner, elevators_observation: Observation):
    observation_component = elevators_observation.components[0]
    previous_state = observation_component.previous_state
    next_state = observation_component.next_state
    test_action_call = observation_component.grounded_action_call

    sam_learning.add_new_action(grounded_action=test_action_call, previous_state=previous_state, next_state=next_state)

    added_action_name = "move-down-slow"

    assert added_action_name in sam_learning.partial_domain.actions
    learned_action_data = sam_learning.partial_domain.actions[added_action_name]
    print()
    print([str(p) for p in learned_action_data.positive_preconditions])
    assert [p.untyped_representation for p in learned_action_data.add_effects] == ["(lift-at ?lift ?f2)"]
    assert [p.untyped_representation for p in learned_action_data.delete_effects] == ["(lift-at ?lift ?f1)"]

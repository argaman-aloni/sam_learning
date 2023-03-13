"""module tests for the Numeric SAM learning algorithm"""
import json
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation
from pytest import fixture

from sam_learning.learners.numeric_sam import NumericSAMLearner
from tests.consts import NUMERIC_DOMAIN_PATH, \
    NUMERIC_PROBLEM_PATH, DEPOT_NUMERIC_TRAJECTORY_PATH, DEPOT_FLUENTS_MAP_PATH, SATELLITE_DOMAIN_PATH, \
    SATELLITE_PROBLEM_PATH, SATELLITE_NUMERIC_TRAJECTORY_PATH, SATELLITE_FLUENTS_MAP_PATH, \
    SATELLITE_PROBLEMATIC_PROBLEM_PATH, SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH, MINECRAFT_DOMAIN_PATH, \
    MINECRAFT_PROBLEM_PATH, MINECRAFT_TRAJECTORY_PATH, MINECRAFT_FLUENTS_MAP_PATH


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
def depot_fluents_map() -> Dict[str, List[str]]:
    with open(DEPOT_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def minecraft_fluents_map() -> Dict[str, List[str]]:
    with open(MINECRAFT_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def satellite_domain() -> Domain:
    domain_parser = DomainParser(SATELLITE_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()


@fixture()
def satellite_problem(satellite_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SATELLITE_PROBLEM_PATH, domain=satellite_domain).parse_problem()


@fixture()
def satellite_observation(satellite_domain: Domain, satellite_problem: Problem) -> Observation:
    return TrajectoryParser(satellite_domain, satellite_problem).parse_trajectory(SATELLITE_NUMERIC_TRAJECTORY_PATH)


@fixture()
def satellite_problem_problematic(satellite_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SATELLITE_PROBLEMATIC_PROBLEM_PATH, domain=satellite_domain).parse_problem()


@fixture()
def satellite_observation_problematic(satellite_domain: Domain, satellite_problem_problematic: Problem) -> Observation:
    return TrajectoryParser(satellite_domain, satellite_problem_problematic). \
        parse_trajectory(SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH)


@fixture()
def satellite_fluents_map() -> Dict[str, List[str]]:
    with open(SATELLITE_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def numeric_sam_learning(depot_domain: Domain, depot_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(depot_domain, depot_fluents_map)


@fixture()
def satellite_sam_learning(satellite_domain: Domain, satellite_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(satellite_domain, satellite_fluents_map)


@fixture()
def minecraft_domain() -> Domain:
    domain_parser = DomainParser(MINECRAFT_DOMAIN_PATH, partial_parsing=True)
    return domain_parser.parse_domain()

@fixture()
def minecraft_problem(minecraft_domain: Domain) -> Problem:
    return ProblemParser(problem_path=MINECRAFT_PROBLEM_PATH, domain=minecraft_domain).parse_problem()


@fixture()
def minecraft_observation(minecraft_domain: Domain, minecraft_problem: Problem) -> Observation:
    return TrajectoryParser(minecraft_domain, minecraft_problem).parse_trajectory(MINECRAFT_TRAJECTORY_PATH)


@fixture()
def minecraft_nsam(minecraft_domain: Domain, minecraft_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(minecraft_domain, minecraft_fluents_map)


def test_add_new_action_adds_action_to_fluents_storage(
        numeric_sam_learning: NumericSAMLearner, numeric_observation: Observation):
    initial_state = numeric_observation.components[0].previous_state
    action_call = numeric_observation.components[0].grounded_action_call
    next_state = numeric_observation.components[0].next_state
    numeric_sam_learning.add_new_action(
        grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in numeric_sam_learning.storage


def test_update_action_updates_action_in_the_storage(
        numeric_sam_learning: NumericSAMLearner, numeric_observation: Observation):
    initial_state = numeric_observation.components[0].previous_state
    action_call = numeric_observation.components[0].grounded_action_call
    next_state = numeric_observation.components[0].next_state
    numeric_sam_learning.add_new_action(
        grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    numeric_sam_learning.update_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in numeric_sam_learning.storage


def test_learn_action_model_returns_learned_model(numeric_sam_learning: NumericSAMLearner,
                                                  numeric_observation: Observation):
    learned_model, learning_metadata = numeric_sam_learning.learn_action_model([numeric_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_domain_returns_learned_model(satellite_sam_learning: NumericSAMLearner,
                                                                       satellite_observation: Observation):
    learned_model, learning_metadata = satellite_sam_learning.learn_action_model([satellite_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_with_problematic_trajectory_domain_returns_learned_model(
        satellite_sam_learning: NumericSAMLearner, satellite_observation_problematic: Observation):
    learned_model, learning_metadata = satellite_sam_learning.learn_action_model([satellite_observation_problematic])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())



def test_learn_action_model_with_minecraft_domain_creates_domain_with_correct_preconditions_and_effects(
        minecraft_nsam: NumericSAMLearner, minecraft_observation: Observation):
    learned_model, learning_metadata = minecraft_nsam.learn_action_model([minecraft_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


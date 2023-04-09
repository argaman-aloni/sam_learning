"""module tests for the Numeric SAM learning algorithm"""
import json
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, Observation
from pytest import fixture

from sam_learning.learners.numeric_sam import NumericSAMLearner
from tests.consts import DEPOT_FLUENTS_MAP_PATH, SATELLITE_FLUENTS_MAP_PATH, \
    SATELLITE_PROBLEMATIC_PROBLEM_PATH, SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH, MINECRAFT_FLUENTS_MAP_PATH


@fixture()
def satellite_problem_problematic(satellite_numeric_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SATELLITE_PROBLEMATIC_PROBLEM_PATH,
                         domain=satellite_numeric_domain).parse_problem()


@fixture()
def satellite_observation_problematic(satellite_numeric_domain: Domain,
                                      satellite_problem_problematic: Problem) -> Observation:
    return TrajectoryParser(satellite_numeric_domain, satellite_problem_problematic). \
        parse_trajectory(SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH)


@fixture()
def depot_fluents_map() -> Dict[str, List[str]]:
    with open(DEPOT_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def minecraft_fluents_map() -> Dict[str, List[str]]:
    with open(MINECRAFT_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def satellite_fluents_map() -> Dict[str, List[str]]:
    with open(SATELLITE_FLUENTS_MAP_PATH, "rt") as json_file:
        return json.load(json_file)


@fixture()
def depot_nsam(depot_domain: Domain, depot_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(depot_domain, depot_fluents_map)


@fixture()
def satellite_nsam(satellite_numeric_domain: Domain, satellite_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(satellite_numeric_domain, satellite_fluents_map)


@fixture()
def minecraft_nsam(minecraft_domain: Domain, minecraft_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(minecraft_domain, minecraft_fluents_map)


def test_add_new_action_adds_action_to_fluents_storage(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    depot_nsam.add_new_action(
        grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in depot_nsam.storage


def test_update_action_updates_action_in_the_storage(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    depot_nsam.add_new_action(
        grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    depot_nsam.update_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in depot_nsam.storage


def test_learn_action_model_returns_learned_model(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    learned_model, learning_metadata = depot_nsam.learn_action_model([depot_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_domain_returns_learned_model(
        satellite_nsam: NumericSAMLearner, satellite_numeric_observation: Observation):
    learned_model, learning_metadata = satellite_nsam.learn_action_model([satellite_numeric_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_with_problematic_trajectory_domain_returns_learned_model(
        satellite_nsam: NumericSAMLearner, satellite_observation_problematic: Observation):
    learned_model, learning_metadata = satellite_nsam.learn_action_model([satellite_observation_problematic])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_with_minecraft_domain_creates_domain_with_correct_preconditions_and_effects(
        minecraft_nsam: NumericSAMLearner, minecraft_observation: Observation):
    learned_model, learning_metadata = minecraft_nsam.learn_action_model([minecraft_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())

"""module tests for the Numeric SAM learning algorithm"""
import json
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import ProblemParser, TrajectoryParser, DomainParser
from pddl_plus_parser.models import Domain, Problem, Observation, Predicate
from pytest import fixture

from sam_learning.learners.numeric_sam import NumericSAMLearner, PolynomialSAMLearning
from tests.consts import (
    DEPOT_FLUENTS_MAP_PATH,
    SATELLITE_FLUENTS_MAP_PATH,
    SATELLITE_PROBLEMATIC_PROBLEM_PATH,
    SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH,
    MINECRAFT_FLUENTS_MAP_PATH,
    sync_snapshot,
    MINECRAFT_MEDIUM_DOMAIN_PATH,
    MINECRAFT_MEDIUM_TRAJECTORY_PATH,
    MINECRAFT_MEDIUM_FLUENTS_MAP_PATH,
    MINECRAFT_SMALL_DOMAIN_PATH,
    MINECRAFT_SMALL_TRAJECTORY_PATH,
    DRIVERLOG_POLY_DOMAIN_PATH,
    DRIVERLOG_POLY_PROBLEM_PATH,
    DRIVERLOG_POLY_TRAJECTORY_PATH,
)


@fixture()
def satellite_problem_problematic(satellite_numeric_domain: Domain) -> Problem:
    return ProblemParser(problem_path=SATELLITE_PROBLEMATIC_PROBLEM_PATH, domain=satellite_numeric_domain).parse_problem()


@fixture()
def satellite_observation_problematic(satellite_numeric_domain: Domain, satellite_problem_problematic: Problem) -> Observation:
    return TrajectoryParser(satellite_numeric_domain, satellite_problem_problematic).parse_trajectory(SATELLITE_PROBLEMATIC_NUMERIC_TRAJECTORY_PATH)


@fixture()
def minecraft_medium_preconditions_fluents_map() -> Dict[str, List[str]]:
    return json.load(open(MINECRAFT_MEDIUM_FLUENTS_MAP_PATH, "rt"))


@fixture()
def minecraft_medium_domain() -> Domain:
    return DomainParser(MINECRAFT_MEDIUM_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def driverlog_polynomial_domain() -> Domain:
    return DomainParser(DRIVERLOG_POLY_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def driverlog_polynomial_problem(driverlog_polynomial_domain: Domain) -> Problem:
    return ProblemParser(problem_path=DRIVERLOG_POLY_PROBLEM_PATH, domain=driverlog_polynomial_domain).parse_problem()


@fixture()
def driverlog_polynomial_observation(driverlog_polynomial_domain: Domain, driverlog_polynomial_problem: Problem) -> Observation:
    return TrajectoryParser(driverlog_polynomial_domain, driverlog_polynomial_problem).parse_trajectory(DRIVERLOG_POLY_TRAJECTORY_PATH)


@fixture()
def driverlog_polynomial_nsam(driverlog_polynomial_domain: Domain, minecraft_medium_preconditions_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return PolynomialSAMLearning(driverlog_polynomial_domain, {action: [] for action in driverlog_polynomial_domain.actions})


@fixture()
def minecraft_medium_observation(minecraft_medium_domain: Domain) -> Observation:
    return TrajectoryParser(minecraft_medium_domain).parse_trajectory(MINECRAFT_MEDIUM_TRAJECTORY_PATH)


@fixture()
def minecraft_medium_sam(minecraft_medium_domain: Domain, minecraft_medium_preconditions_fluents_map: Dict[str, List[str]]) -> NumericSAMLearner:
    return NumericSAMLearner(minecraft_medium_domain, minecraft_medium_preconditions_fluents_map)


@fixture()
def minecraft_small_domain() -> Domain:
    return DomainParser(MINECRAFT_SMALL_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def minecraft_small_observation(minecraft_small_domain: Domain) -> Observation:
    return TrajectoryParser(minecraft_small_domain).parse_trajectory(MINECRAFT_SMALL_TRAJECTORY_PATH)


@fixture()
def minecraft_small_sam(minecraft_small_domain: Domain) -> NumericSAMLearner:
    return NumericSAMLearner(minecraft_small_domain)


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
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in depot_nsam.storage


def test_add_new_action_adds_discrete_preconditions_to_the_learned_action(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    learned_action = depot_nsam.partial_domain.actions[action_call.name]
    action_discrete_preconditions = [precondition for _, precondition in learned_action.preconditions if isinstance(precondition, Predicate)]
    assert len(action_discrete_preconditions) > 0


def test_add_new_action_adds_the_required_predicates_to_the_action_preconditions(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    learned_action = depot_nsam.partial_domain.actions[action_call.name]
    action_discrete_preconditions = [precondition for _, precondition in learned_action.preconditions if isinstance(precondition, Predicate)]
    preconditions_str = {precondition.untyped_representation for precondition in action_discrete_preconditions}
    assert preconditions_str.issuperset({"(at ?x ?y)"})


def test_update_action_updates_action_in_the_storage(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    depot_nsam.update_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    assert action_call.name in depot_nsam.storage


def test_update_action_adds_discrete_preconditions_to_the_learned_action(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    depot_nsam.update_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    learned_action = depot_nsam.partial_domain.actions[action_call.name]
    action_discrete_preconditions = [precondition for _, precondition in learned_action.preconditions if isinstance(precondition, Predicate)]
    assert len(action_discrete_preconditions) > 0


def test_update_action_adds_the_required_predicates_to_the_action_preconditions(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    initial_state = depot_observation.components[0].previous_state
    action_call = depot_observation.components[0].grounded_action_call
    next_state = depot_observation.components[0].next_state
    sync_snapshot(depot_nsam, depot_observation.components[0], depot_observation.grounded_objects)
    depot_nsam.add_new_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    depot_nsam.update_action(grounded_action=action_call, previous_state=initial_state, next_state=next_state)
    learned_action = depot_nsam.partial_domain.actions[action_call.name]
    action_discrete_preconditions = [precondition for _, precondition in learned_action.preconditions if isinstance(precondition, Predicate)]
    preconditions_str = {precondition.untyped_representation for precondition in action_discrete_preconditions}
    assert preconditions_str.issuperset({"(at ?x ?y)"})


def test_handle_single_trajectory_component_does_not_remove_the_required_predicates_to_the_action_preconditions(
    depot_nsam: NumericSAMLearner, depot_observation: Observation
):
    depot_nsam.current_trajectory_objects = depot_observation.grounded_objects
    depot_nsam.handle_single_trajectory_component(depot_observation.components[0])
    learned_action = depot_nsam.partial_domain.actions["drive"]
    action_discrete_preconditions = [precondition for _, precondition in learned_action.preconditions if isinstance(precondition, Predicate)]
    preconditions_str = {precondition.untyped_representation for precondition in action_discrete_preconditions}
    assert preconditions_str.issuperset({"(at ?x ?y)"})


def test_learn_action_model_returns_learned_model(depot_nsam: NumericSAMLearner, depot_observation: Observation):
    learned_model, learning_metadata = depot_nsam.learn_action_model([depot_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_domain_returns_learned_model(satellite_nsam: NumericSAMLearner, satellite_numeric_observation: Observation):
    learned_model, learning_metadata = satellite_nsam.learn_action_model([satellite_numeric_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_for_satellite_with_problematic_trajectory_domain_returns_learned_model(
    satellite_nsam: NumericSAMLearner, satellite_observation_problematic: Observation
):
    learned_model, learning_metadata = satellite_nsam.learn_action_model([satellite_observation_problematic])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_with_minecraft_domain_creates_domain_with_correct_preconditions_and_effects(
    minecraft_nsam: NumericSAMLearner, minecraft_observation: Observation
):
    learned_model, learning_metadata = minecraft_nsam.learn_action_model([minecraft_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_with_minecraft_medium_domain_creates_domain_with_correct_preconditions_and_effects(
    minecraft_medium_sam: NumericSAMLearner, minecraft_medium_observation: Observation
):
    learned_model, learning_metadata = minecraft_medium_sam.learn_action_model([minecraft_medium_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())


def test_learn_action_model_with_minecraft_small_domain_creates_domain_with_correct_preconditions_and_effects(
    minecraft_small_sam: NumericSAMLearner, minecraft_small_observation: Observation
):
    learned_model, learning_metadata = minecraft_small_sam.learn_action_model([minecraft_small_observation])
    print()
    print(learning_metadata)
    print(learned_model.to_pddl())



def test_learn_action_model_with_driverlog_polynomial_domain_returns_non_trivial_polynomial_conditions_to_learned_actions_and_does_not_fail_with_error(
    driverlog_polynomial_nsam: PolynomialSAMLearning, driverlog_polynomial_observation: Observation
):
    try:
        learned_model, learning_metadata = driverlog_polynomial_nsam.learn_action_model([driverlog_polynomial_observation])
        print()
        print(learning_metadata)
        print(learned_model.to_pddl())
    except Exception as e:
        assert False, f"An error occurred: {e}"
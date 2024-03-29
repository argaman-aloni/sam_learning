"""Module test for the vocabulary_creator module."""
from pddl_plus_parser.models import Domain, Problem
from pytest import fixture

from sam_learning.core import VocabularyCreator


@fixture()
def vocabulary_creator() -> VocabularyCreator:
    return VocabularyCreator()


def test_create_vocabulary_creates_grounded_predicates_only_for_those_with_matching_types(
        elevators_domain: Domain, vocabulary_creator: VocabularyCreator, elevators_problem: Problem):
    vocabulary_predicates = vocabulary_creator.create_vocabulary(
        domain=elevators_domain,
        observed_objects={"n1": elevators_problem.objects["n1"], "n2": elevators_problem.objects["n2"]})
    assert list(vocabulary_predicates.keys()) == ['(above ?floor1 ?floor2)', '(next ?n1 ?n2)']


def test_create_vocabulary_creates_grounded_predicates_when_given_two_types_of_objects(
        elevators_domain: Domain, vocabulary_creator: VocabularyCreator, elevators_problem: Problem):
    vocabulary_predicates = vocabulary_creator.create_vocabulary(
        domain=elevators_domain,
        observed_objects={"p0": elevators_problem.objects["p0"], "slow0-0": elevators_problem.objects["slow0-0"]})
    assert list(vocabulary_predicates.keys()) == ['(boarded ?person ?lift)']


def test_create_vocabulary_creates_grounded_predicates_when_constants_and_objects(
        woodworking_domain: Domain, vocabulary_creator: VocabularyCreator, woodworking_problem: Problem):
    vocabulary_predicates = vocabulary_creator.create_vocabulary(
        domain=woodworking_domain,
        observed_objects={"b0": woodworking_problem.objects["b0"],
                          "verysmooth": woodworking_domain.constants["verysmooth"],
                          "grinder0": woodworking_problem.objects["grinder0"]})
    assert set(vocabulary_predicates.keys()) == {'(surface-condition ?obj ?surface)', '(available ?obj)',
                                                 '(is-smooth ?surface)', '(has-colour ?agent ?colour)',
                                                 '(grind-treatment-change ?agent ?old ?new)'}


def test_create_lifted_vocabulary_creates_lifted_predicates_only_for_those_with_matching_types(
        woodworking_domain: Domain, vocabulary_creator: VocabularyCreator):
    vocabulary_predicates = vocabulary_creator.create_lifted_vocabulary(
        domain=woodworking_domain,
        possible_parameters=woodworking_domain.actions["do-grind"].signature)
    assert vocabulary_predicates is not None


def test_create_lifted_functions_vocabulary_creates_correct_functions(
        depot_domain: Domain, vocabulary_creator: VocabularyCreator):
    vocabulary_functions = vocabulary_creator.create_lifted_functions_vocabulary(
        domain=depot_domain,
        possible_parameters=depot_domain.actions["unload"].signature)
    assert vocabulary_functions is not None


def test_create_grounded_actions_vocabulary_creates_all_possible_assignments_of_actions(
        depot_domain: Domain, vocabulary_creator: VocabularyCreator, depot_problem: Problem):
    observed_objects = depot_problem.objects
    vocabulary_actions = vocabulary_creator.create_grounded_actions_vocabulary(
        domain=depot_domain,
        observed_objects=observed_objects)
    assert vocabulary_actions is not None
    assert len({action.name for action in vocabulary_actions}) == len(depot_domain.actions)
    print([str(action) for action in vocabulary_actions])


def test_create_grounded_actions_vocabulary_creates_only_unique_actions(
        depot_domain: Domain, vocabulary_creator: VocabularyCreator, depot_problem: Problem):
    observed_objects = depot_problem.objects
    vocabulary_actions = vocabulary_creator.create_grounded_actions_vocabulary(
        domain=depot_domain,
        observed_objects=observed_objects)
    assert len({str(action) for action in vocabulary_actions}) == len([action for action in vocabulary_actions])
    print([str(action) for action in vocabulary_actions])


def test_create_grounded_actions_vocabulary_creates_actions_that_are_applicable_in_the_state(
        minecraft_large_domain: Domain, vocabulary_creator: VocabularyCreator, minecraft_large_problem: Problem):
    observed_objects = minecraft_large_problem.objects
    vocabulary_actions = vocabulary_creator.create_grounded_actions_vocabulary(
        domain=minecraft_large_domain,
        observed_objects=observed_objects)
    action_signatures = {str(action) for action in vocabulary_actions}
    assert "(tp_to cell15 cell0)" in action_signatures

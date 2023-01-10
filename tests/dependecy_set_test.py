"""Module test for the dependency set class."""
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Predicate
from pytest import fixture

from sam_learning.core.dependency_set import create_antecedents_combination, DependencySet
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH

TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS = 378 + 28


@fixture()
def woodworking_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def woodworking_predicates(woodworking_domain: Domain) -> List[Predicate]:
    return list(woodworking_domain.predicates.values())


def test_create_antecedents_combination_with_max_size_1():
    """Test the creation of antecedents combinations with max size 1."""
    antecedents = {"a", "b", "c"}
    expected_antecedents_combinations = [{"a"}, {"b"}, {"c"}]
    antecedents_combinations = create_antecedents_combination(antecedents, 1)
    assert len(antecedents_combinations) == 3
    for expected_combination in expected_antecedents_combinations:
        assert expected_combination in antecedents_combinations


def test_create_antecedents_combination_with_max_size_2():
    """Test the creation of antecedents combinations with max size 2."""
    antecedents = {"a", "b", "c"}
    expected_antecedents_combinations = [{"a"}, {"b"}, {"c"}, {"a", "b"}, {"a", "c"}, {"b", "c"}]
    antecedents_combinations = create_antecedents_combination(antecedents, 2)
    assert len(antecedents_combinations) == 6
    for expected_combination in expected_antecedents_combinations:
        assert expected_combination in antecedents_combinations


def test_create_antecedents_with_real_domain_predicates(woodworking_predicates: List[Predicate]):
    """Test the creation of antecedents combinations with real domain predicates."""
    antecedents = {predicate.untyped_representation for predicate in woodworking_predicates}
    antecedents_combinations = create_antecedents_combination(antecedents, 2)
    for expected_combination in antecedents:
        assert {expected_combination} in antecedents_combinations

    print(antecedents_combinations)


def test_initialize_dependencies_with_real_domain_predicates_initialize_both_negative_and_positive_predicates(
        woodworking_predicates: List[Predicate]):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    assert len(dependency_set.dependencies) == 2 * len(woodworking_predicates)


def test_initialize_dependencies_with_real_domain_predicates_creates_correct_set_of_literals(
        woodworking_predicates: List[Predicate]):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    # C(28, 2) = 28! / (2! * (28-2)!) = 28! / (2! * 26!) = (28 * 27) / (1 * 2) = 378 sets.
    # The number of sets of size 1 that can be created from 28 objects is 28.
    assert len(dependency_set.dependencies[
                   woodworking_predicates[0].untyped_representation]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS


def test_extract_superset_dependencies_creates_supersets_of_dependencies_containing_the_input_literals():
    test_literals = ["a", "b", "c"]
    dependency_set = DependencySet(max_size_antecedents=3)
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 3)
    dependency_set.dependencies = {literal: possible_literals_combinations for literal in test_literals}

    expected_superset_dependencies = [{"a", "b", "c"}, {"a", "b"}, {"a", "c"}, {"a"}]
    superset_dependencies = dependency_set._extract_superset_dependencies("a", [{"a"}])
    assert len(superset_dependencies) == 4
    for expected_superset_dependency in expected_superset_dependencies:
        assert expected_superset_dependency in superset_dependencies


def test_remove_dependencies_removed_correct_set_of_literals_and_all_subsets(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    dependency_set.remove_dependencies(tested_predicate, predicates_to_remove)
    assert len(dependency_set.dependencies[tested_predicate]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS - 3


def test_is_safe_literal_returns_literal_unsafe_if_contains_more_that_one_item(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    assert not dependency_set.is_safe_literal(tested_predicate, predicates_to_remove)


def test_is_safe_literal_returns_literal_safe_if_contains_zero_items(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    tested_predicate = "(available ?obj)"
    dependency_set.dependencies[tested_predicate] = []
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    assert dependency_set.is_safe_literal(tested_predicate, predicates_to_remove)


def test_is_safe_literal_returns_literal_safe_if_contains_one_item(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    tested_predicate = "(available ?obj)"
    dependency_set.dependencies[tested_predicate] = [{"(available ?obj)"}]
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    assert dependency_set.is_safe_literal(tested_predicate, predicates_to_remove)


def test_is_safe_returns_false_on_initialized_literals_set(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    assert not dependency_set.is_safe(predicates_to_remove)


def test_is_safe_conditional_effect_returns_false_on_initialized_literals_set(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    assert not dependency_set.is_conditional_effect("(is-smooth ?surface)")


def test_extract_restrictive_conditions_creates_non_empty_list(woodworking_predicates: List[Predicate]):
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    literals_str = {literal.untyped_representation for literal in woodworking_predicates}
    literals_str.update({f"(not {literal.untyped_representation})" for literal in woodworking_predicates})
    preconditions = set()
    tested_literal = woodworking_predicates[0].untyped_representation

    conditions = dependency_set.extract_restrictive_conditions(preconditions, tested_literal)
    assert len(conditions) > 0
    print(conditions)


def test_extract_restrictive_conditions_creates_conditions_that_do_not_include_precondition_literal():
    test_literals = ["a", "b", "c"]
    dependency_set = DependencySet(max_size_antecedents=3)
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 3)
    dependency_set.dependencies = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"a"}
    tested_literal = "a"

    conditions = dependency_set.extract_restrictive_conditions(preconditions, tested_literal)
    assert not conditions.startswith("(or ")
    print(conditions)


def test_extract_restrictive_conditions_creates_conditions_that_include_precondition_literal(
        woodworking_predicates: List[Predicate]):
    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    literals_str = {literal.untyped_representation for literal in woodworking_predicates}
    literals_str.update({f"(not {literal.untyped_representation})" for literal in woodworking_predicates})
    tested_predicate = "(available ?obj)"
    preconditions = set()

    conditions = dependency_set.extract_restrictive_conditions(preconditions, tested_predicate)
    assert conditions.startswith(f"(or {tested_predicate} ")


def test_extract_restrictive_conditions_creates_conditions_with_negated_literals():
    test_literals = ["a", "b", "c"]
    dependency_set = DependencySet(max_size_antecedents=1)
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 1)
    dependency_set.dependencies = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"a"}
    tested_literal = "a"

    conditions = dependency_set.extract_restrictive_conditions(preconditions, tested_literal)
    assert "(or (not b))" in conditions
    assert "(or (not a))" in conditions
    assert "(or (not c))" in conditions


def test_remove_preconditions_literals_correctly_removed_preconditions_from_the_depencency_set(
        woodworking_predicates: List[Predicate]):
    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    preconditions = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    dependency_set.remove_preconditions_literals(preconditions)

    for literal in preconditions:
        assert literal not in dependency_set.dependencies


def test_extract_restrictive_conditions_creates_conditions_with_negated_literals_and_positive_literals_when_is_effect():
    test_literals = ["(a)", "(b)", "(c)"]
    dependency_set = DependencySet(max_size_antecedents=1)
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 1)
    dependency_set.dependencies = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"(a)"}
    tested_literal = "(a)"

    conditions = dependency_set.extract_restrictive_conditions(preconditions, tested_literal, is_effect=True)
    assert "(or (not (b)))" in conditions
    assert "(or (not (a)))" in conditions
    assert "(or (not (c)))" in conditions
    assert "(a)" in conditions
    assert "(b)" in conditions
    assert "(c)" in conditions

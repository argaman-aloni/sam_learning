"""Module test for the dependency set class."""
from typing import List

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Predicate
from pytest import fixture

from sam_learning.core.dependency_set import create_antecedents_combination, DependencySet
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH


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
    assert len(dependency_set.dependencies[woodworking_predicates[0].untyped_representation]) == 378


def test_remove_dependencies_removed_correct_set_of_literals_and_all_subsets(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    dependency_set.remove_dependencies(tested_predicate, predicates_to_remove)
    assert len(dependency_set.dependencies[tested_predicate]) == 378 - 3


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

    assert not dependency_set.is_safe_conditional_effect("(is-smooth ?surface)")


def test_extract_restrictive_conditions_creates_not_empty_list(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2)
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    literals_str = {literal.untyped_representation for literal in woodworking_predicates}
    literals_str.update({f"(not {literal.untyped_representation})" for literal in woodworking_predicates})

    conditions = dependency_set.extract_restrictive_conditions()
    assert len(conditions) > 0
    print(conditions)

def test_extract_restrictive_conditions_converts_all_positive_predicates_to_negatives_and_negatives_to_positive(
        woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=1)
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    literals_str = {literal.untyped_representation for literal in woodworking_predicates}
    literals_str.update({f"(not {literal.untyped_representation})" for literal in woodworking_predicates})

    conditions = dependency_set.extract_restrictive_conditions()
    antecedents = dependency_set.dependencies[tested_predicate]
    for antecedent in antecedents:
        antecedent_str = antecedent.pop()
        if antecedent_str.startswith("(not"):
            assert antecedent_str[5:-1] in conditions[0]

        else:
            assert f"(not {antecedent_str})" in conditions[0]


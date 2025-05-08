"""Module test for the dependency set class."""
from typing import List, Set

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Predicate, PDDLType
from pytest import fixture

from sam_learning.core import VocabularyCreator, DependencySet
from sam_learning.core.propositional_operations.logical_expression_operations import create_dnf_combinations, create_cnf_combination
from tests.consts import WOODWORKING_COMBINED_DOMAIN_PATH

TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS = 378 + 28

OBJECT_TYPE = PDDLType(name="object")


@fixture()
def woodworking_domain() -> Domain:
    return DomainParser(WOODWORKING_COMBINED_DOMAIN_PATH, partial_parsing=True).parse_domain()


@fixture()
def woodworking_predicates(woodworking_domain: Domain) -> List[Predicate]:
    domain_predicates = woodworking_domain.predicates.values()
    vocabulary = set(domain_predicates)
    for predicate in domain_predicates:
        negative_predicate = predicate.copy()
        negative_predicate.is_positive = False
        vocabulary.add(negative_predicate)
    return list(vocabulary)


@fixture()
def do_saw_predicates(woodworking_domain: Domain) -> Set[Predicate]:
    lifted_action_signature = woodworking_domain.actions["do-saw-small"].signature
    vocabulary = VocabularyCreator().create_lifted_vocabulary(woodworking_domain, lifted_action_signature)
    return vocabulary


def test_create_antecedents_with_real_domain_predicates(woodworking_predicates: List[Predicate]):
    """Test the creation of antecedents combinations with real domain predicates."""
    antecedents = {predicate.untyped_representation for predicate in woodworking_predicates}
    antecedents_combinations = create_cnf_combination(antecedents, 2)
    for expected_combination in antecedents:
        assert {expected_combination} in antecedents_combinations

    print(antecedents_combinations)


def test_extract_superset_dependencies_when_the_literal_is_not_subset_of_other_dependencies_returns_empty_list():
    """Test the extraction of superset dependencies when the literal is not subset of other dependencies."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.possible_antecedents = {"a": [{"a", "b"}, {"a", "c"}], "b": [{"b", "c"}]}
    superset_dep = dependency_set._extract_superset_dependencies("a", {"a", "d"})
    assert len(superset_dep) == 2


def test_extract_superset_dependencies_creates_supersets_of_dependencies_containing_the_input_literals():
    """Test the extraction of superset dependencies when the literal is a subset of other dependencies."""
    test_literals = ["a", "b", "c"]
    dependency_set = DependencySet(max_size_antecedents=3, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 3)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}

    expected_superset_dependencies = [{"a", "b", "c"}, {"a", "b"}, {"a", "c"}, {"a"}]
    superset_dependencies = dependency_set._extract_superset_dependencies("a", {"a"})
    assert len(superset_dependencies) == 4
    for expected_superset_dependency in expected_superset_dependencies:
        assert expected_superset_dependency in superset_dependencies


def test_initialize_dependencies_with_real_domain_predicates_initialize_both_negative_and_positive_predicates_as_keys(
    woodworking_predicates: List[Predicate],
):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(woodworking_predicates)
    assert len(dependency_set.possible_antecedents) == len(woodworking_predicates)


def test_initialize_dependencies_with_real_domain_predicates_creates_correct_set_of_literals_in_the_values(woodworking_predicates: List[Predicate]):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    # C(28, 2) = 28! / (2! * (28-2)!) = 28! / (2! * 26!) = (28 * 27) / (1 * 2) = 378 sets.
    # The number of sets of size 1 that can be created from 28 objects is 28.
    assert len(dependency_set.possible_antecedents[woodworking_predicates[0].untyped_representation]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS


def test_remove_dependencies_removes_correct_literals_on_simple_case_with_no_superset_literals():
    """Test the removal of a dependency from the dependency set when creating a simple scenario."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    antecedents = [{"a"}, {"b"}, {"c"}, {"a", "b"}, {"a", "c"}, {"b", "c"}]
    init_antecedents_length = len(antecedents)
    dependency_set.possible_antecedents = {"a": antecedents}
    dependency_set.possible_disjunctive_antecedents = {"a": []}

    tested_literal = "a"
    literals_to_remove = {"a", "b"}
    dependency_set.remove_dependencies(tested_literal, literals_to_remove)

    expected_removed_literals = create_cnf_combination(literals_to_remove, 2)
    assert len(dependency_set.possible_antecedents[tested_literal]) == init_antecedents_length - len(expected_removed_literals)


def test_remove_dependencies_removed_correct_set_of_literals_and_all_subsets(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set with real domain predicates used as input."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    expected_removed_literals = create_cnf_combination(predicates_to_remove, 2)
    dependency_set.remove_dependencies(tested_predicate, predicates_to_remove)
    assert len(dependency_set.possible_antecedents[tested_predicate]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS - len(expected_removed_literals)


def test_remove_preconditions_literals_correctly_removed_preconditions_from_the_dependency_set(woodworking_predicates: List[Predicate]):
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))

    preconditions = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    dependency_set.remove_preconditions_literals(preconditions)

    for literal in preconditions:
        assert literal not in dependency_set.possible_antecedents
        for antecedents_disjunction in dependency_set.possible_antecedents.values():
            for conjunction in antecedents_disjunction:
                assert literal not in conjunction


def test_is_safe_literal_returns_literal_unsafe_if_contains_more_that_one_item(woodworking_predicates: List[Predicate]):
    """Test the check if a literal is safe when the literal contains more than one item."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    assert not dependency_set.is_safe_literal(tested_predicate, predicates_to_remove)


def test_construct_restrictive_preconditions_returns_none_if_result_predicate_is_in_preconditions(
    do_saw_predicates: Set[Predicate], woodworking_domain: Domain
):
    test_signature = woodworking_domain.actions["do-saw-small"].signature
    dependency_set = DependencySet(max_size_antecedents=2, action_signature=test_signature, domain_constants=woodworking_domain.constants)
    dependency_set.initialize_dependencies(do_saw_predicates)
    tested_literal = "(available ?p)"

    restrictive_precondition = dependency_set.construct_restrictive_preconditions(preconditions={tested_literal}, literal=tested_literal)
    assert restrictive_precondition is None


def test_construct_restrictive_preconditions_a_precondition_object_with_literals_that_is_not_none(
    do_saw_predicates: Set[Predicate], woodworking_domain: Domain
):
    test_signature = woodworking_domain.actions["do-saw-small"].signature
    dependency_set = DependencySet(max_size_antecedents=2, action_signature=test_signature, domain_constants=woodworking_domain.constants)
    dependency_set.initialize_dependencies(do_saw_predicates)
    tested_literal = "(available ?p)"

    restrictive_precondition = dependency_set.construct_restrictive_preconditions(preconditions=set(), literal=tested_literal)
    assert restrictive_precondition is not None


def test_construct_restrictive_preconditions_returns_none_if_the_negated_antecedents_are_the_same_as_the_preconditions():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 1)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"(not (a ))", "(not (b ))", "(not (c ))"}
    tested_literal = "(a )"

    conditions = dependency_set.construct_restrictive_preconditions(preconditions, tested_literal)
    print(str(conditions))
    assert not conditions


def test_construct_restrictive_preconditions_creates_conditions_that_do_not_include_precondition_literal_if_is_not_effect_and_negated_effect_is_precondition():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 1)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"(not (a ))"}
    tested_literal = "(a )"

    conditions = dependency_set.construct_restrictive_preconditions(preconditions, tested_literal)
    restrictive_conditions = [cond.untyped_representation for _, cond in conditions if isinstance(cond, Predicate)]
    assert sorted(restrictive_conditions) == ["(not (b ))", "(not (c ))"]


def test_construct_restrictive_preconditions_creates_conditions_that_do_not_include_precondition_literal_if_is_effect_and_negated_effect_is_precondition():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 1)
    tested_literal = "(a )"
    dependency_set.possible_antecedents = {tested_literal: possible_literals_combinations}
    preconditions = {"(not (a ))"}

    conditions = dependency_set.construct_restrictive_preconditions(preconditions, tested_literal, is_effect=True)
    restrictive_conditions = [cond.untyped_representation for _, cond in conditions if isinstance(cond, Predicate)]
    assert sorted(restrictive_conditions) == ["(a )", "(b )", "(c )", "(not (b ))", "(not (c ))"]
    print(str(conditions))


def test_construct_restrictive_preconditions_creates_nested_condition_with_correct_elements_with_size_two_antecedents():
    test_literals = ["(a )", "(b )"]
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 2)
    tested_literal = "(effect )"
    dependency_set.possible_antecedents = {tested_literal: possible_literals_combinations}
    condition = dependency_set.construct_restrictive_preconditions(set(), tested_literal, is_effect=False)
    # the preconditions should be (effect ) V (~aV~b)
    print(str(condition))
    assert condition.binary_operator == "or"
    assert len(condition.operands) == 2
    for operand in condition.operands:
        if isinstance(operand, Predicate):
            assert operand.untyped_representation == "(effect )"
        else:
            assert operand.binary_operator == "and"
            assert len(operand.operands) == 2
            for inner_operand in operand.operands:
                assert isinstance(inner_operand, Predicate)
                assert inner_operand.untyped_representation in ["(not (a ))", "(not (b ))"]


def test_construct_restrictive_preconditions_creates_nested_condition_with_correct_elements_with_size_three_antecedents():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=3, action_signature={}, domain_constants={})
    possible_literals_combinations = create_cnf_combination(set(test_literals), 3)
    tested_literal = "(effect )"
    dependency_set.possible_antecedents = {tested_literal: possible_literals_combinations}
    condition = dependency_set.construct_restrictive_preconditions(set(), tested_literal, is_effect=False)
    # the preconditions should be (effect ) V (^~a ^~b ^~c)
    print(str(condition))
    assert condition.binary_operator == "or"
    assert len(condition.operands) == 2
    for operand in condition.operands:
        if isinstance(operand, Predicate):
            assert operand.untyped_representation == "(effect )"
        else:
            assert operand.binary_operator == "and"
            assert len(operand.operands) == 3

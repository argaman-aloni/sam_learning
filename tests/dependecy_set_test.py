"""Module test for the dependency set class."""
from typing import List, Set

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Predicate, PDDLType
from pytest import fixture, raises

from sam_learning.core import VocabularyCreator
from sam_learning.core.dependency_set import create_antecedents_combination, DependencySet, minimize_cnf_clauses, \
    minimize_dnf_clauses
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


def test_minimize_cnf_clauses_empty_clauses():
    clauses = []
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == []


def test_minimize_cnf_clauses_single_clause():
    clauses = [{'(p)', '(q)', '(r)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)', '(q)', '(r)'}]


def test_minimize_cnf_clauses_single_unit_clause():
    clauses = [{'(p)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)'}]


def test_minimize_cnf_clauses_single_unit_clause_and_single_non_unit_clause():
    clauses = [{'(p)'}, {'(p)', '(q)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)'}]


def test_minimize_cnf_clauses_multiple_non_unit_clauses():
    clauses = [{'(p)', '(q)'}, {'(p)', '(r)'}, {'(q)', '(r)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)', '(q)'}, {'(p)', '(r)'}, {'(q)', '(r)'}]


def test_minimize_cnf_clauses_unit_clause_and_complementary_literals():
    clauses = [{'(p)'}, {'(not (q))', '(r)'}, {'(not (p))', '(q)'}, {'(s)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert len(minimized_clauses) == 4
    expected_clauses = [{'(p)'}, {'(not (q))', '(r)'}, {'(q)'}, {'(s)'}]
    for clause in expected_clauses:
        assert clause in minimized_clauses


def test_minimize_cnf_clauses_non_unit_clause_and_complementary_literals():
    clauses = [{'(p)', '(q)'}, {'(not (p))'}, {'(r)'}, {'(not (q))'}, {'(s)'}, {'(t)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(not (p))'}, {'(r)'}, {'(not (q))'}, {'(s)'}, {'(t)'}]


def test_minimize_cnf_clauses_multiple_complementary_literals():
    clauses = [{'(not (p))'}, {'(not q)', 'r'}, {'(not (r))', 'p'}, {'(p)'}]
    with raises(ValueError):
        minimize_cnf_clauses(clauses)


def test_minimize_cnf_clauses_assumptions():
    clauses = [{'(p)', '(not (q))', '(r)'}, {'(not (p))', '(s)'}, {'(q)', '(r)', '(t)'}, {'(not (t))', '(u)', '(v)'}]
    assumptions = {'(p)', '(t)', '(v)'}
    minimized_clauses = minimize_cnf_clauses(clauses, assumptions)
    assert minimized_clauses == [{'(s)'}]


def test_minimize_dnf_clauses_pddl_format_1():
    # Test with one clause, no simplification possible
    clauses = [{"(on table block1)"}]
    expected_minimized_clauses = [{"(on table block1)"}]
    assert minimize_dnf_clauses(clauses) == expected_minimized_clauses


def test_minimize_dnf_clauses_pddl_format_2():
    # Test with one clause, negation of assumption leads to contradiction
    clauses = [{"(on table block1)", "(not (on table block1))"}]
    with raises(ValueError):
        minimize_dnf_clauses(clauses, {"(not (on table block1))"})


def test_minimize_dnf_clauses_pddl_format_3():
    # Test with two clauses, the assumption is part of one clause and its negation is part of the other clause
    clauses = [{"(on table block1)", "(not (on table block2))"}, {"(not (on table block1))", "(on table block2)"}]
    expected_minimized_clauses = [{"(not (on table block2))"}]
    assert minimize_dnf_clauses(clauses, {"(on table block1)"}) == expected_minimized_clauses


def test_minimize_dnf_clauses_pddl_format_4():
    # Test with two clauses, one of them is always false
    clauses = [{"(on table block1)"}, {"(not (on table block1))", "(not (on table block2))"}]
    expected_minimized_clauses = [{"(on table block1)"}, {"(not (on table block1))"}]
    assert minimize_dnf_clauses(clauses, {"(not (on table block2))"}) == expected_minimized_clauses


def test_minimize_dnf_clauses_pddl_format_5():
    # Test with three clauses, with multiple simplifications possible
    clauses = [{"(on table block1)"}, {"(on table block2)"}, {"(not (on table block1))", "(not (on table block2))"}]
    assert minimize_dnf_clauses(clauses) == clauses


def test_minimize_dnf_clauses_no_minimization_needed():
    clauses = [{'(on ?x ?y)'}]
    minimized = minimize_dnf_clauses(clauses)
    assert minimized == clauses


def test_minimize_dnf_clauses_contain_unit_clauses_only():
    clauses = [{'(on ?x ?y)'}, {'(above ?x ?y)'}, {'(near ?x ?y)'}]
    minimized = minimize_dnf_clauses(clauses)
    assert minimized == clauses


def test_minimize_dnf_clauses_returns_empty_list():
    clauses = [{'(on ?x ?y)'}, {'(not (on ?x ?y))'}]
    assert minimize_dnf_clauses(clauses) == []


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
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 3)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}

    expected_superset_dependencies = [{"a", "b", "c"}, {"a", "b"}, {"a", "c"}, {"a"}]
    superset_dependencies = dependency_set._extract_superset_dependencies("a", {"a"})
    assert len(superset_dependencies) == 4
    for expected_superset_dependency in expected_superset_dependencies:
        assert expected_superset_dependency in superset_dependencies


def test_initialize_dependencies_with_real_domain_predicates_initialize_both_negative_and_positive_predicates_as_keys(
        woodworking_predicates: List[Predicate]):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(woodworking_predicates)
    assert len(dependency_set.possible_antecedents) == len(woodworking_predicates)


def test_initialize_dependencies_with_real_domain_predicates_creates_correct_set_of_literals_in_the_values(
        woodworking_predicates: List[Predicate]):
    """Test the initialization of the dependency set with real domain predicates."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    # C(28, 2) = 28! / (2! * (28-2)!) = 28! / (2! * 26!) = (28 * 27) / (1 * 2) = 378 sets.
    # The number of sets of size 1 that can be created from 28 objects is 28.
    assert len(dependency_set.possible_antecedents[
                   woodworking_predicates[0].untyped_representation]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS


def test_remove_dependencies_removes_correct_literals_on_simple_case_with_no_superset_literals():
    """Test the removal of a dependency from the dependency set when creating a simple scenario."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    antecedents = [{"a"}, {"b"}, {"c"}, {"a", "b"}, {"a", "c"}, {"b", "c"}]
    init_antecedents_length = len(antecedents)
    dependency_set.possible_antecedents = {"a": antecedents}

    tested_literal = "a"
    literals_to_remove = {"a", "b"}
    dependency_set.remove_dependencies(tested_literal, literals_to_remove)

    expected_removed_literals = create_antecedents_combination(literals_to_remove, 2)
    assert len(dependency_set.possible_antecedents[tested_literal]) == init_antecedents_length - len(
        expected_removed_literals)


def test_remove_dependencies_removes_correct_literals_on_simple_case_with_superset_literals():
    """Test the removal of a dependency from the dependency set when creating a simple scenario."""
    dependency_set = DependencySet(max_size_antecedents=3, action_signature={}, domain_constants={})
    antecedents = create_antecedents_combination({"a", "b", "c"}, 3)
    dependency_set.possible_antecedents = {"a": antecedents}

    tested_literal = "a"
    literals_to_remove = {"a", "b"}
    dependency_set.remove_dependencies(tested_literal, literals_to_remove, include_supersets=True)
    assert len(dependency_set.possible_antecedents[tested_literal]) == 1


def test_remove_dependencies_removed_correct_set_of_literals_and_all_subsets(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set with real domain predicates used as input."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    dependency_set.initialize_dependencies(set(woodworking_predicates))
    tested_predicate = "(available ?obj)"
    predicates_to_remove = {"(is-smooth ?surface)", "(has-colour ?agent ?colour)"}
    expected_removed_literals = create_antecedents_combination(predicates_to_remove, 2)
    dependency_set.remove_dependencies(tested_predicate, predicates_to_remove)
    assert len(dependency_set.possible_antecedents[tested_predicate]) == TOTAL_NUMBER_OF_WOODWORKING_COMBINATIONS - len(
        expected_removed_literals)


def test_remove_preconditions_literals_correctly_removed_preconditions_from_the_dependency_set(
        woodworking_predicates: List[Predicate]):
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


def test_is_safe_literal_returns_literal_safe_if_contains_zero_items(woodworking_predicates: List[Predicate]):
    """Test the check if a literal is safe when the literal contains zero items."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    tested_predicate = "(available ?obj)"
    dependency_set.possible_antecedents[tested_predicate] = []
    assert dependency_set.is_safe_literal(tested_predicate)


def test_is_safe_literal_returns_literal_safe_if_contains_one_item(woodworking_predicates: List[Predicate]):
    """Test the removal of a dependency from the dependency set."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    tested_predicate = "(available ?obj)"
    dependency_set.possible_antecedents[tested_predicate] = [{"(available ?obj)"}]
    assert dependency_set.is_safe_literal(tested_predicate)


def test_is_conditional_effect_returns_true_when_the_the_literal_contains_a_single_antecedent_different_from_the_literal():
    """Check if a literal is a conditional effect when it contains one antecedent different from the literal."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    tested_predicate = "(available ?obj)"
    dependency_set.possible_antecedents[tested_predicate] = [{"(not (available ?obj))"}]
    assert dependency_set.is_safe_conditional_effect(tested_predicate)


def test_is_conditional_effect_returns_false_when_there_are_no_antecedents_for_the_literal():
    """Check if a literal is a conditional effect when it contains no antecedents."""
    dependency_set = DependencySet(max_size_antecedents=2, action_signature={}, domain_constants={})
    tested_predicate = "(available ?obj)"
    dependency_set.possible_antecedents[tested_predicate] = []
    assert not dependency_set.is_safe_conditional_effect(tested_predicate)


def test_extract_safe_antecedents_extracts_correct_set_of_literals():
    """Test that extract_safe_conditionals returns the correct set of literals."""
    dependency_set = DependencySet(max_size_antecedents=3, action_signature={}, domain_constants={})
    antecedents = {"a", "b", "(not c)"}
    dependency_set.possible_antecedents = {"a": [antecedents]}
    result_antecedents = dependency_set.extract_safe_antecedents("a")
    assert result_antecedents == antecedents


def test_extract_safe_antecedents_empty_set_when_there_are_no_antecedents():
    """Test that extract_safe_conditionals returns the correct set of literals."""
    dependency_set = DependencySet(max_size_antecedents=3, action_signature={}, domain_constants={})
    dependency_set.possible_antecedents = {"a": []}
    result_antecedents = dependency_set.extract_safe_antecedents("a")
    assert len(result_antecedents) == 0


def test_construct_restrictive_preconditions_returns_none_if_result_predicate_is_in_preconditions(
        do_saw_predicates: Set[Predicate], woodworking_domain: Domain):
    test_signature = woodworking_domain.actions["do-saw-small"].signature
    dependency_set = DependencySet(max_size_antecedents=2, action_signature=test_signature,
                                   domain_constants=woodworking_domain.constants)
    dependency_set.initialize_dependencies(do_saw_predicates)
    tested_literal = "(available ?p)"

    restrictive_precondition = dependency_set.construct_restrictive_preconditions(
        preconditions={tested_literal}, literal=tested_literal)
    assert restrictive_precondition is None


def test_construct_restrictive_preconditions_a_precondition_object_with_literals_that_is_not_none(
        do_saw_predicates: Set[Predicate], woodworking_domain: Domain):
    test_signature = woodworking_domain.actions["do-saw-small"].signature
    dependency_set = DependencySet(max_size_antecedents=2, action_signature=test_signature,
                                   domain_constants=woodworking_domain.constants)
    dependency_set.initialize_dependencies(do_saw_predicates)
    tested_literal = "(available ?p)"

    restrictive_precondition = dependency_set.construct_restrictive_preconditions(
        preconditions=set(), literal=tested_literal)
    assert restrictive_precondition is not None


def test_construct_restrictive_preconditions_returns_none_if_the_negated_antecedents_are_the_same_as_the_preconditions():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 1)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"(not (a ))", "(not (b ))", "(not (c ))"}
    tested_literal = "(a )"

    conditions = dependency_set.construct_restrictive_preconditions(preconditions, tested_literal)
    print(str(conditions))
    assert not conditions


def test_construct_restrictive_preconditions_creates_conditions_that_do_not_include_precondition_literal_if_is_not_effect_and_negated_effect_is_precondition():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 1)
    dependency_set.possible_antecedents = {literal: possible_literals_combinations for literal in test_literals}
    preconditions = {"(not (a ))"}
    tested_literal = "(a )"

    conditions = dependency_set.construct_restrictive_preconditions(preconditions, tested_literal)
    restrictive_conditions = [cond.untyped_representation for _, cond in conditions if isinstance(cond, Predicate)]
    assert sorted(restrictive_conditions) == ["(not (b ))", "(not (c ))"]


def test_construct_restrictive_preconditions_creates_conditions_that_do_not_include_precondition_literal_if_is_effect_and_negated_effect_is_precondition():
    test_literals = ["(a )", "(b )", "(c )"]
    dependency_set = DependencySet(max_size_antecedents=1, action_signature={}, domain_constants={})
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 1)
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
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 2)
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
    possible_literals_combinations = create_antecedents_combination(set(test_literals), 3)
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

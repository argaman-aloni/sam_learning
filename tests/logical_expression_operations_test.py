"""Module test for the logical expression operations module."""

from pytest import raises

from sam_learning.core.logical_expression_operations import create_dnf_combinations, minimize_cnf_clauses, \
    minimize_dnf_clauses, create_cnf_combination


def test_create_dnf_combinations_with_max_antecedents_size_two_and_3_literals_returns_correct_combinations():
    antecedents = ["a", "b", "c"]
    expected_antecedents_combinations = [[["a"], ["b"]], [["a"], ["c"]], [["b"], ["c"]]]
    antecedents_combinations = create_dnf_combinations(antecedents, 2)
    assert len(antecedents_combinations) == len(expected_antecedents_combinations)
    for expected_combination in expected_antecedents_combinations:
        assert expected_combination in antecedents_combinations


def test_create_dnf_combinations_returns_objects_sorted_by_length_and_then_lexicographically():
    antecedents = ["a", "b", "c"]
    expected_antecedents_combinations = sorted([[["a"], ["b"]], [["a"], ["c"]], [["b"], ["c"]],
                                                [["a"], ["b", "c"]], [["a", "b"], ["c"]], [["a", "c"], ["b"]],
                                                [["a"], ["b"], ["c"]]],
                                               key=lambda x: sum([len(item) for item in x]))
    antecedents_combinations = create_dnf_combinations(antecedents, 3)
    assert len(antecedents_combinations) == len(expected_antecedents_combinations)
    assert antecedents_combinations == expected_antecedents_combinations


def test_minimize_cnf_clauses_empty_clauses():
    clauses = []
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == []


def test_minimize_cnf_clauses_single_clause():
    """Test the minimization of a single clause.

    The clause should not be changed since it represents the expression (p V q V r).
    """
    clauses = [{'(p)', '(q)', '(r)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)', '(q)', '(r)'}]


def test_minimize_cnf_clauses_single_unit_clause():
    """Test the minimization of a single unit clause.

    The clause should not be changed since it represents the expression (p).
    """
    clauses = [{'(p)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)'}]


def test_minimize_cnf_clauses_single_unit_clause_and_single_non_unit_clause():
    """Test the minimization of a single unit clause and a single non-unit clause.

    The combined expression is p ^ (p V q). That means p should be true and q should be removed (since irrelevant).
    """
    clauses = [{'(p)'}, {'(p)', '(q)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)'}]


def test_minimize_cnf_clauses_multiple_non_unit_clauses():
    """Test the minimization of multiple non-unit clauses.

    The combined expression is (p V q) ^ (p V r) ^ (q V r). This should remain the same.
    """
    clauses = [{'(p)', '(q)'}, {'(p)', '(r)'}, {'(q)', '(r)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert minimized_clauses == [{'(p)', '(q)'}, {'(p)', '(r)'}, {'(q)', '(r)'}]


def test_minimize_cnf_clauses_unit_clause_and_complementary_literals():
    """Test the minimization of a unit clause and complementary literals.

    The minimized expression is (p) ^ (r) ^ (q) ^ (s).
    """
    clauses = [{'(p)'}, {'(not (q))', '(r)'}, {'(not (p))', '(q)'}, {'(s)'}]
    minimized_clauses = minimize_cnf_clauses(clauses)
    assert len(minimized_clauses) == 4
    assert sum([len(item) for item in minimized_clauses]) == 5
    expected_clauses = [{'(p)'}, {'(not (q))', '(r)'}, {'(q)'}, {'(s)'}]
    for clause in expected_clauses:
        assert clause in minimized_clauses

    # now making sure that another iteration changes the clauses to the most reduced form
    terminal_form = minimize_cnf_clauses(minimized_clauses)
    assert len(terminal_form) == 4
    assert sum([len(item) for item in terminal_form]) == 4
    expected_clauses = [{'(p)'}, {'(r)'}, {'(q)'}, {'(s)'}]
    for clause in expected_clauses:
        assert clause in minimized_clauses


def test_minimize_cnf_clauses_non_unit_clause_and_complementary_literals():
    """Test the minimization of a non-unit clause and complementary literals.

    The expression is unsatisfiable.
    """
    clauses = [{'(p)', '(q)'}, {'(not (p))'}, {'(r)'}, {'(not (q))'}, {'(s)'}, {'(t)'}]
    with raises(ValueError):
        minimize_cnf_clauses(clauses)


def test_minimize_cnf_clauses_multiple_complementary_literals():
    """Test the minimization of multiple complementary literals.

    The expression is unsatisfiable.
    """
    clauses = [{'(not (p))'}, {'(not q)', 'r'}, {'(not (r))', 'p'}, {'(p)'}]
    with raises(ValueError):
        minimize_cnf_clauses(clauses)


def test_minimize_cnf_clauses_assumptions():
    """Test the minimization of clauses with assumptions, that is literals that must be true.

    The expression is (p V ~q V r) ^ (~p V s) ^ (p V r V t) ^ (~t V u V v).
    The assumptions are p == T, t == T, v == T.
    The minimized expression is (s).
    """
    clauses = [{'(p)', '(not (q))', '(r)'}, {'(not (p))', '(s)'}, {'(q)', '(r)', '(t)'}, {'(not (t))', '(u)', '(v)'}]
    assumptions = {'(p)', '(t)', '(v)'}
    minimized_clauses = minimize_cnf_clauses(clauses, assumptions)
    assert minimized_clauses == [{'(s)'}]


def test_minimize_dnf_clauses_pddl_format_1():
    # Test with one clause, no simplification possible
    clauses = [{"(on table block1)"}]
    expected_minimized_clauses = [{"(on table block1)"}]
    assert minimize_dnf_clauses([[["(a)"], ["(b)"]], [["(not (a))"], ["(c)"]], [["(b)"], ["(not (c))"]],
                                 [["(a)"], ["(b)", "(not (c))"]], [["(not (a))", "(b)"], ["(c)"]], [["(a)", "(c)"], ["(not (b))"]],
                                 [["(a)"], ["(not (b))"], ["(c)"]]], assumptions={"(a)"}) == expected_minimized_clauses


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
    antecedents_combinations = create_cnf_combination(antecedents, 1)
    assert len(antecedents_combinations) == 3
    for expected_combination in expected_antecedents_combinations:
        assert expected_combination in antecedents_combinations


def test_create_antecedents_combination_with_max_size_2():
    """Test the creation of antecedents combinations with max size 2."""
    antecedents = {"a", "b", "c"}
    expected_antecedents_combinations = [{"a"}, {"b"}, {"c"}, {"a", "b"}, {"a", "c"}, {"b", "c"}]
    antecedents_combinations = create_cnf_combination(antecedents, 2)
    assert len(antecedents_combinations) == 6
    for expected_combination in expected_antecedents_combinations:
        assert expected_combination in antecedents_combinations

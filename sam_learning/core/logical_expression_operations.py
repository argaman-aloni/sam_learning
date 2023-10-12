"""Module defining logical expression operations."""
import itertools
import re
from typing import Set, List, Optional
from lark import Lark, Transformer

from sympy import Symbol, And, Or, simplify_logic, true

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1
LITERAL_REGEX = r"(\([\?w+[\-?_?\w]*\s?[?\w+\s?]*\))"

cnf_grammar = """
    start: and_expr  
    or_expr: atom | not_expr | or_expr "|" or_expr
    and_expr: atom | not_expr | and_expr "&" and_expr
    not_expr: "~" atom
    atom: PREDICATE

    PREDICATE: /(\([\?w+[\-?_?\w]*\s?[?\w+\s?]*\))/

    %import common.CNAME -> NAME
    %import common.WS_INLINE
    %ignore WS_INLINE
"""


# Define a transformer to build the AST
class LogicalExpressionTransformer(Transformer):
    def start(self, items):
        return items[0]

    def or_expr(self, items):
        print(items)
        if len(items) == 1:
            return items[0]
        else:
            left, right = items
            return f"(or {left} {right})"

    def not_expr(self, items):
        print(items)
        return f"(not {items[0]})"

    def and_expr(self, items):
        if len(items) == 1:
            return items[0]
        else:
            left, right = items
            return f"(and {left} {right})"

    def atom(self, items):
        print(items[0])
        return str(items[0])


def _flip_single_predicate(predicate: str) -> str:
    """Flips the sign of the given predicate.

    :param predicate: the predicate to flip.
    :return: the flipped predicate.
    """
    if predicate.startswith(NOT_PREFIX):
        return predicate[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]

    return f"(not {predicate})"


def create_cnf_combination(input_literals: Set[str], max_antecedents_size: int,
                           exclude_literals: Optional[Set[str]] = None) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param input_literals: the list of input_literals.
    :param max_antecedents_size: the maximal size of the antecedents' combination.
    :param exclude_literals: the literals to exclude from the antecedents combinations.
    :return: all possible subsets of the antecedents up to the given size.
    """
    cnf_combinations = []
    antecedents_to_use = input_literals - exclude_literals if exclude_literals is not None else input_literals
    for subset_size in range(1, max_antecedents_size + 1):
        possible_combinations = [set(combination) for combination in itertools.combinations(
            antecedents_to_use, subset_size)]
        for combination in possible_combinations:
            cnf_combinations.append(combination)

    return cnf_combinations


def create_dnf_combinations(antecedents: Set[str], max_antecedents_size: int) -> List[
    List[List[str]]]:
    """Creates all the combinations of disjunctive antecedents.

    Note: This is a nested approach since the disjunctions can have internal conjunctions.

    Example: (a or (b and c)) or (d and e)) is a valid disjunction of antecedents.

    :param antecedents: the antecedents to create the combinations from.
    :param max_antecedents_size: the maximal number of antecedents in the combination.
    :return: list containing all possible disjunctions of the antecedents.
    """
    # Assuming the maximal number of antecedents is higher than 1
    partial_combinations = [sorted(list(item)) for item in create_cnf_combination(
        input_literals=antecedents, max_antecedents_size=max_antecedents_size)]
    all_combinations = []
    for i in range(2, max_antecedents_size + 1):
        all_combinations += [sorted(list(combination)) for combination in itertools.combinations(
            partial_combinations, i)]

    filtered_combinations = []
    for item_set in all_combinations:
        if sum([len(item) for item in item_set]) > max_antecedents_size or \
                sum([len(item) for item in item_set]) != len({i for item in item_set for i in item}):
            continue

        filtered_combinations.append(item_set)

    return sorted(filtered_combinations, key=lambda x: (sum([len(item) for item in x]), len(x),
                                                        "".join([i for item in x for i in item])))


def check_complementary_literals(clause: Set[str]) -> bool:
    """Checks if any two literals in the clause are complementary.

    :param clause: the clause to check.
    :return: True if any two literals in the clause are complementary, False otherwise.
    """
    regex = re.compile(LITERAL_REGEX)
    for first_literal, second_literal in itertools.combinations(clause, 2):
        first_extracted_literal = regex.search(first_literal).group()
        second_extracted_literal = regex.search(second_literal).group()
        if (first_extracted_literal == second_extracted_literal and
                len([literal for literal in [first_literal, second_literal] if literal.startswith(NOT_PREFIX)]) == 1):
            # The literals are the same and one of them is the negation of the other, so they are complementary
            return True

    return False


def minimize_cnf_clauses(clauses: List[Set[str]], assumptions: Set[str] = None) -> List[Set[str]]:
    """Minimizes the CNF clauses based on unit clauses and complementary literals.

    Note:
        CNFs are clauses in the form of (a or b or c) and (d or e or f) and (g or h or i) and ...

    :param clauses: the CNF clauses to minimize.
    :param assumptions: the assumptions to use for the minimization.
    :return: the minimized CNF clauses.
    """
    used_assumptions = assumptions or set()
    minimized_clauses = [clause for clause in clauses.copy() if len(clause) == 1 if
                         not clause.intersection(used_assumptions)]
    unit_clauses = {literal for clause in minimized_clauses for literal in clause}
    if check_complementary_literals(unit_clauses):
        raise ValueError("The unit clauses are contradicting one another!")

    used_assumptions.update(unit_clauses)
    non_unit_clauses = [clause for clause in clauses if len(clause) > 1]
    for clause in non_unit_clauses:
        # Checking if there are complementary literals in the clause - if so, the clause is always true
        if check_complementary_literals(clause) or any([assumption in clause for assumption in used_assumptions]):
            continue

        for assumption in used_assumptions:
            negated_assumption = f"{NOT_PREFIX} {assumption})" if not assumption.startswith(NOT_PREFIX) else \
                assumption[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]
            if check_complementary_literals(clause.union({assumption})):
                clause.remove(negated_assumption)

        # There are no assumptions in the clause
        if len(clause) == 0:
            # if reached here then the clause and one of the assumptions result in a contradiction
            raise ValueError("The clauses are contradicting the assumptions!")

        if len(clause) > 0:
            minimized_clauses.append(clause)

    return minimized_clauses


def negate_and_convert_to_cnf(clauses: List[List[List[str]]]) -> List[Set[str]]:
    """Negates DNF expressions and converts them to CNF.

    :param clauses: the DNF clauses to negate and converted to CNF.
    """
    cnf_clauses = []
    for clause in clauses:
        for conjunction in clause:
            cnf_clauses.append({_flip_single_predicate(predicate) for predicate in conjunction})

    return cnf_clauses


def minimize_dnf_clauses(
        expressions_list: List[List[List[str]]], assumptions: Set[str] = frozenset()) -> str:
    """Minimizes the DNF clauses based on unit clauses and complementary literals.

    Note:
        DNF clause are in the form of (a and b and c) or (d and e and f) or (g and h and i) or ...

    :param expressions_list: the DNF clauses to minimize.
    :param assumptions: the assumptions to use for the minimization.
    :return: a PDDL string representing the minimized DNF clauses.
    """
    symbols = {}
    dnf_compiled_expressions = []
    for dnf_expression in expressions_list:
        single_dnf_expression = []
        for cnf_expression in dnf_expression:
            cnf_compiled_expression = []
            for literal in cnf_expression:
                if literal in assumptions:
                    cnf_compiled_expression.append(true)
                    continue

                normalized_literal = literal if not literal.startswith(NOT_PREFIX) else \
                    literal[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]
                if normalized_literal not in symbols:
                    symbols[normalized_literal] = Symbol(normalized_literal)

                current_symbol = symbols[normalized_literal] if not literal.startswith(NOT_PREFIX) else \
                    ~symbols[normalized_literal]
                cnf_compiled_expression.append(current_symbol)

            single_dnf_expression.append(And(*cnf_compiled_expression))

        dnf_compiled_expressions.append(Or(*single_dnf_expression))

    sympy_expression = And(*dnf_compiled_expressions)

    parser = Lark(cnf_grammar, parser='lalr', transformer=LogicalExpressionTransformer())
    return str(parser.parse(str(simplify_logic(sympy_expression, form='cnf'))))

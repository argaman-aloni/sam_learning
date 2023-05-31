"""Module representing the dependency set of an action."""
import itertools
import logging
from typing import Set, Dict, List, Optional, Union

from pddl_plus_parser.models import Predicate, Precondition, SignatureType, PDDLConstant

from sam_learning.core.discrete_utilities import extract_predicate_data

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1


def check_complementary_literals(clause: Set[str]) -> bool:
    """Checks if any two literals in the clause are complementary.

    :param clause: the clause to check.
    :return: True if any two literals in the clause are complementary, False otherwise.
    """
    for first_literal, second_literal in itertools.combinations(clause, 2):
        if first_literal.startswith(NOT_PREFIX) and not second_literal.startswith(NOT_PREFIX) and \
                first_literal[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX] == second_literal:
            # Complementary literals
            return True

        if not first_literal.startswith(NOT_PREFIX) and second_literal.startswith(NOT_PREFIX) and \
                first_literal == second_literal[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]:
            # Complementary literals
            return True

    return False


def minimize_cnf_clauses(clauses: List[Set[str]], assumptions: Set[str] = None) -> List[Set[str]]:
    """Minimizes the CNF clauses based on unit clauses and complementary literals.

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
            if assumption.startswith(NOT_PREFIX) and check_complementary_literals(clause.union({assumption})):
                # We can remove the complementary literal
                clause.remove(assumption[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX])

            elif not assumption.startswith(NOT_PREFIX) and check_complementary_literals(clause.union({assumption})):
                # We can remove the complementary literal
                clause.remove(f"{NOT_PREFIX} {assumption})")

        # There are no assumptions in the clause
        if len(clause) > 0:
            minimized_clauses.append(clause)

    return minimized_clauses


def minimize_dnf_clauses(clauses: List[Set[str]], assumptions: Set[str] = frozenset()) -> List[Set[str]]:
    """Minimizes the DNF clauses based on unit clauses and complementary literals.

    :param clauses: the DNF clauses to minimize.
    :param assumptions: the assumptions to use for the minimization.
    :return: the minimized DNF clauses.
    """
    if any([clause.issubset(assumptions) for clause in clauses]):
        # There is a clause that is a subset of the assumptions, so we don't need to add it
        return []

    unit_clauses = {next(iter(clause)) for clause in clauses if len(clause) == 1}
    if check_complementary_literals(unit_clauses):
        # since this is a DNF and wwe have a literal and its negation combined with OR - the clause is always true
        return []

    minimized_clauses = [clause for clause in clauses if len(clause) == 1]
    non_unit_clauses = [clause for clause in clauses if len(clause) > 1]
    found_contradiction = len(minimized_clauses) == 0
    for clause in non_unit_clauses:
        # Checking if there are complementary literals in the clause, if so, the clause is always false (don't add it)
        if check_complementary_literals(clause):
            continue

        # if there is a clause with an assumption negation, then this clause is always False
        is_contradicting_assumptions = False
        for assumption in assumptions:
            if assumption.startswith(NOT_PREFIX) and check_complementary_literals(clause.union({assumption})) or \
                    not assumption.startswith(NOT_PREFIX) and check_complementary_literals(clause.union({assumption})):
                is_contradicting_assumptions = True
                break

        if is_contradicting_assumptions:
            continue

        # There are no complementary literals in the clause and the assumptions are not contradicting the clause
        current_minimized_clause = clause.difference(assumptions)
        if len(current_minimized_clause) > 0:
            minimized_clauses.append(clause.difference(assumptions))
            found_contradiction = False

    if found_contradiction:
        raise ValueError("The clauses are contradicting themselves!")

    return minimized_clauses


def create_antecedents_combination(antecedents: Set[str], max_antecedents_size: int,
                                   exclude_literals: Optional[Set[str]] = None) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param antecedents: the list of antecedents that may be trigger for conditional effects.
    :param max_antecedents_size: the maximal size of the antecedents' combination.
    :param exclude_literals: the literals to exclude from the antecedents combinations.
    :return: all possible subsets of the antecedents up to the given size.
    """
    antecedents_combinations = []
    antecedents_to_use = antecedents - exclude_literals if exclude_literals is not None else antecedents
    for subset_size in range(1, max_antecedents_size + 1):
        possible_combinations = [set(combination) for combination in itertools.combinations(
            antecedents_to_use, subset_size)]
        for combination in possible_combinations:
            antecedents_combinations.append(combination)

    return antecedents_combinations


class DependencySet:
    """Class representing the dependency set of an action."""
    possible_antecedents: Dict[str, List[Set[str]]]
    max_antecedents: int
    action_signature: SignatureType
    domain_constants: Dict[str, PDDLConstant]
    logger: logging.Logger

    def __init__(self, max_size_antecedents: int, action_signature: SignatureType,
                 domain_constants: Dict[str, PDDLConstant]):
        self.possible_antecedents = {}
        self.max_antecedents = max_size_antecedents
        self.action_signature = action_signature
        self.domain_constants = domain_constants
        self.logger = logging.getLogger(__name__)

    def _extract_superset_dependencies(self, literal: str, dependencies_to_remove: Set[str]) -> List[Set[str]]:
        """Extracts the superset dependencies of the given literal.

        :param literal: the literal to check.
        :param dependencies_to_remove: the dependencies to remove.
        :return: all supersets that include a member of the dependencies to remove.
        """
        self.logger.debug(f"Extracting superset dependencies for literal {literal}")
        superset_dependencies = []
        for antecedents_conjunction in self.possible_antecedents[literal]:
            for dependency_to_remove in dependencies_to_remove:
                if dependency_to_remove in antecedents_conjunction:
                    superset_dependencies.append(antecedents_conjunction)

        return superset_dependencies

    @staticmethod
    def _flip_single_predicate(predicate: str) -> str:
        """Flips the sign of the given predicate.

        :param predicate: the predicate to flip.
        :return: the flipped predicate.
        """
        if predicate.startswith(NOT_PREFIX):
            return predicate[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]

        return f"(not {predicate})"

    def _extract_negated_minimized_antecedents(
            self, unsafe_antecedents: List[Set[str]], preconditions: Set[str]) -> Union[List[Set[str]], bool]:
        """

        :param unsafe_antecedents:
        :param preconditions:
        :return:
        """
        negated_conditions_statement = []
        for conjunction in unsafe_antecedents:
            negated_conditions_statement.append({self._flip_single_predicate(predicate) for predicate in conjunction})

        try:
            minimized_clauses = minimize_cnf_clauses(negated_conditions_statement, preconditions.copy())
            if len(minimized_clauses) == 0:
                return True

            return minimized_clauses

        except ValueError:
            self.logger.warning("The CNF unit clauses are contradicting one another!")
            return False

    def _post_process_preconditions(self, precondition: Precondition) -> Union[Precondition, bool]:
        """Post processes the preconditions to remove unnecessary conditions.

        :param precondition: the precondition to post process.
        :return: the post processed precondition.
        """
        self.logger.debug(f"Post processing precondition {precondition}")
        if precondition.binary_operator == "or":
            same_level_predicates = [operand for operand in precondition.operands if isinstance(operand, Predicate)]
            if check_complementary_literals({predicate.untyped_representation for predicate in same_level_predicates}):
                return True

        return precondition

    def _create_negated_antecedent_preconditions(
            self, negated_minimized_antecedents: Union[List[Set[str]], bool]) -> Union[Precondition, Predicate]:
        """Creates the preconditions for the negated antecedents.

        :param negated_minimized_antecedents: the literals set representing the negated antecedents.
        :return: the preconditions for the negated antecedents.
        """
        negated_antecedents_preconditions = Precondition("and")
        if len(negated_minimized_antecedents) == 1:
            disjunction = next(iter(negated_minimized_antecedents))
            if len(disjunction) == 1:
                return extract_predicate_data(action_signature=self.action_signature,
                                              predicate_str=disjunction.pop(),
                                              domain_constants=self.domain_constants)

            or_condition = Precondition("or")
            for antecedent_literal in disjunction:
                or_condition.add_condition(extract_predicate_data(action_signature=self.action_signature,
                                                                  predicate_str=antecedent_literal,
                                                                  domain_constants=self.domain_constants))
            return or_condition

        for disjunction in negated_minimized_antecedents:
            if len(disjunction) == 1:
                negated_antecedents_preconditions.add_condition(
                    extract_predicate_data(action_signature=self.action_signature,
                                           predicate_str=disjunction.pop(),
                                           domain_constants=self.domain_constants))
                continue

            or_condition = Precondition("or")
            for antecedent_literal in disjunction:
                or_condition.add_condition(extract_predicate_data(action_signature=self.action_signature,
                                                                  predicate_str=antecedent_literal,
                                                                  domain_constants=self.domain_constants))

            negated_antecedents_preconditions.add_condition(or_condition)

        return negated_antecedents_preconditions

    def _handle_contradicting_antecedents(
            self, is_effect: bool, literal: str,
            positive_minimized_antecedents: Set[str]) -> Union[Predicate, Precondition]:
        self.logger.debug(f"The negation of the antecedents for {literal} contains contradiction, "
                          f"so the rest of the DNF must be true")
        if not is_effect:
            lifted_result_predicate = extract_predicate_data(action_signature=self.action_signature,
                                                             predicate_str=literal,
                                                             domain_constants=self.domain_constants)
            return lifted_result_predicate
        # the literal was observed as an effect of the action, so either the positive clauses are
        # true or the literal itself is true
        if check_complementary_literals(positive_minimized_antecedents):
            lifted_result_predicate = extract_predicate_data(action_signature=self.action_signature,
                                                             predicate_str=literal,
                                                             domain_constants=self.domain_constants)
            return lifted_result_predicate
        positive_antecedents = Precondition("and")
        for antecedent_literal in positive_minimized_antecedents:
            positive_antecedents.add_condition(extract_predicate_data(action_signature=self.action_signature,
                                                                      predicate_str=antecedent_literal,
                                                                      domain_constants=self.domain_constants))
        resulting_preconditions = Precondition("or")
        lifted_result_predicate = extract_predicate_data(action_signature=self.action_signature,
                                                         predicate_str=literal,
                                                         domain_constants=self.domain_constants)
        resulting_preconditions.add_condition(lifted_result_predicate)
        resulting_preconditions.add_condition(positive_antecedents)
        return resulting_preconditions

    def _handle_effect_literal(
            self, lifted_result_predicate: Predicate,
            negated_antecedents_preconditions: Union[Precondition, Predicate],
            negated_result: str, positive_minimized_antecedents: Set[str],
            preconditions: Set[str]) -> Optional[Precondition]:
        """Handles the creation of the conditional's effect preconditions.

        :param lifted_result_predicate: the lifted result predicate.
        :param negated_antecedents_preconditions: the negated antecedents preconditions.
        :param negated_result: the negated result.
        :param positive_minimized_antecedents: the positive minimized antecedents.
        :param preconditions: the preconditions.
        :return: the conditional's effect preconditions or None if the antecedents return a tautology.
        """
        restrictive_precondition = Precondition("or")
        positive_antecedents = Precondition("and")
        for antecedent_literal in positive_minimized_antecedents:
            positive_antecedents.add_condition(extract_predicate_data(action_signature=self.action_signature,
                                                                      predicate_str=antecedent_literal,
                                                                      domain_constants=self.domain_constants))
        restrictive_precondition.add_condition(positive_antecedents)
        restrictive_precondition.add_condition(negated_antecedents_preconditions)
        if negated_result not in preconditions:
            restrictive_precondition.add_condition(lifted_result_predicate)

        final_precondition = self._post_process_preconditions(restrictive_precondition)
        if final_precondition is True:
            return None

        return restrictive_precondition

    def initialize_dependencies(self, lifted_literals: Set[Predicate],
                                antecedents: Optional[Set[Predicate]] = None) -> None:
        """Initialize the dependencies with positive and negative literals.

        :param lifted_literals: the lifted bounded literals matching the action.
        :param antecedents: in case that the antecedents may contain different literals from the results
            (e.g. in the case of universal effects).
        """
        literals_str = {literal.untyped_representation for literal in lifted_literals}
        for literal in literals_str:
            if antecedents is not None:
                antecedents_literals = {antecedent.untyped_representation for antecedent in antecedents}
                self.possible_antecedents[literal] = create_antecedents_combination(
                    antecedents_literals, self.max_antecedents)

            else:
                self.possible_antecedents[literal] = create_antecedents_combination(literals_str, self.max_antecedents)

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str], include_supersets: bool = False) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        :param include_supersets: whether to include supersets of the dependencies to remove.
        """
        self.logger.debug(f"Removing dependencies {literals_to_remove} for literal {literal}")
        dependencies_to_remove = create_antecedents_combination(literals_to_remove, self.max_antecedents)
        if include_supersets:
            dependencies_to_remove = self._extract_superset_dependencies(literal, literals_to_remove)

        for dependency in dependencies_to_remove:
            if dependency in self.possible_antecedents[literal]:
                self.possible_antecedents[literal].remove(dependency)

    def remove_preconditions_literals(self, preconditions_literals: Set[str]) -> None:
        """Removes the preconditions literals from the dependency set (from both the antecedents and the results).

        :param preconditions_literals: the preconditions of the action.
        """
        for literal in preconditions_literals:
            self.possible_antecedents.pop(literal, None)

        for literal in self.possible_antecedents:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

    def is_safe_literal(self, literal: str, preconditions_literals: Optional[Set[str]] = None) -> bool:
        """Determines whether the literal is safe in terms of number of antecedents.

        :param literal: the literal to check.
        :param preconditions_literals: the preconditions of the action.
        :return: True if the dependency set is safe, False otherwise.
        """
        if preconditions_literals is not None:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

        return len(self.possible_antecedents[literal]) <= 1

    def is_safe_conditional_effect(self, literal: str) -> bool:
        """Determines whether the literal is a conditional effect with safe number of antecedents.

        :param literal: the literal to check.
        :return: True if the dependency set is safe, False otherwise.
        """
        self.logger.info(f"Determining whether the literal {literal} is a conditional effect "
                         f"with safe number of antecedents")
        return len(self.possible_antecedents[literal]) == 1 and self.possible_antecedents[literal][0] != {literal}

    def is_possible_result(self, literal: str) -> bool:
        """Determines whether the literal is a possible result.

        :param literal: the literal to check.
        :return: True if the literal is a key in the dependency set, False otherwise.
        """
        return literal in self.possible_antecedents

    def extract_safe_antecedents(self, literal: str) -> Set[str]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the safe antecedents for the literals which is the result of the conditional effect.
        """
        self.logger.info("Extracting the tuple of the safe antecedents for the literal %s", literal)
        antecedents_copy = self.possible_antecedents[literal].copy()
        if len(antecedents_copy) == 0:
            return set()

        safe_antecedents = antecedents_copy.pop()
        return safe_antecedents

    def construct_restrictive_preconditions(
            self, preconditions: Set[str], literal: str,
            is_effect: bool = False) -> Optional[Union[Precondition, Predicate]]:
        """Extracts the safe conditional effects from the dependency set.

        :param preconditions: the preconditions of the action.
        :param literal: the literal to create restrictive preconditions for.
        :param is_effect: whether the literal is an effect.
        :return: the negative and positive conditions that need to be added or None.
        """
        if literal in preconditions:
            return None

        unsafe_antecedents = self.possible_antecedents[literal]
        self.logger.debug("Constructing restrictive preconditions from the unsafe antecedents for literal %s", literal)
        negated_minimized_antecedents = self._extract_negated_minimized_antecedents(unsafe_antecedents, preconditions)
        positive_minimized_antecedents = set()
        for disjunction in unsafe_antecedents:
            positive_minimized_antecedents.update(disjunction)

        if isinstance(negated_minimized_antecedents, bool) and negated_minimized_antecedents:
            self.logger.debug(f"The negation of the antecedents for {literal} is the same as the preconditions"
                              f" so it can never be triggered!")
            return None

        if isinstance(negated_minimized_antecedents, bool) and not negated_minimized_antecedents:
            return self._handle_contradicting_antecedents(is_effect, literal, positive_minimized_antecedents)

        # the negated antecedents are the minimized set of literals with no contradictions
        negated_antecedents_preconditions = self._create_negated_antecedent_preconditions(negated_minimized_antecedents)

        negated_result = f"{NOT_PREFIX} {literal})" if not literal.startswith(NOT_PREFIX) else \
            literal[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]
        lifted_result_predicate = extract_predicate_data(action_signature=self.action_signature,
                                                         predicate_str=literal,
                                                         domain_constants=self.domain_constants)

        if is_effect:
            return self._handle_effect_literal(lifted_result_predicate, negated_antecedents_preconditions,
                                               negated_result, positive_minimized_antecedents, preconditions)

        if negated_result in preconditions:
            return negated_antecedents_preconditions

        not_result_precondition = Precondition("or")
        not_result_precondition.add_condition(lifted_result_predicate)
        not_result_precondition.add_condition(negated_antecedents_preconditions)
        final_result = self._post_process_preconditions(not_result_precondition)
        if final_result is True:
            return None

        return final_result

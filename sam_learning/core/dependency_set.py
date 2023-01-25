"""Module representing the dependency set of an action."""
import itertools
import logging
from typing import Set, Dict, List, Tuple, Optional

from pddl_plus_parser.models import Predicate

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1


def create_antecedents_combination(antecedents: Set[str], max_antecedents_size: int,
                                   exclude_literals: Optional[Set[str]] = None) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param antecedents: the list of antecedents that may be trigger for conditional effects.
    :param max_antecedents_size: the maximal size of the antecedents' combination.
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
    dependencies: Dict[str, List[Set[str]]]
    max_size_antecedents: int
    logger: logging.Logger

    def __init__(self, max_size_antecedents: int):
        self.dependencies = {}
        self.max_size_antecedents = max_size_antecedents
        self.logger = logging.getLogger(__name__)

    def _extract_superset_dependencies(self, literal: str, dependencies_to_remove: List[Set[str]]) -> List[Set[str]]:
        """Extracts the superset dependencies of the given literal.

        :param literal: the literal to check.
        :param dependencies_to_remove: the dependencies to remove.
        :return: all supersets that include a member of the dependencies to remove.
        """
        self.logger.debug(f"Extracting superset dependencies for literal {literal}")
        superset_dependencies = []
        for dependency in self.dependencies[literal]:
            for dependency_to_remove in dependencies_to_remove:
                if dependency_to_remove.issubset(dependency):
                    superset_dependencies.append(dependency)

        return superset_dependencies

    @staticmethod
    def _create_negated_antecedents(antecedents: Set[str]) -> str:
        """Creates the negated antecedents.

        :param antecedents: the antecedents to negate.
        :return: the negated antecedents as a combined string.
        """
        negated_antecedents = []
        for antecedent in antecedents:
            antecedent_to_add = antecedent[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX] if \
                antecedent.startswith(NOT_PREFIX) else f"{NOT_PREFIX} {antecedent})"
            negated_antecedents.append(antecedent_to_add)

        if len(negated_antecedents) > 1:
            return f"(or {' '.join(negated_antecedents)})"

        return negated_antecedents[0]

    def initialize_dependencies(self, lifted_literals: Set[Predicate],
                                antecedents: Optional[Set[Predicate]] = None) -> None:
        """Initialize the dependencies with positive and negative literals.

        :param lifted_literals: the lifted bounded literals matching the action.
        :param antecedents: in case that the antecedents may contain different literals from the results
            (e.g. in the case of universal effects).
        """
        literals_str = {literal.untyped_representation for literal in lifted_literals}
        literals_str.update({f"(not {literal.untyped_representation})" for literal in lifted_literals})
        for literal in literals_str:
            if antecedents is not None:
                antecedents_literals = {antecedent.untyped_representation for antecedent in antecedents}
                self.dependencies[literal] = create_antecedents_combination(
                    antecedents_literals, self.max_size_antecedents)

            else:
                self.dependencies[literal] = create_antecedents_combination(literals_str, self.max_size_antecedents)

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str], include_supersets: bool = False) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        :param include_supersets: whether to include supersets of the dependencies to remove.
        """
        self.logger.info(f"Removing dependencies {literals_to_remove} for literal {literal}")
        dependencies_to_remove = create_antecedents_combination(literals_to_remove, self.max_size_antecedents)
        superset_dependencies = []
        if include_supersets:
            superset_dependencies = self._extract_superset_dependencies(literal, dependencies_to_remove)
            dependencies_to_remove.extend(superset_dependencies)

        for dependency in dependencies_to_remove:
            if dependency in self.dependencies[literal]:
                self.dependencies[literal].remove(dependency)

    def remove_preconditions_literals(self, preconditions_literals: Set[str]) -> None:
        """Removes the preconditions literals from the dependency set (from both the antecedents and the results).

        :param preconditions_literals: the preconditions of the action.
        """
        for literal in preconditions_literals:
            self.dependencies.pop(literal, None)

        for literal in self.dependencies:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

    def is_safe_literal(self, literal: str, preconditions_literals: Optional[Set[str]] = None) -> bool:
        """Determines whether the literal is safe in terms of number of antecedents.

        :param literal: the literal to check.
        :param preconditions_literals: the preconditions of the action.
        :return: True if the dependency set is safe, False otherwise.
        """
        if preconditions_literals is not None:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

        return len(self.dependencies[literal]) <= 1

    def is_conditional_effect(self, literal: str) -> bool:
        """Determines whether the literal is a conditional effect with safe number of antecedents.

        :param literal: the literal to check.
        :return: True if the dependency set is safe, False otherwise.
        """
        self.logger.info("Determining whether the literal %s is a conditional effect with safe number of antecedents",
                         literal)
        return len(self.dependencies[literal]) == 1 and self.dependencies[literal][0] != {literal}

    def is_possible_result(self, literal: str) -> bool:
        """Determines whether the literal is a possible result.

        :param literal: the literal to check.
        :return: True if the literal is a key in the dependency set, False otherwise.
        """
        return literal in self.dependencies

    def extract_safe_conditionals(self, literal: str) -> Tuple[Set[str], Set[str]]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the safe conditional effects.
        """
        self.logger.info("Extracting the tuple of the safe antecedents for the literal %s", literal)
        safe_conditionals = self.dependencies[literal].copy()
        safe_conditions = safe_conditionals.pop()
        positive_predicates = set()
        negative_predicates = set()
        for condition in safe_conditions:
            if condition.startswith("(not "):
                negative_predicates.add(f"{condition[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]}")

            else:
                positive_predicates.add(condition)

        return positive_predicates, negative_predicates

    def construct_restrictive_preconditions(
            self, preconditions: Set[str], literal: str, is_effect: bool = False) -> Optional[str]:
        """Extracts the safe conditional effects from the dependency set.

        :param preconditions: the preconditions of the action.
        :param literal: the literal to create restrictive preconditions for.
        :param is_effect: whether the literal is an effect.
        :return: the negative and positive conditions that need to be added or None.
        """
        self.logger.debug("Constructing restrictive preconditions from the unsafe antecedents for literal %s", literal)
        unsafe_antecedents = self.dependencies[literal]
        negated_conditions_statement = []
        positive_antecedents = set()
        for antecedent_conjunction in unsafe_antecedents:
            negated_conditions_statement.append(self._create_negated_antecedents(antecedent_conjunction))
            positive_antecedents.update(antecedent_conjunction)

        if set(negated_conditions_statement) == preconditions:
            self.logger.debug(f"The negation of the antecedents for {literal} is the same as the preconditions"
                              f" so it can never be triggered!")
            return None

        antecedent_statement = f"(and {' '.join(positive_antecedents)})"
        negated_result = f"{NOT_PREFIX} {literal})"
        if is_effect:
            return f"(or {antecedent_statement} (and {' '.join(negated_conditions_statement)}))" \
                if negated_result in preconditions \
                else f"(or {literal} (and {' '.join(negated_conditions_statement)}) {antecedent_statement})"

        return f"(and {' '.join(negated_conditions_statement)})" if negated_result in preconditions \
            else f"(or {literal} (and {' '.join(negated_conditions_statement)}))"

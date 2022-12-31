"""Module representing the dependency set of an action."""
import itertools
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional

from pddl_plus_parser.models import Predicate

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1


def create_antecedents_combination(antecedents: Set[str], max_antecedents_size: int,
                                   exclude_literals: Optional[Set[str]] = None) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param antecedents: the list of antecedents that may be trigger for conditional effects.
    :param max_antecedents_size: the maximal size of the antecedents combination.
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

    def __init__(self, max_size_antecedents: int):
        self.dependencies = {}
        self.max_size_antecedents = max_size_antecedents

    def initialize_dependencies(self, lifted_literals: Set[Predicate]) -> None:
        """Initialize the dependencies with positive and negative literals.

        :param lifted_literals: the lifted bounded literals matching the action.
        """
        literals_str = {literal.untyped_representation for literal in lifted_literals}
        literals_str.update({f"(not {literal.untyped_representation})" for literal in lifted_literals})
        for literal in literals_str:
            self.dependencies[literal] = create_antecedents_combination(
                literals_str, self.max_size_antecedents, exclude_literals={literal})

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str]) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        """
        dependencies_to_remove = create_antecedents_combination(literals_to_remove, self.max_size_antecedents)
        for dependency in dependencies_to_remove:
            if dependency in self.dependencies[literal]:
                self.dependencies[literal].remove(dependency)

    def remove_preconditions_literals(self, preconditions_literals: Set[str]) -> None:
        """Removes the preconditions literals from the dependency set.

        :param preconditions_literals: the preconditions of the action.
        """
        for literal in preconditions_literals:
            self.dependencies.pop(literal, None)

    def is_safe_conditional_effect(self, literal: str) -> bool:
        """Determines whether the literal is a conditional effect with safe number of antecedents.

        :param literal: the literal to check.
        :return: True if the dependency set is safe, False otherwise.
        """
        return len(self.dependencies[literal]) == 1 and self.dependencies[literal][0] != {literal}

    def is_safe_literal(self, literal: str, preconditions_literals: Optional[Set[str]] = None) -> bool:
        """Determines whether the literal is safe in terms of number of antecedents.

        :param literal: the literal to check.
        :param preconditions_literals: the preconditions of the action.
        :return: True if the dependency set is safe, False otherwise.
        """
        if preconditions_literals is not None:
            self.remove_dependencies(literal, preconditions_literals)

        return len(self.dependencies[literal]) <= 1

    def is_safe(self, preconditions_literals: Set[str]) -> bool:
        """Determines whether the dependency set of an action is safe for all possible lifted literals.

        :param preconditions_literals: the preconditions of the action.
        :return: True if the entire dependency set is safe, False otherwise.
        """
        for literal in self.dependencies:
            if not self.is_safe_literal(literal, preconditions_literals):
                return False

        return True

    def extract_safe_conditionals(self, literal: str) -> Tuple[Set[str], Set[str]]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the safe conditional effects.
        """
        safe_conditionals = self.dependencies[literal].copy()
        safe_conditions = safe_conditionals.pop()
        positive_predicates = set()
        negative_predicates = set()
        for condition in safe_conditions:
            if condition.startswith("(not "):
                negative_predicates.add(f"{condition[AFTER_NOT_PREFIX_INDEX:-RIGHT_BRACKET_INDEX]}")

            else:
                positive_predicates.add(condition)

        return positive_predicates, negative_predicates

    def _extract_unsafe_antecedents(self) -> Dict[str, Set[str]]:
        """Extracts the safe antecedents from the dependency set.

        :return: the safe antecedents.
        """
        safe_conditions = defaultdict(set)
        # assuming that at this point the precondition is already removed from the dependency set
        for literal in self.dependencies:
            if not self.is_safe_literal(literal):
                unsafe_antecedents = set()
                for antecedents in self.dependencies[literal]:
                    unsafe_antecedents.update(antecedents)

                safe_conditions[literal] = unsafe_antecedents

        return safe_conditions

    def _construct_restrictive_preconditions(self, unsafe_antecedents: Dict[str, Set[str]]) -> List[str]:
        """Constructs the restrictive preconditions from the unsafe antecedents.

        :param unsafe_antecedents: the unsafe antecedents.
        :return: the restrictive preconditions.
        """
        restrictive_preconditions = []
        for effect, antecedents in unsafe_antecedents.items():
            negated_antecedents = []
            for antecedent in antecedents:
                antecedent_to_add = antecedent[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX] if antecedent.startswith(
                    "(not ") else f"(not {antecedent})"
                if antecedent_to_add != effect:
                    negated_antecedents.append(antecedent_to_add)

            if len(negated_antecedents) == 0:
                continue

            restrictive_preconditions.append(f"(or {effect} (and {' '.join(negated_antecedents)}))")

        return restrictive_preconditions

    def extract_restrictive_conditions(self) -> List[str]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the negative and positive conditions that need to be added.
        """
        unsafe_conditions = self._extract_unsafe_antecedents()
        return self._construct_restrictive_preconditions(unsafe_conditions)

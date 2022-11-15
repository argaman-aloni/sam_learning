"""Module representing the dependency set of an action."""
import itertools
from typing import Set, Dict, List

from pddl_plus_parser.models import Predicate


def create_antecedents_combination(antecedents: Set[str], max_antecedents_size) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param antecedents: the list of antecedents that may be trigger for conditional effects.
    :param max_antecedents_size: the maximal size of the antecedents combination.
    :return: all possible subsets of the antecedents up to the given size.
    """
    antecedents_combinations = []
    for subset_size in range(1, max_antecedents_size + 1):
        possible_combinations = [set(combination) for combination in itertools.combinations(antecedents, subset_size)]
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
        antecedents_combinations = create_antecedents_combination(literals_str, self.max_size_antecedents)
        self.dependencies = {literal: antecedents_combinations.copy() for literal in literals_str}

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str]) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        """
        dependencies_to_remove = create_antecedents_combination(literals_to_remove, self.max_size_antecedents)
        for dependency in dependencies_to_remove:
            if dependency in self.dependencies[literal]:
                self.dependencies[literal].remove(dependency)

    def is_conditional_effect(self, literal: str, preconditions_literals: Set[str]) -> bool:
        """Determines whether the literal is a conditional effect with safe number of antecedents.

        :param literal: the literal to check.
        :param preconditions_literals: the preconditions of the action.
        :return: True if the dependency set is safe, False otherwise.
        """
        self.remove_dependencies(literal, preconditions_literals)
        return len(self.dependencies[literal]) == 1 and self.dependencies[literal][0] != {literal}

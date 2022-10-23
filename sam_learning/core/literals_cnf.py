"""Module representing a data strudture that manages the matching of lifted predicates to their possible executing actions."""
from typing import List, Dict, NoReturn, Set, Tuple

from pddl_plus_parser.models import Predicate


class LiteralCNF:
    """Class that manages the matching of lifted predicates to their possible executing actions."""

    possible_lifted_effects: List[List[Tuple[str, str]]]
    not_effects: Dict[str, Set[str]]

    def __init__(self, action_names: List[str]):
        """Initialize the class."""
        self.possible_lifted_effects = []
        self.not_effects = {action_name: set() for action_name in action_names}

    def add_not_effect(self, action_name: str, predicate: Predicate) -> NoReturn:
        """

        :param action_name:
        :param predicate:
        :return:
        """
        self.not_effects[action_name].add(predicate.untyped_representation)
        for possible_joint_effect in self.possible_lifted_effects:
            if (action_name, predicate.untyped_representation) in possible_joint_effect:
                possible_joint_effect.remove((action_name, predicate.untyped_representation))

    def add_possible_effect(self, possible_joint_effect: List[Tuple[str, str]]) -> NoReturn:
        """Add a possible joint effect to the list of possible effects.

        :param possible_joint_effect: the possible joint effect.
        """
        filtered_joint_effect = []
        for (action_name, lifted_predicate) in possible_joint_effect:
            if lifted_predicate in self.not_effects[action_name]:
                continue

            filtered_joint_effect.append((action_name, lifted_predicate))
        self.possible_lifted_effects.append(filtered_joint_effect)

    def add_positive_predicates(self, predicate: Predicate, bounded_lifted_predicates: List[Predicate]) -> NoReturn:
        """Add a positive predicates to the list of predicates.

        :param predicate: the domain predicate.
        :param bounded_lifted_predicates: the bounded action lifted predicates.
        """
        self.positive_predicates.append({predicate: bounded_lifted_predicates})

    def add_negative_predicate(self, predicate: Predicate, bounded_lifted_predicates: List[Predicate]) -> NoReturn:
        """Add a negative predicates to the list of predicates.

        :param predicate: the domain predicate.
        :param bounded_lifted_predicates: the bounded action lifted predicates.
        """
        self.negative_predicates.append({predicate: bounded_lifted_predicates})


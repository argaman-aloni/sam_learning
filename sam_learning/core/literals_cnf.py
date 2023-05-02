"""Represents a data structure that manages the matching of lifted predicates to their possible executing actions."""
from typing import List, Dict, Set, Tuple

from pddl_plus_parser.models import Predicate


class LiteralCNF:
    """Class that manages the matching of lifted predicates to their possible executing actions."""

    possible_lifted_effects: List[List[Tuple[str, str]]]
    not_effects: Dict[str, Set[str]]

    def __init__(self, action_names: List[str]):
        """Initialize the class."""
        self.possible_lifted_effects = []
        self.not_effects = {action_name: set() for action_name in action_names}

    def add_not_effect(self, action_name: str, predicate: Predicate) -> None:
        """Adds a predicate that was determined to NOT be an effect of the action.

        :param action_name: the name of the action in which the predicate is not part of its effects.
        :param predicate: the predicate that is not the action's effect.
        """
        redundant_items_indexes = []
        self.not_effects[action_name].add(predicate.untyped_representation)
        for index, possible_joint_effect in enumerate(self.possible_lifted_effects):
            if (action_name, predicate.untyped_representation) in possible_joint_effect:
                possible_joint_effect.remove((action_name, predicate.untyped_representation))
                if len(possible_joint_effect) == 0:
                    redundant_items_indexes.append(index)

        for index in redundant_items_indexes:
            self.possible_lifted_effects.pop(index)

    def add_possible_effect(self, possible_joint_effect: List[Tuple[str, str]]) -> None:
        """Add a possible joint effect to the list of possible effects.

        :param possible_joint_effect: a list of tuples of the form (action_name, predicate).
        """
        filtered_joint_effect = []
        for (action_name, lifted_predicate) in possible_joint_effect:
            if lifted_predicate in self.not_effects[action_name]:
                continue

            filtered_joint_effect.append((action_name, lifted_predicate))

        if filtered_joint_effect in self.possible_lifted_effects:
            return

        self.possible_lifted_effects.append(filtered_joint_effect)

    def is_action_safe(self, action_name: str, action_preconditions: Set[str]) -> bool:
        """Checks if an action is safe to execute based on this CNF clause.

        :param action_name: the name of the action.
        :param action_preconditions: the preconditions of the action.
        :return: True if the action is safe to execute, False otherwise.
        """
        for lifted_options in self.possible_lifted_effects:
            if action_name in [action for (action, _) in lifted_options]:
                if len(lifted_options) == 1:
                    continue

                for (action, predicate) in lifted_options:
                    if action == action_name and predicate not in action_preconditions:
                        return False

        return True

    def is_action_acting_in_cnf(self, action_name: str) -> bool:
        """Checks if an action is acting in this CNF clause.

        :param action_name: the name of the action.
        :return: True if the action is acting in this CNF clause, False otherwise.
        """
        for possible_joint_effect in self.possible_lifted_effects:
            if action_name in [action for (action, _) in possible_joint_effect]:
                return True

        if len(self.not_effects[action_name]) > 0:
            return True

        return False

    def extract_action_effects(self, action_name: str, action_preconditions: Set[str]) -> List[str]:
        """Extract the effects that an action is acting on.

        :param action_name: the name of the action.
        :param action_preconditions: the preconditions of the action.
        :return: the list of effects that the action is acting on.
        """
        effects = []
        for possible_joint_effect in self.possible_lifted_effects:
            if len(possible_joint_effect) == 1 and \
                    action_name in [action for (action, _) in possible_joint_effect]:
                (_, effect) = possible_joint_effect[0]
                if effect not in action_preconditions:
                    effects.append(effect)

        return effects

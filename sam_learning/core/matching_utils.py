"""Auxiliary functions to help the matching process."""
from itertools import permutations
from typing import List, Tuple, Set

from pddl_plus_parser.models import State, GroundedPredicate


def create_fully_observable_predicates(
        state: State, negative_state_predicates: Set[GroundedPredicate]) -> Tuple[
    List[GroundedPredicate], List[GroundedPredicate]]:
    """Creates a list of fully observable predicates that represent the input state.

    :param state: the state that contains only the positive predicates.
    :param negative_state_predicates: the negative predicates that are not in the state.
    :return: tuple containing the positive and negative predicates.
    """
    positive_predicates = []
    negative_predicates = []

    for negative_state_predicate in negative_state_predicates:
        lifted_predicate_name = negative_state_predicate.lifted_untyped_representation
        if negative_state_predicate.lifted_untyped_representation not in state.state_predicates:
            negative_predicates.append(negative_state_predicate)
            continue

        state_predicate_strs = [predicate.untyped_representation for predicate in
                                state.state_predicates[lifted_predicate_name]]
        if negative_state_predicate.untyped_representation not in state_predicate_strs:
            negative_predicates.append(negative_state_predicate)
            continue

    for lifted_predicate_name, grounded_state_predicates in state.state_predicates.items():
        positive_predicates.extend(grounded_state_predicates)

    return positive_predicates, negative_predicates


def create_signature_permutations(call_parameters: List[str], lifted_signature: List[str],
                                  subset_size: int) -> List[Tuple[Tuple[str]]]:
    """Choose r items our of a list size n.

    :param call_parameters: the parameters in which the action was called with.
    :param lifted_signature: the action's lifted signature definition including possible constants.
    :param subset_size: the size of the subset.
    :return: a list containing matches of grounded objects and their lifted action parameter name counterparts.
    """
    matching_signatures = zip(call_parameters, list(lifted_signature))
    matching_permutations = list(permutations(matching_signatures, subset_size))
    return matching_permutations


def contains_duplicates(parameter_objects: List[str]) -> bool:
    """Validate that the predicate has only one possible match in the literal.

    :param parameter_objects: the of objects to test if contains duplicates.
    :return: whether there are duplicates.
    """
    return len(set(parameter_objects)) != len(parameter_objects)


def extract_effects(previous_state: State, next_state: State,
                    add_predicates_sign: bool = False) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
    """Extracts discrete the effects from the state object.

    :param previous_state: the previous state object containing the grounded literals.
    :param next_state: the next state object containing its grounded literals.
    :param add_predicates_sign: whether to add the sign to the delete effects.
    :return: tuple containing the add and delete effects of the action.
    """
    prev_state_predicate = previous_state.state_predicates
    next_state_predicate = next_state.state_predicates

    add_effects = set()
    delete_effects = set()

    # Checking all the add effects
    for lifted_predicate, grounded_predicates in next_state_predicate.items():
        if lifted_predicate not in prev_state_predicate:
            add_effects.update(grounded_predicates)
            continue

        add_effects.update(grounded_predicates.difference(prev_state_predicate[lifted_predicate]))

    # Checking all delete effects
    for lifted_predicate, grounded_predicates in prev_state_predicate.items():
        delete_effects.update(grounded_predicates.difference(next_state_predicate[lifted_predicate]))

    if add_predicates_sign:
        for delete_effect in delete_effects:
            delete_effect.is_positive = False

    return add_effects, delete_effects

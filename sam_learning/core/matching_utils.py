from itertools import permutations
from typing import List, Tuple, Dict, Set

from pddl_plus_parser.models import State, GroundedPredicate


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
    :return: whether or not there are duplicates.
    """
    return len(set(parameter_objects)) != len(parameter_objects)

def extract_effects(previous_state: State,
                    next_state: State) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
    """Extracts discrete the effects from the state object.

    :param previous_state: the previous state object containing the grounded literals.
    :param next_state: the next state object containing its grounded literals.
    :return: dictionary describing the add and delete effects of the action.
    """
    prev_state_predicate = previous_state.state_predicates
    next_state_predicate = next_state.state_predicates

    add_effects = set()
    delete_effects = set()

    # Checking all the add effects
    for lifted_predicate, grounded_predicates in next_state_predicate.items():
        if lifted_predicate not in prev_state_predicate:
            add_effects[lifted_predicate] = grounded_predicates
            continue

        add_effects.update(grounded_predicates.difference(prev_state_predicate[lifted_predicate]))

    # Checking all the delete effects
    for lifted_predicate, grounded_predicates in prev_state_predicate.items():
        delete_effects.update(grounded_predicates.difference(next_state_predicate[lifted_predicate]))

    return add_effects, delete_effects

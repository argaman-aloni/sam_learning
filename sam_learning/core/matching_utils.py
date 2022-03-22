from itertools import permutations
from typing import List, Tuple


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

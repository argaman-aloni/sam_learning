"""File containing utilities for conditional / universal SAM modules."""

from collections import defaultdict
from typing import Dict, List, Optional, Generator, Tuple

from pddl_plus_parser.models import ActionCall, Predicate, PDDLConstant, PDDLObject, PDDLType, ConditionalEffect

from sam_learning.core.learner_domain import LearnerDomain, LearnerAction

NOT_PREFIX = "(not"
FORALL = "forall"


def extract_predicate_data(
        action: LearnerAction, predicate_str: str, domain_constants: Dict[str, PDDLConstant],
        additional_parameter: Optional[str] = None, additional_parameter_type: Optional[PDDLType] = None) -> Predicate:
    """Extracts the lifted bounded predicate from the string.

    :param action: the action that contains the predicate.
    :param predicate_str: the string representation of the predicate.
    :param domain_constants: the constants of the domain if exists.
    :return: the predicate object matching the string.
    """
    predicate_data = predicate_str.replace("(", "").replace(")", "").split(" ")
    predicate_data = [data for data in predicate_data if data != ""]
    predicate_name = predicate_data[0]
    combined_signature = {**action.signature}
    if additional_parameter is not None:
        combined_signature[additional_parameter] = additional_parameter_type

    combined_signature.update({constant.name: constant.type for constant in domain_constants.values()})
    predicate_signature = {parameter: combined_signature[parameter] for parameter in predicate_data[1:]}
    return Predicate(predicate_name, predicate_signature)


def create_additional_parameter_name(
        domain: LearnerDomain, grounded_action: ActionCall, pddl_type: PDDLType) -> str:
    """Creates a unique name for the additional parameter.

    :param domain: the domain containing the action definition.
    :param grounded_action: the grounded action that had been observed.
    :param pddl_type: the new type that is being added as a parameter.
    :return: the new parameter name.
    """
    index = 1
    additional_parameter_name = f"?{pddl_type.name[0]}"
    while additional_parameter_name in domain.actions[grounded_action.name].signature:
        additional_parameter_name = f"?{pddl_type.name[0]}{index}"
        index += 1

    return additional_parameter_name


def find_unique_objects_by_type(
        trajectory_objects: Dict[str, PDDLObject],
        exclude_list: Optional[List[str]] = None) -> Dict[str, List[PDDLObject]]:
    """Returns a dictionary containing a single object of each type.

    :param trajectory_objects: the objects that were observed in the trajectory.
    :return: a dictionary containing the type as key and a list of objects as value.
    """
    unique_objects_by_type = defaultdict(list)
    for object_name, pddl_object in trajectory_objects.items():
        if exclude_list is not None and object_name in exclude_list:
            continue

        unique_objects_by_type[pddl_object.type.name].append(pddl_object)

    return unique_objects_by_type


def iterate_over_objects_of_same_type(
        trajectory_objects: Dict[str, PDDLObject],
        action_additional_parameters: Dict[str, str],
        exclude_list: Optional[List[str]] = None) -> Generator[Tuple[PDDLObject, str, str], None, None]:
    """Iterate over the objects of the same type.

    :param trajectory_objects: the objects that were observed in the trajectory.
    :param action_additional_parameters: the additional parameters of the action for any possible type.
    :param exclude_list: the list of objects to exclude.
    :return: generator of tuples containing the type name and object.
    """
    objects_by_type = find_unique_objects_by_type(trajectory_objects, exclude_list)
    for parameter_type, parameter_name in action_additional_parameters.items():
        for pddl_object in objects_by_type[parameter_type]:
            yield pddl_object, parameter_type, parameter_name


def check_equal_antecedents(effect: ConditionalEffect, other_effect: ConditionalEffect) -> bool:
    """Checks if two conditional effects' antecedents are equal.

    :param effect: the first effect.
    :param other_effect: the second effect.
    :return: True if the effects' antecedents are equal, False otherwise.
    """
    if len(effect.positive_conditions) != len(other_effect.positive_conditions):
        return False

    for condition in effect.positive_conditions:
        if condition not in other_effect.positive_conditions:
            return False

    if len(effect.negative_conditions) != len(other_effect.negative_conditions):
        return False

    for condition in effect.negative_conditions:
        if condition not in other_effect.negative_conditions:
            return False

    return True
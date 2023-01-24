"""File containing utilities for conditional SAM module."""

from collections import defaultdict
from typing import Dict, List, Optional, Generator, Tuple, Set

from pddl_plus_parser.models import ActionCall, Predicate, PDDLConstant, PDDLObject, PDDLType, GroundedPredicate

from sam_learning.core.learner_domain import LearnerDomain, LearnerAction
from sam_learning.core.predicates_matcher import PredicatesMatcher

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


def extract_quantified_effects(
        grounded_action: ActionCall,
        grounded_add_effects: Set[GroundedPredicate],
        grounded_del_effects: Set[GroundedPredicate],
        trajectory_objects: Dict[str, PDDLObject],
        matcher: PredicatesMatcher,
        action_additional_parameters: Dict[str, str]) -> \
        Generator[Tuple[Set[GroundedPredicate], Set[GroundedPredicate], str, PDDLObject], None, None]:
    """Extract the quantified effects from the grounded effects as a generator.

    :param grounded_action: the action that is currently being observed.
    :param grounded_add_effects: the grounded add effects.
    :param grounded_del_effects: the grounded delete effects.
    :param trajectory_objects: the objects that were observed in the trajectory.
    :param matcher: the matcher object.
    :param action_additional_parameters: the additional parameters of the action for any possible type.
    :return: generator of tuples containing the add effects, delete effects, type name and object.
    """
    objects_by_type = find_unique_objects_by_type(trajectory_objects, grounded_action.parameters)
    for pddl_type_name, pddl_objects in objects_by_type.items():
        additional_parameter_name = action_additional_parameters[pddl_type_name]
        for pddl_object in pddl_objects:
            lifted_add_effects = matcher.get_possible_literal_matches(
                grounded_action, list(grounded_add_effects), pddl_object.name, additional_parameter_name)
            lifted_delete_effects = matcher.get_possible_literal_matches(
                grounded_action, list(grounded_del_effects), pddl_object.name, additional_parameter_name)

            yield lifted_add_effects, lifted_delete_effects, pddl_type_name, pddl_object

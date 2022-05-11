"""Module to match action call parameters to the functions observed in the state."""
import logging
from itertools import permutations
from typing import List, Dict

from pddl_plus_parser.models import Domain, PDDLFunction, ActionCall, Action


class NumericFunctionMatcher:
    """Class that matched numeric functions to the actions according to their parameters."""

    matcher_domain: Domain
    logger: logging.Logger

    def __init__(self, domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.matcher_domain = domain

    @staticmethod
    def create_possible_function_signatures(action_parameters: List[str],
                                            numeric_function: PDDLFunction) -> List[str]:
        """Create all possible grounded calls of the function.

        Note: Assuming the input function has less or the same number of parameters as the action.

        :param action_parameters: the parameters that the action was called with.
        :param numeric_function:the numeric function that is defined in the domain.
        :return: possible unsigned representations that will be used to map the correct state function.
        """
        num_function_params = len(numeric_function.signature)
        possible_parameters = permutations(action_parameters, num_function_params)
        possible_untyped_signatures = []
        for permutation in possible_parameters:
            untyped_signature_str = " ".join(permutation)
            possible_untyped_signatures.append(f"({numeric_function.name} {untyped_signature_str})")

        return possible_untyped_signatures

    def lift_matched_parameters(self, executed_action: Action, grounded_call_parameters: List[str],
                                grounded_function: PDDLFunction) -> PDDLFunction:
        """Matches the parameters of the function to the lifted action definition.

        :param executed_action: the lifted action that was called in the observation.
        :param grounded_call_parameters: the observed action call objects.
        :param grounded_function: the observed numeric function.
        :return: the lifted function definition with the value set according to the observation.
        """
        self.logger.info("Starting to lift the matched parameters!")
        lifted_parameters = executed_action.parameter_names
        grounded_objects = list(grounded_function.signature.keys())
        lifted_signature = {}
        for function_obj in grounded_objects:
            if function_obj in self.matcher_domain.constants:
                lifted_signature[function_obj] = self.matcher_domain.constants[function_obj].type

            else:
                if executed_action.name == "load":
                    print(grounded_call_parameters.index(function_obj), function_obj)
                lifted_param_name = lifted_parameters[grounded_call_parameters.index(function_obj)]
                lifted_param_type = executed_action.signature[lifted_param_name]
                lifted_signature[lifted_param_name] = lifted_param_type

        lifted_state_function = PDDLFunction(name=grounded_function.name,
                                             signature=lifted_signature)
        lifted_state_function.set_value(grounded_function.value)
        return lifted_state_function

    def match_state_functions(self, action_call: ActionCall,
                              grounded_state_fluents: Dict[str, PDDLFunction]) -> Dict[str, PDDLFunction]:
        """Match the state functions to the action according to the grounded object names.

        :param action_call: the action that was called in the observation.
        :param grounded_state_fluents: the numeric state fluents.
        :return: a dictionary containing the lifted numeric state function with their assigned values.
        """
        self.logger.info(f"Starting to search for matches for the grounded action - {str(action_call)}")
        possible_matches = {}
        for domain_function in self.matcher_domain.functions.values():
            if len(domain_function.signature) > len(action_call.parameters) and len(self.matcher_domain.constants) == 0:
                self.logger.debug(f"Function - {domain_function.name} has too many parameters, skipping.")
                continue

            if len(domain_function.signature) == 0:
                grounded_function = grounded_state_fluents[domain_function.untyped_representation]
                self.logger.debug(f"Function {domain_function.name} has zero parameters. "
                                  f"Instantly considering as match.")
                matched_function = PDDLFunction(name=domain_function.name, signature={})
                matched_function.set_value(grounded_function.value)
                possible_matches[domain_function.untyped_representation] = matched_function
                continue

            call_parameters = action_call.parameters + list(self.matcher_domain.constants.keys())
            possible_function_signatures = self.create_possible_function_signatures(
                call_parameters, domain_function)
            for unsigned_representation in possible_function_signatures:
                if unsigned_representation in grounded_state_fluents:
                    self.logger.info(f"found a possible match to the action - {unsigned_representation}")
                    matched_lifted_function = self.lift_matched_parameters(
                        self.matcher_domain.actions[action_call.name],
                        action_call.parameters,
                        grounded_state_fluents[unsigned_representation])
                    possible_matches[matched_lifted_function.untyped_representation] = matched_lifted_function

        return possible_matches

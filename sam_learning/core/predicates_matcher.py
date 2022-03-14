"""Matches predicates to their corresponding actions based on the common types."""
import logging
from typing import List, Tuple, Optional

from pddl_plus_parser.models import Domain, Predicate, GroundedPredicate, ActionCall

from sam_learning.core.matching_utils import contains_duplicates, create_signature_combinations


class PredicatesMatcher:
    """Class that matches predicates according to the needed properties in the learning process."""

    matcher_domain: Domain
    logger: logging.Logger

    def __init__(self, domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.matcher_domain = domain

    def _extract_combinations_data(self, possible_combinations: Tuple[Tuple[str]]) -> Tuple[List[str], List[str]]:
        """extracts the data of the combinations so that the matcher would be able to use it.

        :param possible_combinations: the possible combinations of the parameters.
        :return: the extracted parameters and the objects.
        """
        call_objects = []
        lifted_params = []
        for obj, param in possible_combinations:
            call_objects.append(obj)
            lifted_params.append(param)

        return call_objects, lifted_params

    def match_predicate_to_action_literals(
            self, grounded_predicate: GroundedPredicate, action_call: ActionCall) -> Optional[List[Predicate]]:
        """Matches the action objects to the predicate objects.

        :param grounded_predicate: the grounded predicate that was observed.
        :param action_call: the action that was called in the observation.
        :return: lifted predicates with signatures matching the action.
        """
        self.logger.info(f"Trying to match the grounded predicate - {grounded_predicate.untyped_representation} "
                         f"to the action call {str(action_call)}")
        lifted_action_data = self.matcher_domain.actions[action_call.name]
        constants = self.matcher_domain.constants
        grounded_predicate_call = grounded_predicate.grounded_objects
        action_grounded_objects = action_call.parameters + list(constants.keys())
        lifted_action_params = list(lifted_action_data.signature.keys()) + list(constants.keys())
        if contains_duplicates(action_call.parameters):
            self.logger.debug(f"Action {str(action_call)} was executed with duplicated objects!")

        possible_parameter_permutations = create_signature_combinations(
            action_grounded_objects, lifted_action_params, len(grounded_predicate_call))
        possible_matches = []
        for possible_permutation in possible_parameter_permutations:
            possible_match_action_objects, lifted_parameters = self._extract_combinations_data(possible_permutation)
            if set(possible_match_action_objects) == set(grounded_predicate_call):
                combined_types = {**lifted_action_data.signature}
                combined_types.update({name: constants[name].type for name in constants})
                possible_matches.append(Predicate(
                    name=grounded_predicate.name, signature={
                        lifted_parameter_name: combined_types[lifted_parameter_name] for lifted_parameter_name in
                        lifted_parameters
                    }))

        return possible_matches

    def get_possible_literal_matches(
            self, grounded_action_call: ActionCall, state_literals: List[GroundedPredicate]) -> List[Predicate]:
        """Get a list of possible preconditions for the action according to the previous state.

        :param grounded_action_call: the grounded action that was executed according to the trajectory.
        :param state_literals: the list of literals that we try to match according to the action.
        :return: a list of possible preconditions for the action that is being executed.
        """
        self.logger.info(f"Finding the possible matches for the grounded action - {str(grounded_action_call)}")
        possible_matches = []
        for state_predicate in state_literals:
            matches = self.match_predicate_to_action_literals(state_predicate, grounded_action_call)
            if matches is None:
                continue

            possible_matches.extend(matches)

        return possible_matches

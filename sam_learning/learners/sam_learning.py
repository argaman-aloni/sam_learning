"""The Safe Action Model Learning algorithm module."""
import logging
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, NoReturn, Dict, Set

from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, PDDLObject

from sam_learning.core import PredicatesMatcher, extract_effects, LearnerDomain, contains_duplicates, VocabularyCreator, \
    LearnerAction


class SAMLearner:
    """Class that represents the safe action model learner algorithm."""

    logger: logging.Logger
    partial_domain: LearnerDomain
    matcher: PredicatesMatcher
    observed_actions: List[str]

    def __init__(self, partial_domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.partial_domain = LearnerDomain(domain=partial_domain)
        self.matcher = PredicatesMatcher(partial_domain)
        self.observed_actions = []
        self.vocabulary_creator = VocabularyCreator()

    def _reduce_constants_from_effects(self, grounded_action: ActionCall, lifted_add_effects: List[Predicate],
                                       lifted_delete_effects: List[Predicate]):
        """Removed predicates that contain constants from the effects of the action to not err and add extra effects.

        :param grounded_action: the grounded action.
        :param lifted_add_effects: the lifted add effects of the action.
        :param lifted_delete_effects: the lifted delete effects of the action.
        """
        constants = list(self.partial_domain.constants.keys())
        add_effects_to_remove = set()
        for effect in lifted_add_effects:
            for constant in constants:
                if constant in effect.signature and constant in grounded_action.parameters:
                    add_effects_to_remove.add(effect)

        delete_effects_to_remove = set()
        for effect in lifted_delete_effects:
            for constant in constants:
                if constant in effect.signature and constant in grounded_action.parameters:
                    delete_effects_to_remove.add(effect)

        for effect in add_effects_to_remove:
            lifted_add_effects.remove(effect)

        for effect in delete_effects_to_remove:
            lifted_delete_effects.remove(effect)

    def _handle_action_effects(
            self, grounded_action: ActionCall, previous_state: State,
            next_state: State) -> Tuple[List[Predicate], List[Predicate]]:
        """Finds the effects generated from the previous and the next state on this current step.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        :return: the effect containing the add and del list of predicates.
        """
        self.logger.debug(f"Starting to learn the effects of {str(grounded_action)}.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))
        if len(self.partial_domain.constants) > 0:
            self._reduce_constants_from_effects(grounded_action, lifted_add_effects, lifted_delete_effects)

        return lifted_add_effects, lifted_delete_effects

    def _update_action_preconditions(
            self, grounded_action: ActionCall, previous_state: State) -> NoReturn:
        """Updates the preconditions of an action after it was observed at least once.

        :param grounded_action: the grounded action that is being executed in the trajectory component.
        :param previous_state: the state that was seen prior to the action's execution.
        """
        current_action = self.partial_domain.actions[grounded_action.name]
        possible_preconditions = set()
        for lifted_predicate_name, grounded_state_predicates in previous_state.state_predicates.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_state_predicates))
            possible_preconditions.update(lifted_matches)

        if len(possible_preconditions) > 0:
            current_action.positive_preconditions.intersection_update(possible_preconditions)
            current_action.negative_preconditions.difference_update(possible_preconditions)

        else:
            self.logger.warning(f"while handling the action {grounded_action.name} "
                                f"inconsistency occurred, since we do not allow for duplicates we do not update the "
                                f"preconditions.")

    def _add_negative_predicates(self, grounded_action: ActionCall, state: State,
                                 observed_objects: Dict[str, PDDLObject]) -> Set[Predicate]:
        """Adds a negative predicate to the action when it is first encountered.

        :param grounded_action: the action that was encountered.
        :param state: the state in which the vocabulary is based on.
        :param observed_objects: the objects that were observed in the trajectory.
        :return: the possible negative predicates that were added to the action.
        """
        possible_negative_predicates = set()
        vocabulary = self.vocabulary_creator.create_vocabulary(self.partial_domain, observed_objects)
        for lifted_predicate_name, grounded_missing_predicates in vocabulary.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            if lifted_predicate_name not in state.state_predicates:
                lifted_matches = self.matcher.get_possible_literal_matches(
                    grounded_action, list(grounded_missing_predicates))
                possible_negative_predicates.update(lifted_matches)
                continue

            filtered_grounded_state_predicates = [predicate for predicate in grounded_missing_predicates if predicate
                                                  not in state.state_predicates[lifted_predicate_name]]
            if len(filtered_grounded_state_predicates) > 0:
                lifted_matches = self.matcher.get_possible_literal_matches(
                    grounded_action, filtered_grounded_state_predicates)
                possible_negative_predicates.update(lifted_matches)

        return possible_negative_predicates

    def _add_positive_predicates(self, grounded_action: ActionCall, state: State) -> Set[Predicate]:
        """Adds positive predicates for the not possible delete effects vocabulary.

        :param grounded_action: the action that was encountered.
        :param state: the state in which the vocabulary is based on.
        :return: the possible negative predicates that were added to the action.
        """
        possible_positive_predicates = set()
        for predicate_set in state.state_predicates.values():
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(predicate_set))
            possible_positive_predicates.update(lifted_matches)

        return possible_positive_predicates

    def add_new_action(self, grounded_action: ActionCall, previous_state: State,
                       next_state: State, observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        :param observed_objects: the objects that were observed in the current trajectory.
            state.
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        self._add_new_action_preconditions(grounded_action, observed_action, observed_objects, previous_state)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def _add_new_action_preconditions(self, grounded_action: ActionCall, observed_action: LearnerAction,
                                      observed_objects: Dict[str, PDDLObject], previous_state: State) -> NoReturn:
        """General method to add new action's discrete preconditions.

        :param grounded_action: the action that is currently being executed.
        :param observed_action: the action that is being added to the domain.
        :param observed_objects: the objects that were observed in the current trajectory.
        :param previous_state: the state that the action was executed on.
        """
        possible_preconditions = set()
        negative_predicates = self._add_negative_predicates(grounded_action, previous_state, observed_objects)
        for lifted_predicate_name, grounded_state_predicates in previous_state.state_predicates.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_state_predicates))
            possible_preconditions.update(lifted_matches)

        observed_action.positive_preconditions.update(possible_preconditions)
        observed_action.negative_preconditions.update(negative_predicates)

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]
        self._update_action_preconditions(grounded_action, previous_state)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def _verify_parameter_duplication(self, grounded_action: ActionCall) -> bool:
        """Verifies if the action was called with duplicated objects in a trajectory component.

        :param grounded_action: the grounded action observed in the trajectory triplet.
        :return: whether the action contains duplicated parameters.
        """
        has_duplicates = contains_duplicates(grounded_action.parameters)
        if has_duplicates:
            action = self.partial_domain.actions[grounded_action.name]
            grounded_signature_map = defaultdict(list)
            for grounded_param, lifted_param in zip(grounded_action.parameters, action.parameter_names):
                grounded_signature_map[grounded_param].append(lifted_param)

            for lifted_duplicates_list in grounded_signature_map.values():
                for (obj1, obj2) in combinations(lifted_duplicates_list, 2):
                    action.inequality_preconditions.discard((obj1, obj2))

        return has_duplicates

    def handle_single_trajectory_component(self, component: ObservedComponent,
                                           observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        :param observed_objects: the objects that were observed in the current trajectory.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        if self._verify_parameter_duplication(grounded_action):
            self.logger.warning(f"{str(grounded_action)} contains duplicated parameters! Not suppoerted in SAM.")
            return

        if action_name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state, observed_objects)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self.deduce_initial_inequality_preconditions()
        for observation in observations:
            for component in observation.components:
                self.handle_single_trajectory_component(component, observation.grounded_objects)

        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

    def deduce_initial_inequality_preconditions(self) -> NoReturn:
        """Tries to deduce which objects in the actions' signature cannot be equal."""
        self.logger.debug("Starting to deduce inequality preconditions")
        for action_name, action_data in self.partial_domain.actions.items():
            for (lifted_param1, lifted_param2) in combinations(action_data.parameter_names, 2):
                if action_data.signature[lifted_param1] == action_data.signature[lifted_param2]:
                    action_data.inequality_preconditions.add((lifted_param1, lifted_param2))

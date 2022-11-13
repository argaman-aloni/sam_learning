"""The Safe Action Model Learning algorithm module."""
import logging
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Set

from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, PDDLObject, \
    GroundedPredicate

from sam_learning.core import PredicatesMatcher, extract_effects, LearnerDomain, contains_duplicates, VocabularyCreator, \
    LearnerAction


class SAMLearner:
    """Class that represents the safe action model learner algorithm.

    Notice: This class does not support domains with constants or with the same object mapped to multiple parameters.
    """

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

    def _handle_action_effects(
            self, grounded_action: ActionCall, previous_state: State,
            next_state: State, grounded_negative_state_predicates: Set[GroundedPredicate]) -> Tuple[
        List[Predicate], List[Predicate]]:
        """Finds the effects generated from the previous and the next state on this current step.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        :return: the effect containing the add and del list of predicates.
        """
        self.logger.debug(f"Starting to learn the effects of {str(grounded_action)}.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        self.logger.debug("Updating the negative state predicates based on the action's execution.")
        grounded_negative_state_predicates.update(grounded_del_effects)
        grounded_negative_state_predicates.difference_update(grounded_add_effects)
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))
        return lifted_add_effects, lifted_delete_effects

    def _update_action_preconditions(
            self, grounded_action: ActionCall, previous_state: State) -> None:
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

    def _add_negative_predicates(self, grounded_action: ActionCall,
                                 negative_state_predicates: Set[GroundedPredicate]) -> Set[Predicate]:
        """Adds a negative predicate to the action when it is first encountered.

        :param grounded_action: the action that was encountered.
        :param state: the state in which the vocabulary is based on.
        :param initial_state: the initial state of the the currently iterated trajectory.
        :return: the possible negative predicates that were added to the action.
        """
        possible_negative_predicates = set()
        lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(negative_state_predicates))
        possible_negative_predicates.update(lifted_matches)

        return possible_negative_predicates

    def _add_new_action_preconditions(self, grounded_action: ActionCall, observed_action: LearnerAction,
                                      negative_state_predicates: Set[GroundedPredicate], previous_state: State) -> None:
        """General method to add new action's discrete preconditions.

        :param grounded_action: the action that is currently being executed.
        :param observed_action: the action that is being added to the domain.
        :param negative_state_predicates: the negative predicates that were encountered in the state.
        :param previous_state: the state that the action was executed on.
        """
        possible_preconditions = set()
        negative_predicates = self._add_negative_predicates(grounded_action, negative_state_predicates)
        for lifted_predicate_name, grounded_state_predicates in previous_state.state_predicates.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_state_predicates))
            possible_preconditions.update(lifted_matches)

        observed_action.positive_preconditions.update(possible_preconditions)
        observed_action.negative_preconditions.update(negative_predicates)

    def add_new_action(self, grounded_action: ActionCall, previous_state: State,
                       next_state: State, negative_state_predicates: Set[GroundedPredicate]) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        :param negative_state_predicates: the negative predicates that were encountered in the state.
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        self._add_new_action_preconditions(grounded_action, observed_action, negative_state_predicates, previous_state)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state, negative_state_predicates)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State,
            negative_state_predicates: Set[GroundedPredicate]) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        :param negative_state_predicates: the negative predicates that were encountered in the state.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]
        self._update_action_preconditions(grounded_action, previous_state)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state, negative_state_predicates)

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

    def _create_complete_world_state(self, observed_trajectory_objects: Dict[str, PDDLObject],
                                     state: State) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        """Creates a complete representation of the world state from the observed objects in the trajectory.

        :param observed_trajectory_objects: the objects extracted from the trajectory.
        :param state: the state that contains only the predicates that are true in it.
        :return: a complete representation of the world state containing both positive predicates and negative
            predicates.
        """
        self.logger.debug("Creating a complete world state")
        positive_state_predicates = set()
        negative_state_predicates = set()
        vocabulary = self.vocabulary_creator.create_vocabulary(domain=self.partial_domain,
                                                               observed_objects=observed_trajectory_objects)

        for lifted_predicate_name, possible_state_predicates in vocabulary.items():
            if lifted_predicate_name not in state.state_predicates:
                negative_state_predicates.update(possible_state_predicates)
                continue

            state_predicate_strs = [predicate.untyped_representation for predicate in
                                    state.state_predicates[lifted_predicate_name]]
            filtered_grounded_state_predicates = [predicate for predicate in possible_state_predicates if
                                                  predicate.untyped_representation not in state_predicate_strs]
            negative_state_predicates.update(filtered_grounded_state_predicates)

        for lifted_predicate_name, grounded_state_predicates in state.state_predicates.items():
            positive_state_predicates.update(grounded_state_predicates)

        return positive_state_predicates, negative_state_predicates

    def handle_single_trajectory_component(
            self, component: ObservedComponent,
            negative_state_predicates: Set[GroundedPredicate]) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        :param initial_state: the initial state of the currently iterated trajectory.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        if self._verify_parameter_duplication(grounded_action):
            self.logger.warning(f"{str(grounded_action)} contains duplicated parameters! Not suppoerted in SAM.")
            return

        if action_name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state, negative_state_predicates)

        else:
            self.update_action(grounded_action, previous_state, next_state, negative_state_predicates)

    def deduce_initial_inequality_preconditions(self) -> None:
        """Tries to deduce which objects in the actions' signature cannot be equal."""
        self.logger.debug("Starting to deduce inequality preconditions")
        for action_name, action_data in self.partial_domain.actions.items():
            for (lifted_param1, lifted_param2) in combinations(action_data.parameter_names, 2):
                if action_data.signature[lifted_param1] == action_data.signature[lifted_param2]:
                    action_data.inequality_preconditions.add((lifted_param1, lifted_param2))

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self.deduce_initial_inequality_preconditions()
        for observation in observations:
            observed_trajectory_objects = observation.grounded_objects
            initial_state = observation.components[0].previous_state
            positive_state_predicates, negative_state_predicates = self._create_complete_world_state(
                observed_trajectory_objects, initial_state)
            for component in observation.components:
                self.handle_single_trajectory_component(component, negative_state_predicates)

        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

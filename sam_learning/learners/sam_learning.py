"""The Safe Action Model Learning algorithm module."""
import logging
import time
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Set

from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, PDDLObject, \
    GroundedPredicate

from sam_learning.core import PredicatesMatcher, extract_effects, LearnerDomain, contains_duplicates, \
    VocabularyCreator


class SAMLearner:
    """Class that represents the safe action model learner algorithm.

    Notice: This class does not support domains with constants or with the same object mapped to multiple parameters.
    """

    logger: logging.Logger
    partial_domain: LearnerDomain
    matcher: PredicatesMatcher
    observed_actions: List[str]
    safe_actions: List[str]
    previous_state_positive_predicates: Set[GroundedPredicate]
    previous_state_negative_predicates: Set[GroundedPredicate]
    next_state_negative_predicates: Set[GroundedPredicate]
    next_state_positive_predicates: Set[GroundedPredicate]
    current_trajectory_objects: Dict[str, PDDLObject]
    learning_start_time: float
    learning_end_time: float

    def __init__(self, partial_domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.partial_domain = LearnerDomain(domain=partial_domain)
        self.matcher = PredicatesMatcher(partial_domain)
        self.observed_actions = []
        self.vocabulary_creator = VocabularyCreator()
        self.safe_actions = []
        self.previous_state_positive_predicates = set()
        self.previous_state_negative_predicates = set()
        self.next_state_positive_predicates = set()
        self.next_state_negative_predicates = set()
        self.current_trajectory_objects = {}
        self.learning_start_time = 0
        self.learning_end_time = 0

    def _remove_unobserved_actions_from_partial_domain(self):
        """Removes the actions that were not observed from the partial domain."""
        self.logger.debug("Removing unobserved actions from the partial domain")
        actions_to_remove = [action for action in self.partial_domain.actions if action not in self.observed_actions]
        for action in actions_to_remove:
            self.partial_domain.actions.pop(action)

    def _create_complete_world_state(self, relevant_objects: Dict[str, PDDLObject],
                                     state: State) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        """Creates a complete representation of the world state from the observed objects in the trajectory.

        :param relevant_objects: the objects extracted from the trajectory which are relevant to the action.
        :param state: the state that contains only the predicates that are true in it.
        :return: a complete representation of the world state containing both positive predicates and negative
            predicates.
        """
        self.logger.debug("Creating a complete world state")
        positive_state_predicates = set()
        negative_state_predicates = set()
        vocabulary = self.vocabulary_creator.create_vocabulary(domain=self.partial_domain,
                                                               observed_objects=relevant_objects)

        for lifted_predicate_name, possible_missing_predicates in vocabulary.items():
            if lifted_predicate_name not in state.state_predicates:
                negative_state_predicates.update([GroundedPredicate(name=p.name, signature=p.signature,
                                                                    object_mapping=p.object_mapping, is_positive=False)
                                                  for p in possible_missing_predicates])
                continue

            for predicate in state.state_predicates[lifted_predicate_name]:
                predicate.is_positive = True

            state_predicate_strs = [predicate.untyped_representation for predicate in
                                    state.state_predicates[lifted_predicate_name]]
            filtered_grounded_state_predicates = [predicate for predicate in possible_missing_predicates if
                                                  predicate.untyped_representation not in state_predicate_strs]
            filtered_grounded_state_predicates_str = [predicate.untyped_representation for predicate in
                                                      filtered_grounded_state_predicates]
            negative_state_predicates.update([GroundedPredicate(name=p.name, signature=p.signature,
                                                                object_mapping=p.object_mapping, is_positive=False)
                                              for p in filtered_grounded_state_predicates])

        for lifted_predicate_name, grounded_state_predicates in state.state_predicates.items():
            positive_state_predicates.update([GroundedPredicate(name=p.name, signature=p.signature,
                                                                object_mapping=p.object_mapping, is_positive=True)
                                              for p in grounded_state_predicates])

        return positive_state_predicates, negative_state_predicates

    def _create_fully_observable_triplet_predicates(
            self, current_action: ActionCall, previous_state: State, next_state: State,
            should_ignore_action: bool = False) -> None:
        """Sets the fully observable previous state and next state predicates for the current action.

        :param current_action: the action that is currently being executed.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        :param should_ignore_action: whether the action should be ignored and then use all trajectory objects or not.
        """
        relevant_objects = {object_name: object_data for object_name, object_data in
                            self.current_trajectory_objects.items()
                            if object_name in current_action.parameters} \
            if not should_ignore_action else self.current_trajectory_objects
        self.previous_state_positive_predicates, self.previous_state_negative_predicates = \
            self._create_complete_world_state(relevant_objects=relevant_objects, state=previous_state)
        self.next_state_positive_predicates, self.next_state_negative_predicates = \
            self._create_complete_world_state(relevant_objects=relevant_objects, state=next_state)

    def _handle_action_effects(self, grounded_action: ActionCall, previous_state: State,
                               next_state: State) -> Tuple[List[Predicate], List[Predicate]]:
        """Finds the effects generated from the previous and the next state on this current step.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        :return: the effect containing the add and del list of predicates.
        """
        self.logger.debug(f"Starting to learn the effects of {str(grounded_action)}.")
        grounded_add_effects, grounded_del_effects = extract_effects(
            previous_state, next_state, add_predicates_sign=True)
        self.logger.debug("Updating the negative state predicates based on the action's execution.")
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))
        return lifted_add_effects, lifted_delete_effects

    def _update_action_preconditions(self, grounded_action: ActionCall) -> None:
        """Updates the preconditions of an action after it was observed at least once.

        :param grounded_action: the grounded action that is being executed in the trajectory component.
        """
        current_action = self.partial_domain.actions[grounded_action.name]
        positive_predicates = set(self.matcher.get_possible_literal_matches(
            grounded_action, list(self.previous_state_positive_predicates)))
        negative_predicates = set(self.matcher.get_possible_literal_matches(
            grounded_action, list(self.previous_state_negative_predicates)))

        current_action.positive_preconditions.intersection_update(positive_predicates)
        current_action.negative_preconditions.intersection_update(negative_predicates)

    def _add_new_action_preconditions(self, grounded_action: ActionCall) -> None:
        """General method to add new action's discrete preconditions.

        :param grounded_action: the action that is currently being executed.
        """
        observed_action = self.partial_domain.actions[grounded_action.name]
        possible_preconditions = set(self.matcher.get_possible_literal_matches(
            grounded_action, list(self.previous_state_positive_predicates)))
        negative_predicates = set(self.matcher.get_possible_literal_matches(
            grounded_action, list(self.previous_state_negative_predicates)))

        observed_action.positive_preconditions = possible_preconditions
        observed_action.negative_preconditions = negative_predicates

    def _construct_learning_report(self) -> Dict[str, str]:
        """Constructs the learning report of the learned actions.

        :return: the learning report.
        """
        observed_unsafe_actions = set(self.observed_actions).difference(self.safe_actions)
        unobserved_actions = set(self.partial_domain.actions.keys()).difference(self.observed_actions)

        learning_report = {action_name: "OK" for action_name in self.safe_actions}
        learning_report.update({action_name: "NOT SAFE" for action_name in self.partial_domain.actions
                                if action_name in observed_unsafe_actions})
        learning_report.update({action_name: "UNOBSERVED" for action_name in unobserved_actions})
        learning_report["learning_time"] = str(self.learning_end_time - self.learning_start_time)
        return learning_report

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        self._add_new_action_preconditions(grounded_action)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)
        observed_action.negative_preconditions.difference_update(lifted_delete_effects)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]
        self._update_action_preconditions(grounded_action)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)
        observed_action.negative_preconditions.difference_update(lifted_delete_effects)
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

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state

        if self._verify_parameter_duplication(grounded_action):
            self.logger.warning(f"{str(grounded_action)} contains duplicated parameters! Not suppoerted in SAM.")
            return

        self._create_fully_observable_triplet_predicates(grounded_action, previous_state, next_state)
        if grounded_action.name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def deduce_initial_inequality_preconditions(self) -> None:
        """Tries to deduce which objects in the actions' signature cannot be equal."""
        self.logger.debug("Starting to deduce inequality preconditions")
        for action_name, action_data in self.partial_domain.actions.items():
            for (lifted_param1, lifted_param2) in combinations(action_data.parameter_names, 2):
                if action_data.signature[lifted_param1] == action_data.signature[lifted_param2]:
                    action_data.inequality_preconditions.add((lifted_param1, lifted_param2))

    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        pass

    def start_measure_learning_time(self) -> None:
        """Starts measuring the learning time."""
        self.logger.info("Starting to measure the time it takes to learn the action model!")
        self.learning_start_time = time.time()

    def end_measure_learning_time(self) -> None:
        """Ends measuring the learning time."""
        self.learning_end_time = time.time()
        self.logger.info(f"Finished learning the action model in "
                         f"{self.learning_end_time - self.learning_start_time} seconds.")

    def are_state_different(self, previous_state: State, next_state: State) -> bool:
        """Checks if the previous state differs from the next state.

        :param previous_state: the previous state.
        :param next_state: the next state.
        :return: whether the states differ.
        """
        if previous_state.serialize() == next_state.serialize():
            self.logger.warning("The previous state is the same as the next state. "
                                "This is not supported by the SAFE action model.")
            return False

        return True

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self.start_measure_learning_time()
        self.deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                if not self.are_state_different(component.previous_state, component.next_state):
                    continue

                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        self.end_measure_learning_time()
        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

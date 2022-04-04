"""The Safe Action Model Learning algorithm module."""
import logging
from typing import List, Tuple, NoReturn

from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent

from sam_learning.core import PredicatesMatcher, extract_effects, LearnerDomain, contains_duplicates


class SAMLearner:
    """Class that represents the safe action model learner algorithm."""

    logger: logging.Logger
    observations: List[Observation]
    partial_domain: LearnerDomain
    matcher: PredicatesMatcher
    observed_actions: List[str]

    def __init__(self, partial_domain: Domain):
        self.logger = logging.getLogger(__name__)
        self.observations = []
        self.partial_domain = LearnerDomain(domain=partial_domain)
        self.matcher = PredicatesMatcher(partial_domain)
        self.observed_actions = []

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

        else:
            self.logger.warning(f"while handling the action {grounded_action.name} "
                                f"inconsistency occurred, since we do not allow for duplicates we do not update the "
                                f"preconditions.")

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        possible_preconditions = set()
        for lifted_predicate_name, grounded_state_predicates in previous_state.state_predicates.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_state_predicates))
            possible_preconditions.update(lifted_matches)

        observed_action.positive_preconditions.update(possible_preconditions)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)
        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

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
        return contains_duplicates(grounded_action.parameters)

    def handle_single_trajectory_component(self, component: ObservedComponent) -> NoReturn:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        if self._verify_parameter_duplication(grounded_action):
            self.logger.warning(f"{str(grounded_action)} contains duplicated parameters! Not suppoerted in SAM.")
            return

        if action_name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def learn_action_model(self, observations: List[Observation]) -> LearnerDomain:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        for observation in observations:
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        return self.partial_domain

"""An extension to the SAM learning algorithm that can learn when the matching process is not injective."""
from collections import defaultdict
from typing import List, Tuple, Dict, Set

from pddl_plus_parser.models import Observation, Predicate, ActionCall, State, Domain, ObservedComponent, \
    GroundedPredicate

from sam_learning.core import extract_effects, LearnerDomain, LiteralCNF
from sam_learning.learners import SAMLearner


class ExtendedSAM(SAMLearner):
    """An extension to SAM That can learn in cases of non injective matching results."""

    positive_literals_cnf: Dict[str, LiteralCNF]
    negative_literals_cnf: Dict[str, LiteralCNF]
    should_create_proxy_actions: bool  # whether to create proxy actions for the maybe effects.

    def __init__(self, partial_domain: Domain, should_create_proxy_actions: bool = False):
        super().__init__(partial_domain)
        self.possible_add_effects = defaultdict(set)
        self.possible_delete_effects = defaultdict(set)
        self.should_create_proxy_actions = should_create_proxy_actions

    def _extract_maybe_delete_effects(self, grounded_action: ActionCall, grounded_del_effects: Set[GroundedPredicate],
                                      must_be_delete_effects: List[Predicate]) -> None:
        """Extracts delete effects based on non injective matches.

        :param grounded_action: the action that is being handled.
        :param grounded_del_effects: the grounded predicates that were present in the pre-state and removed from the
            post state.
        :param must_be_delete_effects: the list of predicates that are definitely delete effects.
        """
        for effect in grounded_del_effects:
            lifted_literal_matches = self.matcher.match_predicate_to_action_literals(effect, grounded_action)
            if len(lifted_literal_matches) == 0:
                continue

            elif len(lifted_literal_matches) == 1:
                must_be_delete_effects.append(lifted_literal_matches[0])

            else:
                self.logger.debug("Encountered a non injective match! adding to maybe delete effects.")
                self.possible_delete_effects[grounded_action.name].update(lifted_literal_matches)

    def _extract_maybe_add_effects(self, grounded_action: ActionCall, grounded_add_effects: Set[GroundedPredicate],
                                   must_be_add_effects: List[Predicate]) -> None:
        """Extracts the add effects based on non injective matches.

        :param grounded_action: the action that is being handled.
        :param grounded_add_effects: the predicates that were added to the state after executing the action.
        :param must_be_add_effects: the list of predicates that are definitely add effects.
        """
        for effect in grounded_add_effects:
            lifted_literal_matches = self.matcher.match_predicate_to_action_literals(effect, grounded_action)
            if len(lifted_literal_matches) == 0:
                continue
            elif len(lifted_literal_matches) == 1:
                must_be_add_effects.append(lifted_literal_matches[0])

            else:
                self.logger.debug("Encountered a non injective match! adding to maybe add effects.")
                self.possible_add_effects[grounded_action.name].update(lifted_literal_matches)

    def _extract_non_injective_matches(
            self, grounded_action: ActionCall,
            grounded_add_effects: Set[GroundedPredicate],
            grounded_del_effects: Set[GroundedPredicate]) -> Tuple[List[Predicate], List[Predicate]]:
        """Extracts the effects where some of them might be based on non injective matches.

        :param grounded_action: the action that is being handled.
        :param grounded_add_effects: the add effects that were observed based on the diff between the post and pre
            states.
        :param grounded_del_effects: the delete effects that were observed based on the diff between the post and pre
            states.
        :return: the effects that are definitely add and delete effects (based on injective matching process).
        """
        self.logger.debug(f"Extracting non injective matches of action {grounded_action.name}.")
        must_be_add_effects = []
        must_be_delete_effects = []
        self._extract_maybe_add_effects(grounded_action, grounded_add_effects, must_be_add_effects)
        self._extract_maybe_delete_effects(grounded_action, grounded_del_effects, must_be_delete_effects)

        return must_be_add_effects, must_be_delete_effects

    def _remove_impossible_effects(self, grounded_action: ActionCall,
                                   state_predicates: Dict[str, Set[GroundedPredicate]]) -> None:
        """Removes effects based on the second SAM rule - if an literal has not been observed in the post state
        it cannot be an effect.

        :param grounded_action: the action that is being handled.
        :param state_predicates: the grounded predicates observed in the post state.
        """
        self.logger.info(f"Removing impossible effects of action {grounded_action.name} based on rule 2 of SAM.")
        observed_predicates = set()
        for lifted_predicate, grounded_predicates in state_predicates.items():
            observed_predicates.update(grounded_predicates)

        lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(observed_predicates))
        self.possible_add_effects[grounded_action.name].intersection_update(lifted_matches)
        self.possible_delete_effects[grounded_action.name].difference_update(lifted_matches)

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
        must_be_add_effects, must_be_delete_effects = \
            self._extract_non_injective_matches(grounded_action, grounded_add_effects, grounded_del_effects)

        self._remove_impossible_effects(grounded_action, next_state.state_predicates)
        return must_be_add_effects, must_be_delete_effects

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.discrete_effects.update(set(lifted_add_effects).union(lifted_delete_effects))
        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        observed_action = self.partial_domain.actions[action_name]
        super()._update_action_preconditions(grounded_action)
        lifted_add_effects, lifted_delete_effects = self._handle_action_effects(
            grounded_action, previous_state, next_state)

        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=grounded_action,
            observation_objects=self.current_trajectory_objects)
        if action_name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def create_proxy_actions(self) -> None:
        """Create proxy actions to handle the non injective effects."""
        self.logger.info("Starting to create proxy actions.")
        for action in self.partial_domain.actions.values():
            self.possible_add_effects[action.name].difference_update(action.add_effects)
            self.possible_delete_effects[action.name].difference_update(action.delete_effects)
            if len(self.possible_add_effects[action.name]) > 0 or len(self.possible_delete_effects[action.name]) > 0:
                self.logger.debug(f"Not all ambiguities were solved! Creating proxy actions for {action.name}.")
                # TODO: complete this part.

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        if self.should_create_proxy_actions:
            self.logger.info("Starting to create proxy actions for the effects that were not determined.")
            self.create_proxy_actions()

        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

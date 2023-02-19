"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, MultiAgentObservation, ActionCall, State, \
    GroundedPredicate, JointActionCall

from sam_learning.core import LearnerDomain, extract_effects, LiteralCNF, LearnerAction
from sam_learning.learners import SAMLearner


class MultiAgentSAM(SAMLearner):
    """Class designated to learning action models from multi-agent trajectories with joint actions."""
    logger: logging.Logger
    literals_cnf: Dict[str, LiteralCNF]
    preconditions_fluent_map: Dict[str, List[str]]
    lifted_bounded_predicates: Dict[str, Dict[str, Set[Tuple[str, Predicate]]]]
    safe_actions: List[str]

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.literals_cnf = {}
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.lifted_bounded_predicates = {action_name: defaultdict(set) for action_name in
                                          self.partial_domain.actions.keys()}
        self.safe_actions = []

    def _initialize_cnfs(self) -> None:
        """Initialize the CNFs for the action model."""
        self.logger.debug("Initializing CNFs for the action model.")
        for predicate in self.partial_domain.predicates.values():
            self.literals_cnf[predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))
            negative_predicate = Predicate(name=predicate.name, signature=predicate.signature, is_positive=False)
            self.literals_cnf[negative_predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))

    def _extract_relevant_not_effects(
            self, in_state_predicates: Set[GroundedPredicate],
            removed_predicates: Set[GroundedPredicate],
            executing_actions: List[ActionCall], relevant_action: ActionCall) -> List[GroundedPredicate]:
        """Extracts the literals that cannot be an effect of the relevant action.

        :param in_state_predicates: the predicates that appear in the next state and cannot be delete effects of the action.
        :param removed_predicates: the predicates that are missing in the next state and cannot be add effects of the action.
        :param executing_actions: the actions that are executed in the joint action triplet.
        :param relevant_action: the current action that is being tested.
        :return: the literals that cannot be effects of the action.
        """
        not_effects = []
        cannot_be_add_effects = [grounded_predicate for grounded_predicate in removed_predicates if relevant_action in
                                 self.compute_interacting_actions(grounded_predicate, executing_actions)]
        for not_add_effect in cannot_be_add_effects:
            not_effects.append(GroundedPredicate(name=not_add_effect.name, signature=not_add_effect.signature,
                                                 object_mapping=not_add_effect.object_mapping, is_positive=True))

        cannot_be_del_effects = [grounded_predicate for grounded_predicate in in_state_predicates if relevant_action in
                                 self.compute_interacting_actions(grounded_predicate, executing_actions)]
        for not_del_effect in cannot_be_del_effects:
            not_effects.append(GroundedPredicate(name=not_del_effect.name, signature=not_del_effect.signature,
                                                 object_mapping=not_del_effect.object_mapping, is_positive=False))

        return not_effects

    def _is_action_safe(self, action: LearnerAction, preconditions_to_filter: Set[Predicate]) -> bool:
        """Checks if the given action is safe to execute.

        :param action: the lifted action that is to be learned.
        :return: whether the action is safe to execute.
        """
        for domain_literal, cnf in self.literals_cnf.items():
            if domain_literal not in self.lifted_bounded_predicates[action.name] or \
                    not cnf.is_action_acting_in_cnf(action.name):
                self.logger.debug(f"The literal {domain_literal} does not relate to {action.name}.")
                continue

            bounded_action_predicates = self.lifted_bounded_predicates[action.name][domain_literal]
            bounded_predicates_str = {predicate_str for predicate_str, _ in bounded_action_predicates}
            bounded_predicates_str.difference_update([p.untyped_representation for p in preconditions_to_filter])
            if len(bounded_predicates_str) == 0:
                continue

            if not cnf.is_action_safe(action_name=action.name, bounded_lifted_predicates=bounded_predicates_str):
                self.logger.debug("Action %s is not safe to execute!", action.name)
                return False

        return True

    def add_not_effect_to_cnf(
            self, executed_action: ActionCall, not_effects: List[GroundedPredicate]) -> None:
        """Adds a predicate that cannot be an action's effect to the correct CNF.

        :param executed_action: the action that is being executed in the current joint action triplet.
        :param not_effects: the predicates that cannot be the effects of the action.
        """
        for predicate in not_effects:
            bounded_lifted_literal = self.matcher.get_injective_match(predicate, executed_action)
            if bounded_lifted_literal is None:
                continue

            self.literals_cnf[predicate.lifted_untyped_representation].add_not_effect(
                executed_action.name, bounded_lifted_literal)
            self.lifted_bounded_predicates[executed_action.name][predicate.lifted_untyped_representation].add(
                (bounded_lifted_literal.untyped_representation, bounded_lifted_literal))

    def add_must_be_effect_to_cnf(self, executed_action: ActionCall, grounded_effects: Set[GroundedPredicate]) -> None:
        """Adds an effect that has no ambiguities on which action caused it.

        :param executed_action: the action that caused the effect.
        :param grounded_effects: the grounded predicate that is affected by the action.
        """
        self.logger.info("Adding the effects that contain no ambiguity to the CNF.")
        for grounded_literal in grounded_effects:
            lifted_effect = self.matcher.get_injective_match(grounded_literal, executed_action)
            if lifted_effect is None:
                continue

            self.literals_cnf[grounded_literal.lifted_untyped_representation].add_possible_effect(
                [(executed_action.name, lifted_effect.untyped_representation)])
            domain_predicate = self.partial_domain.predicates[lifted_effect.name]
            self.lifted_bounded_predicates[executed_action.name][domain_predicate.untyped_representation].add(
                (lifted_effect.untyped_representation, lifted_effect))

    def compute_interacting_actions(self, grounded_predicate: GroundedPredicate, executing_actions: List[ActionCall]):
        """Computes the set of actions that interact with a certain predicate.

        :param grounded_predicate: the effect predicate that is being interacted by possibly more than one action.
        :param executing_actions: the actions that are being executed in the joint action.
        :return: the actions that interact with the predicate.
        """
        self.logger.debug(f"Computing the set of actions that interact with predicate "
                          f"{grounded_predicate.untyped_representation}.")
        interacting_actions = []
        for action in executing_actions:
            action_parameters = action.parameters
            action_parameters.extend(self.partial_domain.constants.keys())
            predicate_parameters = set(grounded_predicate.grounded_objects)
            if predicate_parameters.issubset(action_parameters):
                interacting_actions.append(action)

        self.logger.debug(f"The actions {[str(action) for action in interacting_actions]} "
                          f"interact with the predicate {grounded_predicate.untyped_representation}")
        return interacting_actions

    def extract_effects_from_cnf(self, action: LearnerAction, relevant_preconditions: Set[Predicate]) -> None:
        """Extracts the action's relevant effects from the CNF object.

        :param action: the action that is currently being handled.
        :param relevant_preconditions: the preconditions of the action to filter the possible effects from.
        """
        effects = set()
        relevant_preconditions_str = {predicate.untyped_representation for predicate in relevant_preconditions}
        for domain_predicate, cnf in self.literals_cnf.items():
            cnf_effects = cnf.extract_action_effects(action.name)
            for effect in cnf_effects:
                bounded_predicates = [predicate_obj for lifted_representation, predicate_obj in
                                      self.lifted_bounded_predicates[action.name][domain_predicate]
                                      if lifted_representation == effect]
                if len(bounded_predicates) == 0 or bounded_predicates[0].untyped_representation in relevant_preconditions_str:
                    continue

                effects.add(bounded_predicates[0])

        action.add_effects = {predicate for predicate in effects if predicate.is_positive}
        action.delete_effects = {predicate for predicate in effects if not predicate.is_positive}

    def handle_concurrent_execution(
            self, grounded_effect: GroundedPredicate, executing_actions: List[ActionCall]) -> None:
        """Handles the case where effects can be achieved from more than one action.

        :param grounded_effect: the effect that is being targeted by more than one action.
        :param executing_actions: the actions that are part of the joint action.
        """
        self.logger.info("Handling concurrent execution of actions.")
        interacting_actions = self.compute_interacting_actions(grounded_effect, executing_actions)
        if len(interacting_actions) == 1:
            self.add_must_be_effect_to_cnf(interacting_actions[0], {grounded_effect})
            return

        concurrent_execution = []
        for interacting_action in interacting_actions:
            lifted_match = self.matcher.get_injective_match(grounded_effect, interacting_action)
            concurrent_execution.append((interacting_action.name, lifted_match.untyped_representation))
            self.lifted_bounded_predicates[interacting_action.name][grounded_effect.lifted_untyped_representation].add(
                (lifted_match.untyped_representation, lifted_match))

        self.literals_cnf[grounded_effect.lifted_untyped_representation].add_possible_effect(concurrent_execution)

    def update_single_agent_executed_action(
            self, executed_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Handles the situations where only one agent executed an action in a joint action.

        :param executed_action: the single operational action in the joint action.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self.logger.info(f"Handling the execution of the single action - {str(executed_action)}.")
        observed_action = self.partial_domain.actions[executed_action.name]
        if executed_action.name not in self.observed_actions:
            super()._add_new_action_preconditions(executed_action)
            self.observed_actions.append(observed_action.name)

        else:
            super()._update_action_preconditions(executed_action)

        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state,
                                                                     add_predicates_sign=True)
        self.logger.debug("Updating the literals that must be effects of the action.")
        self.add_must_be_effect_to_cnf(executed_action, grounded_add_effects.union(grounded_del_effects))
        not_effects = self._extract_relevant_not_effects(
            in_state_predicates=self.next_state_positive_predicates,
            removed_predicates=self.next_state_negative_predicates, executing_actions=[executed_action],
            relevant_action=executed_action)
        self.add_not_effect_to_cnf(executed_action, not_effects)

    def update_multiple_executed_actions(
            self, joint_action: JointActionCall, previous_state: State, next_state: State) -> None:
        """Handles the case where more than one action is executed in a single trajectory triplet.

        :param joint_action: the joint action that was executed.
        :param previous_state: the state prior to the joint action's execution.
        :param next_state: the state following the joint action's execution.
        """
        self.logger.info("Learning when multiple actions are executed concurrently.")
        executing_actions = joint_action.operational_actions
        for executed_action in executing_actions:
            observed_action = self.partial_domain.actions[executed_action.name]
            self._create_fully_observable_triplet_predicates(executed_action, previous_state, next_state)
            if executed_action.name not in self.observed_actions:
                super()._add_new_action_preconditions(executed_action)
                self.observed_actions.append(observed_action.name)

            else:
                super()._update_action_preconditions(executed_action)

        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state,
                                                                     add_predicates_sign=True)
        for executed_action in executing_actions:
            not_effects = self._extract_relevant_not_effects(
                in_state_predicates=self.next_state_positive_predicates,
                removed_predicates=self.next_state_negative_predicates, executing_actions=[executed_action],
                relevant_action=executed_action)
            self.add_not_effect_to_cnf(executed_action, not_effects)

        for grounded_add_effect in grounded_add_effects.union(grounded_del_effects):
            self.handle_concurrent_execution(grounded_add_effect, executing_actions)

    def handle_multi_agent_trajectory_component(self, component: MultiAgentComponent) -> None:
        """Handles a single multi-agent triplet in the observed trajectory.

        :param component: the triplet to handle.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state

        if joint_action.action_count == 1:
            executing_action = joint_action.operational_actions[0]
            self._create_fully_observable_triplet_predicates(executing_action, previous_state, next_state)
            self.update_single_agent_executed_action(executing_action, previous_state, next_state)
            return

        self.logger.debug("More than one action is being executed in the current triplet.")
        self.update_multiple_executed_actions(joint_action, previous_state, next_state)

    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        super()._remove_unobserved_actions_from_partial_domain()
        for action in self.partial_domain.actions.values():
            self.logger.debug("Constructing safe action for %s", action.name)
            action_preconditions = action.positive_preconditions.union(action.negative_preconditions)
            if not self._is_action_safe(action, action_preconditions):
                self.logger.warning("Action %s is not safe to execute!", action.name)
                continue

            self.logger.debug("Action %s is safe to execute.", action.name)
            self.safe_actions.append(action.name)
            self.extract_effects_from_cnf(action, action_preconditions)

    def learn_combined_action_model(
            self, observations: List[MultiAgentObservation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        super().start_measure_learning_time()
        self._initialize_cnfs()

        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_multi_agent_trajectory_component(component)

        self.construct_safe_actions()
        self.logger.info("Finished learning the action model!")
        super().start_measure_learning_time()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report

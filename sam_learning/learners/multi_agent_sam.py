"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, NOP_ACTION, \
    MultiAgentObservation, ActionCall, State, GroundedPredicate, JointActionCall, PDDLObject

from sam_learning.core import LearnerDomain, extract_effects, LiteralCNF, LearnerAction, \
    create_fully_observable_predicates
from sam_learning.learners import SAMLearner


class MultiAgentSAM(SAMLearner):
    """Class designated to learning action models from multi-agent trajectories with joint actions."""
    logger: logging.Logger
    positive_literals_cnf: Dict[str, LiteralCNF]
    negative_literals_cnf: Dict[str, LiteralCNF]
    preconditions_fluent_map: Dict[str, List[str]]
    concurrency_constraint: int
    lifted_bounded_predicates: Dict[str, Dict[str, Set[Tuple[str, Predicate]]]]
    safe_actions: List[str]

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
                 concurrency_constraint: int = 2):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.positive_literals_cnf = {}
        self.negative_literals_cnf = {}
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.concurrency_constraint = concurrency_constraint
        self.lifted_bounded_predicates = {action_name: defaultdict(set) for action_name in
                                          self.partial_domain.actions.keys()}
        self.safe_actions = []

    def _initialize_cnfs(self) -> None:
        """Initialize the CNFs for the action model."""
        self.logger.debug("Initializing CNFs for the action model.")
        for predicate in self.partial_domain.predicates.values():
            self.positive_literals_cnf[predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))
            self.negative_literals_cnf[predicate.untyped_representation] = \
                LiteralCNF(action_names=list(self.partial_domain.actions.keys()))

    @staticmethod
    def _locate_executing_action(joint_action: JointActionCall) -> List[ActionCall]:
        """Locates the actions that are being executed in the joint action.

        :param joint_action: the joint action to search in.
        :return: the actions that are being executed.
        """
        executing_actions = []
        for action in joint_action.actions:
            if action.name != NOP_ACTION:
                executing_actions.append(action)

        return executing_actions

    def _lift_predicate(self, grounded_predicate: GroundedPredicate, action_call: ActionCall) -> Optional[Predicate]:
        """Lifts a grounded predicate to a lifted predicate.

        :param grounded_predicate: the grounded predicate to lift.
        :param action_call: the action call that the grounded predicate belongs to.
        :return: the lifted predicate if the predicate was lifted, None otherwise.
        """
        self.logger.info("Lifting a single predicate while assuming there is injective matching function.")
        lifted_predicate = self.matcher.match_predicate_to_action_literals(grounded_predicate, action_call)
        if len(lifted_predicate) == 0:
            return None

        if len(lifted_predicate) > 1:
            self.logger.warning("Not supporting non injective matches.")
            return None

        return lifted_predicate[0]

    def _extract_relevant_not_effects(
            self, grounded_predicates: List[GroundedPredicate],
            executing_actions: List[ActionCall], relevant_action: ActionCall) -> List[GroundedPredicate]:
        """Extracts the relevant not effects of the action.

        :param grounded_predicates: the grounded predicates that are affected by the action.
        :param executing_actions: the actions that are being executed in the joint action.
        :param relevant_action: the action that is being learned.
        :return: the relevant not effects of the action.
        """
        return [grounded_predicate for grounded_predicate in grounded_predicates if relevant_action in
                self.compute_interacting_actions(grounded_predicate, executing_actions)]

    def _is_action_safe(self, action: LearnerAction, literals_cnf: Dict[str, LiteralCNF],
                        preconditions_to_filter: Set[Predicate]) -> bool:
        """Checks if the given action is safe to execute.

        :param action: the lifted action that is to be learned.
        :param literals_cnf: the literals CNF containing information about the literals the action affects.
        :return: whether the action is safe to execute.
        """
        for positive_domain_literal, cnf in literals_cnf.items():
            if positive_domain_literal not in self.lifted_bounded_predicates[action.name] or \
                    not cnf.is_action_acting_in_cnf(action.name):
                self.logger.debug(f"The literal {positive_domain_literal} does not relate to {action.name}.")
                continue

            bounded_action_predicates = self.lifted_bounded_predicates[action.name][positive_domain_literal]
            bounded_predicates_str = {predicate_str for predicate_str, _ in bounded_action_predicates}
            bounded_predicates_str.difference_update([p.untyped_representation for p in preconditions_to_filter])
            if len(bounded_predicates_str) == 0:
                continue

            if not cnf.is_action_safe(action_name=action.name, bounded_lifted_predicates=bounded_predicates_str):
                self.logger.debug("Action %s is not safe to execute!", action.name)
                return False

        return True

    def add_not_effect_to_cnf(
            self, executed_action: ActionCall, next_state_predicates: List[GroundedPredicate],
            literals_cnf: Dict[str, LiteralCNF]) -> None:
        """Adds a predicate that cannot be an action's effect to the correct CNF.

        :param executed_action: the action that is being executed in the current joint action triplet.
        :param next_state_predicates: the predicates that are in the state following the action's execution.
        :param literals_cnf: the CNF to add the predicate to.
        """
        for predicate in next_state_predicates:
            bounded_lifted_literal = self._lift_predicate(predicate, executed_action)
            if bounded_lifted_literal is None:
                continue

            literals_cnf[predicate.lifted_untyped_representation].add_not_effect(
                executed_action.name, bounded_lifted_literal)
            self.lifted_bounded_predicates[executed_action.name][predicate.lifted_untyped_representation].add(
                (bounded_lifted_literal.untyped_representation, bounded_lifted_literal))

    def add_must_be_effect_to_cnf(self, executed_action: ActionCall,
                                  grounded_effects: Set[GroundedPredicate],
                                  literals_cnf: Dict[str, LiteralCNF]) -> None:
        """Adds an effect that has no ambiguities on which action caused it.

        :param executed_action: the action that caused the effect.
        :param grounded_effects: the grounded predicate that is affected by the action.
        :param literals_cnf: the CNF representation of the possible effects.
        """
        self.logger.info("Adding the effects that contain no ambiguity to the CNF.")
        for grounded_literal in grounded_effects:
            lifted_effect = self._lift_predicate(grounded_literal, executed_action)
            if lifted_effect is None:
                continue

            literals_cnf[grounded_literal.lifted_untyped_representation].add_possible_effect(
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

    def extract_effects_from_cnf(self, action: LearnerAction, literals_cnf: Dict[str, LiteralCNF],
                                 relevant_preconditions: Set[Predicate]) -> Set[Predicate]:
        """Extracts the action's relevant effects from the CNF object.

        :param action: the action that is currently being handled.
        :param literals_cnf: the CNF dictionary containing information about the effects.
        :param relevant_preconditions: the preconditions of the action to filter the possible effects from.
        :return: the lifted bounded predicates.
        """
        effects = set()
        relevant_preconditions_str = {predicate.untyped_representation for predicate in relevant_preconditions}
        for domain_predicate, cnf in literals_cnf.items():
            cnf_effects = cnf.extract_action_effects(action.name)
            for effect in cnf_effects:
                bounded_predicates = [predicate_obj for lifted_representation, predicate_obj in
                                      self.lifted_bounded_predicates[action.name][domain_predicate]
                                      if lifted_representation == effect]
                if bounded_predicates[0].untyped_representation in relevant_preconditions_str:
                    continue

                effects.add(bounded_predicates[0])

        return effects

    def handle_concurrent_execution(
            self, grounded_effect: GroundedPredicate,
            executing_actions: List[ActionCall], literals_cnf: Dict[str, LiteralCNF]) -> None:
        """Handles the case where effects can be achieved from more than one action.

        :param grounded_effect: the effect that is being targeted by more than one action.
        :param executing_actions: the actions that are part of the joint action.
        :param literals_cnf: the literals CNF to add the effect to.
        """
        self.logger.info("Handling concurrent execution of actions.")
        interacting_actions = self.compute_interacting_actions(grounded_effect, executing_actions)
        if len(interacting_actions) == 1:
            self.add_must_be_effect_to_cnf(interacting_actions[0], {grounded_effect}, literals_cnf)
            return

        concurrent_execution = []
        for interacting_action in interacting_actions:
            lifted_match = self._lift_predicate(grounded_effect, interacting_action)
            concurrent_execution.append((interacting_action.name, lifted_match.untyped_representation))
            self.lifted_bounded_predicates[interacting_action.name][grounded_effect.lifted_untyped_representation].add(
                (lifted_match.untyped_representation, lifted_match))

        literals_cnf[grounded_effect.lifted_untyped_representation].add_possible_effect(concurrent_execution)

    def update_single_agent_executed_action(
            self, executed_action: ActionCall, previous_state: State, next_state: State,
            negative_state_predicates: Set[GroundedPredicate]) -> None:
        """Handles the situations where only one agent executed an action in a joint action.

        :param executed_action: the single operational action in the joint action.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        :param negative_state_predicates: the negative state predicates.
        """
        self.logger.info(f"Handling the execution of {str(executed_action)}.")
        positive_next_state_predicates, negative_next_state_predicates = \
            create_fully_observable_predicates(next_state, negative_state_predicates)
        observed_action = self.partial_domain.actions[executed_action.name]
        if executed_action.name not in self.observed_actions:
            super()._add_new_action_preconditions(executed_action, observed_action,
                                                  negative_state_predicates, previous_state)
            self.observed_actions.append(observed_action.name)

        else:
            super()._update_action_preconditions(executed_action, previous_state)

        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        self.logger.debug("Updating the negative state predicates based on the action's execution.")
        self.add_must_be_effect_to_cnf(executed_action, grounded_add_effects, self.positive_literals_cnf)
        self.add_must_be_effect_to_cnf(executed_action, grounded_del_effects, self.negative_literals_cnf)

        relevant_not_add_effects = self._extract_relevant_not_effects(
            negative_next_state_predicates, [executed_action], executed_action)
        relevant_not_del_effects = self._extract_relevant_not_effects(
            positive_next_state_predicates, [executed_action], executed_action)

        if len(relevant_not_add_effects) > 0:
            self.add_not_effect_to_cnf(executed_action, relevant_not_add_effects, self.positive_literals_cnf)

        if len(relevant_not_del_effects) > 0:
            self.add_not_effect_to_cnf(executed_action, relevant_not_del_effects, self.negative_literals_cnf)

    def update_multiple_executed_actions(
            self, joint_action: JointActionCall, previous_state: State, next_state: State,
            negative_state_predicates: Set[GroundedPredicate]) -> None:
        """Handles the case where more than one action is executed in a single trajectory triplet.

        :param joint_action: the joint action that was executed.
        :param previous_state: the state prior to the joint action's execution.
        :param next_state: the state following the joint action's execution.
        :param negative_state_predicates: the negative state predicates.
        """
        self.logger.info("Learning when multiple actions are executed concurrently.")
        executing_actions = self._locate_executing_action(joint_action)
        for executed_action in executing_actions:
            observed_action = self.partial_domain.actions[executed_action.name]
            if executed_action.name not in self.observed_actions:
                super()._add_new_action_preconditions(executed_action, observed_action, negative_state_predicates,
                                                      previous_state)
                self.observed_actions.append(observed_action.name)

            else:
                super()._update_action_preconditions(executed_action, previous_state)

        positive_next_state_predicates, negative_next_state_predicates = \
            create_fully_observable_predicates(next_state, negative_state_predicates)
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        self.logger.debug("Updating the negative state predicates based on the action's execution.")
        for executed_action in executing_actions:
            relevant_not_add_effects = self._extract_relevant_not_effects(
                negative_next_state_predicates, executing_actions, executed_action)
            relevant_not_del_effects = self._extract_relevant_not_effects(
                positive_next_state_predicates, executing_actions, executed_action)

            if len(relevant_not_add_effects) > 0:
                self.add_not_effect_to_cnf(executed_action, relevant_not_add_effects, self.positive_literals_cnf)
            if len(relevant_not_del_effects) > 0:
                self.add_not_effect_to_cnf(executed_action, relevant_not_del_effects, self.negative_literals_cnf)

        for grounded_add_effect in grounded_add_effects:
            self.handle_concurrent_execution(grounded_add_effect, executing_actions, self.positive_literals_cnf)

        for grounded_del_effect in grounded_del_effects:
            self.handle_concurrent_execution(grounded_del_effect, executing_actions, self.negative_literals_cnf)

    def handle_multi_agent_trajectory_component(
            self, component: MultiAgentComponent, trajectory_objects: Dict[str, PDDLObject]) -> None:
        """Handles a single multi-agent triplet in the observed trajectory.

        :param component: the triplet to handle.
        :param trajectory_objects: the objects observed in the current trakectory.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state
        _, negative_state_predicates = super()._create_complete_world_state(trajectory_objects, previous_state)

        action_count = joint_action.action_count
        if action_count == 1:
            executing_action = self._locate_executing_action(joint_action)[0]
            self.update_single_agent_executed_action(executing_action, previous_state, next_state,
                                                     negative_state_predicates)
            return

        self.logger.debug("More than one action is being executed in the current triplet.")
        self.update_multiple_executed_actions(joint_action, previous_state, next_state,
                                              negative_state_predicates)

    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        for action in self.partial_domain.actions.values():
            if action.name not in self.observed_actions:
                continue

            self.logger.debug("Constructing safe action for %s", action.name)
            if not self._is_action_safe(action, self.positive_literals_cnf, action.positive_preconditions) or not \
                    self._is_action_safe(action, self.negative_literals_cnf, action.negative_preconditions):
                self.logger.warning("Action %s is not safe to execute!", action.name)
                action.positive_preconditions = set()
                action.negative_preconditions = set()
                continue

            self.logger.debug("Action %s is safe to execute.", action.name)
            action.add_effects = self.extract_effects_from_cnf(action, self.positive_literals_cnf,
                                                               action.positive_preconditions)
            action.delete_effects = self.extract_effects_from_cnf(action, self.negative_literals_cnf,
                                                                  action.negative_preconditions)
            self.safe_actions.append(action.name)

    def learn_combined_action_model(
            self, observations: List[MultiAgentObservation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self._initialize_cnfs()
        super().deduce_initial_inequality_preconditions()
        for index, observation in enumerate(observations):
            trajectory_objects = observation.grounded_objects
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_multi_agent_trajectory_component(component, trajectory_objects)

        self.construct_safe_actions()
        self.logger.info("Finished learning the action model!")
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report

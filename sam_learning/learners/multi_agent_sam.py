"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
from collections import defaultdict
from typing import Dict, List, NoReturn, Tuple, Set, Optional

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, PDDLObject, NOP_ACTION, \
    MultiAgentObservation, ActionCall, State, GroundedPredicate

from sam_learning.core import LearnerDomain, extract_effects, MultiActionPredicateMatching
from sam_learning.learners import SAMLearner


class MultiAgentSAM(SAMLearner):
    """Class designated to learning action models from multi-agent trajectories with joint actions."""
    logger: logging.Logger
    might_be_add_effects: Dict[str, Set[Predicate]]
    might_be_delete_effects: Dict[str, Set[Predicate]]
    preconditions_fluent_map: Dict[str, List[str]]
    joint_actions: Dict[str, MultiActionPredicateMatching]
    concurrency_constraint: int

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
                 concurrency_constraint: int = 2):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.observed_actions = []
        self.might_be_add_effects = defaultdict(set)
        self.might_be_delete_effects = defaultdict(set)
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.joint_actions = {}
        self.concurrency_constraint = concurrency_constraint

    @staticmethod
    def _update_might_be_effects(action_might_be_effects: Set[Predicate], new_effects: List[Predicate]):
        """Update the action's effects on the cases where it was not the only action that was executed.

        :param action_might_be_effects: the effects of the action that was observed up to this point.
        :param new_effects: the new effects to be added to the action.
        """
        if len(action_might_be_effects) > 0:
            intersection_add_effects = action_might_be_effects.intersection(new_effects)
            new_observed_effects = set(new_effects).difference(intersection_add_effects)
            return intersection_add_effects.union(new_observed_effects)

        return set(new_effects)

    def _create_negative_grounded_predicates(
            self, state: State, observed_objects: Dict[str, PDDLObject]) -> Set[GroundedPredicate]:
        """

        :param state:
        :param observed_objects:
        :return:
        """
        possible_negative_predicates = set()
        vocabulary = self.vocabulary_creator.create_vocabulary(self.partial_domain, observed_objects)
        for lifted_predicate_name, grounded_missing_predicates in vocabulary.items():
            if lifted_predicate_name not in state.state_predicates:
                possible_negative_predicates.update(grounded_missing_predicates)
                continue

            possible_negative_predicates.update([predicate for predicate in grounded_missing_predicates if predicate
                                                 not in state.state_predicates[lifted_predicate_name]])

        return possible_negative_predicates

    def _add_positive_predicates(self, grounded_action: ActionCall, state: State) -> Set[Predicate]:
        """Adds positice predicates that are observed in the state.

        :param grounded_action: the action that was encountered.
        :param state: the state belonging to the currently observed triplet.
        :return: the lifted predicates that were observed in the state.
        """
        possible_positive_predicates = set()
        for predicates in state.state_predicates.values():
            lifted_matches = self.matcher.get_possible_literal_matches(
                grounded_action, list(predicates))
            possible_positive_predicates.update(lifted_matches)

        return possible_positive_predicates

    def _handle_single_agent_action_effects(self, grounded_action: ActionCall, previous_state: State,
                                            next_state: State, number_operational_actions: int) -> NoReturn:
        """Handles the effects of a single agent action executed in a joint action.

        :param grounded_action: the single agent action that was executed.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous state.
        :param number_operational_actions: the number of actions executed by different agents in the current triplet
            different from NOP.
        """
        self.logger.debug(f"Starting to learn the effects of {str(grounded_action)}.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))

        if number_operational_actions == 1:
            self.logger.debug(f"The action {grounded_action.name} is the only operational action in the trajectory.")
            self.must_be_add_effects[grounded_action.name].update(lifted_add_effects)
            self.must_be_delete_effects[grounded_action.name].update(lifted_delete_effects)
            return

        self.might_be_add_effects[grounded_action.name] = self._update_might_be_effects(
            self.might_be_add_effects[grounded_action.name], lifted_add_effects)
        self.might_be_delete_effects[grounded_action.name] = self._update_might_be_effects(
            self.might_be_delete_effects[grounded_action.name], lifted_delete_effects)

    def _update_actions_must_be_effects(self) -> NoReturn:
        """Updates the literals that positively belong to the action's effects."""
        for action_name, must_be_effects in self.must_be_add_effects.items():
            self.logger.debug(f"Adding the must be add effects for action - {action_name}")
            self.partial_domain.actions[action_name].add_effects = must_be_effects

        for action_name, must_be_effects in self.must_be_delete_effects.items():
            self.logger.debug(f"Adding the must be delete effects for action - {action_name}")
            self.partial_domain.actions[action_name].delete_effects = must_be_effects

    def _update_actions_might_be_effects(self) -> NoReturn:
        """Updates the literals that might be effects of the single agent actions."""
        for action_name, might_be_effect in self.might_be_add_effects.items():
            self.logger.debug(f"Adding the must be add effects for action - {action_name}")
            self.partial_domain.actions[action_name].add_effects.update(might_be_effect)
            self.partial_domain.actions[action_name].add_effects.difference_update(
                self.must_not_be_add_effect[action_name])

        for action_name, might_be_effect in self.might_be_delete_effects.items():
            self.logger.debug(f"Adding the must be delete effects for action - {action_name}")
            self.partial_domain.actions[action_name].delete_effects.update(might_be_effect)
            self.partial_domain.actions[action_name].delete_effects.difference_update(
                self.must_not_be_delete_effect[action_name])

    def add_new_single_agent_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State,
            observed_objects: Dict[str, PDDLObject], num_operational_actions: int) -> NoReturn:
        """Adds a new single agent action to the domain.

        :param grounded_action: the single agent action that was executed.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the joint action on the previous state.
        :param observed_objects: the objects that were observed in the trajectory.
        :param num_operational_actions: the number of actions executed by the agents in the current triplet,
            different from NOP.
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action, observed_action, observed_objects, previous_state)

        self._handle_single_agent_action_effects(grounded_action, previous_state, next_state, num_operational_actions)
        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_single_agent_action(self, grounded_action: ActionCall, previous_state: State, next_state: State,
                                   num_operational_actions: int, observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """Updates the single agents actions preconditions and effects based on the observation's data.

        :param grounded_action: the single agent action that was executed.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the joint action on the previous state.
        :param observed_objects: the objects that were observed in the trajectory.
        :param num_operational_actions: the number of actions executed by the agents in the current triplet,
            different from NOP.
        """
        self.logger.info(f"Updating the action {grounded_action.name}.")
        self._update_action_preconditions(grounded_action, previous_state)

        self._handle_single_agent_action_effects(grounded_action, previous_state, next_state, num_operational_actions)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def update_one_executed_action(self, executed_action: ActionCall, previous_state: State, next_state: State,
                                   observed_objects: Dict[str, PDDLObject]) -> NoReturn:
        """

        :param executed_action:
        :param previous_state:
        :param next_state:
        :param observed_objects:
        :return:
        """
        negative_pre_state_predicates = super()._add_negative_predicates(executed_action, previous_state,
                                                                         observed_objects)
        pre_state_predicates = set()
        for positive_predicates in previous_state.state_predicates.values():
            pre_state_predicates.update(positive_predicates)

        pre_state_predicates.update(negative_pre_state_predicates)
        observed_action = self.partial_domain.actions[executed_action.name]
        if executed_action.name not in self.observed_actions:
            super().add_new_action(executed_action, previous_state, next_state, observed_objects)
            return

        super()._update_action_preconditions(executed_action, previous_state)
        # now we observe the action again, we need to clear the action's ambiguities
        negative_state_predicates = self._add_negative_predicates(executed_action, next_state, observed_objects)
        lifted_add_effects, lifted_delete_effects = super()._handle_action_effects(
            executed_action, previous_state, next_state)
        observed_action.add_effects.update(lifted_add_effects)
        observed_action.delete_effects.update(lifted_delete_effects)

        add_effects_to_remove = set(lifted_add_effects).union(negative_state_predicates)
        delete_effects_to_remove = set(lifted_delete_effects).union(
            self._add_positive_predicates(executed_action, next_state))
        for effect_to_remove in add_effects_to_remove:
            self.might_be_add_effects[observed_action.name].discard(effect_to_remove)

        for delete_effect in delete_effects_to_remove:
            self.might_be_delete_effects[observed_action.name].discard(delete_effect)

    def update_multiple_executed_actions(
            self, joint_actions: List[ActionCall], previous_state: State, next_state: State,
            objects: Dict[str, PDDLObject], agents: List[str]) -> NoReturn:
        """

        :param joint_actions:
        :param previous_state:
        :param next_state:
        :param objects:
        :param agents:
        :return:
        """
        for agent_name, single_agent_action in zip(agents, joint_actions):
            if single_agent_action.name == NOP_ACTION:
                self.logger.debug(f"The agent {agent_name} did not perform an action in this joint action triplet.")
                continue

            self._update_action_preconditions(single_agent_action, previous_state)

        positive_next_state_predicates = set()
        for positive_predicates in next_state.state_predicates.values():
            positive_next_state_predicates.update(positive_predicates)

        negative_state_predicates = self._create_negative_grounded_predicates(next_state, objects)
        # calculate action(l) for each literal in the post state - first for the positive literals and then the negative
        for literal in positive_next_state_predicates:
            if not self.is_add_effect(previous_state, literal):
                continue

            interacting_actions = self._get_interacting_actions(literal, joint_actions)
            if len(interacting_actions) == 0:
                self.logger.debug(f"No action interacts with the literal {literal.untyped_representation}.")
                continue

            if len(interacting_actions) == 1:
                interacting_action = self.partial_domain.actions[interacting_actions[0].name]
                interacting_action.add_effects.add(literal)
                continue

            self.logger.debug(f"More than one action is interacting with - {literal.untyped_representation}!")
            joint_action_name = ",".join([action.name for action in interacting_actions])
            domain_predicate = self.partial_domain.predicates[literal.name]
            literal_constraints = {domain_predicate: [self.matcher.match_predicate_to_action_literals(literal, action)
                                   for action in interacting_actions]}
            if joint_action_name not in self.joint_actions:
                multi_action_predicate_matching = MultiActionPredicateMatching()
                multi_action_predicate_matching.positive_predicates.append(literal_constraints)
                self.joint_actions[joint_action_name] = multi_action_predicate_matching

            else:
                self.joint_actions[joint_action_name].positive_predicates.append(literal_constraints)


    def handle_multi_agent_trajectory_component(
            self, component: MultiAgentComponent, objects: Dict[str, PDDLObject], agents: List[str]) -> NoReturn:
        """Handles a single multi agent triplet in the observed trajectory.

        :param component: the triplet to handle.
        :param objects: the objects that were observed in the trajectory.
        :param agents: the names of the agents that participate in the actions.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state

        action_count = joint_action.action_count
        if action_count == 1:
            executing_action = None
            self.logger.debug(f"The trajectory component contains a single action!")
            for agent_name, single_agent_action in zip(agents, joint_action.actions):
                if single_agent_action.name != NOP_ACTION:
                    executing_action = single_agent_action
                    break

            self.update_one_executed_action(executing_action, previous_state, next_state, objects)
            return

        self.logger.debug("More than one action is being executed in the current triplet.")
        self.update_multiple_executed_actions(joint_action.actions, previous_state, next_state, objects, agents)

    def learn_combined_action_model(self, observations: List[MultiAgentObservation],
                                    agent_names: List[str]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :param agent_names: the names of the agents that interact with the environment.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        for observation in observations:
            for component in observation.components:
                self.handle_multi_agent_trajectory_component(component, observation.grounded_objects, agent_names)

        self._update_actions_must_be_effects()
        if self.concurrency_constraint == 1:
            self._update_actions_might_be_effects()

        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

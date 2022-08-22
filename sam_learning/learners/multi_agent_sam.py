"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
from collections import defaultdict
from typing import Dict, List, NoReturn, Tuple, Set

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, PDDLObject, NOP_ACTION, \
    MultiAgentObservation, ActionCall, State

from sam_learning.core import LearnerDomain, extract_effects
from sam_learning.learners import SAMLearner


class MultiAgentSAM(SAMLearner):
    logger: logging.Logger
    must_be_add_effects: Dict[str, Set[Predicate]]
    must_be_delete_effects: Dict[str, Set[Predicate]]
    might_be_add_effects: Dict[str, Set[Predicate]]
    might_be_delete_effects: Dict[str, Set[Predicate]]

    def __init__(self, partial_domain: Domain):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.observed_actions = []
        self.must_be_add_effects = defaultdict(set)
        self.must_be_delete_effects = defaultdict(set)
        self.might_be_add_effects = defaultdict(set)
        self.might_be_delete_effects = defaultdict(set)

    @staticmethod
    def _update_might_be_effects(action_might_be_effects: Set[Predicate], new_effects: List[Predicate]):
        """Update the effects of the action that might be observed.

        :param action_might_be_effects: the effects of the action that might be observed.
        :param new_effects: the new effects to be added to the action.
        """
        if len(action_might_be_effects) > 0:
            intersection_add_effects = action_might_be_effects.intersection(new_effects)
            new_observed_effects = set(new_effects).difference(intersection_add_effects)
            return intersection_add_effects.union(new_observed_effects)

        return set(new_effects)

    def _handle_single_agent_action_effects(self, grounded_action: ActionCall, previous_state: State,
                                            next_state: State, number_operational_actions: int) -> NoReturn:
        """

        :param grounded_action:
        :param previous_state:
        :param next_state:
        :param number_operational_actions:
        :return:
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
        """

        :return:
        """
        for action_name, must_be_effects in self.must_be_add_effects.items():
            self.logger.debug(f"Adding the must be add effects for action - {action_name}")
            self.partial_domain.actions[action_name].add_effects = must_be_effects

        for action_name, must_be_effects in self.must_be_delete_effects.items():
            self.logger.debug(f"Adding the must be delete effects for action - {action_name}")
            self.partial_domain.actions[action_name].delete_effects = must_be_effects

    def add_new_single_agent_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State,
            observed_objects: Dict[str, PDDLObject], num_operational_actions: int) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        :param observed_objects: the objects that were observed in the current trajectory.
            state.
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        possible_preconditions = set()
        negative_predicates = self._add_negative_predicates(grounded_action, previous_state, observed_objects)

        for lifted_predicate_name, grounded_state_predicates in previous_state.state_predicates.items():
            self.logger.debug(f"trying to match the predicate - {lifted_predicate_name} "
                              f"to the action call - {str(grounded_action)}")
            lifted_matches = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_state_predicates))
            possible_preconditions.update(lifted_matches)

        observed_action.positive_preconditions.update(possible_preconditions)
        observed_action.negative_preconditions.update(negative_predicates)
        self._handle_single_agent_action_effects(grounded_action, previous_state, next_state, num_operational_actions)

        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_single_agent_action(self, grounded_action: ActionCall, previous_state: State, next_state: State,
                                   num_operational_actions: int) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        self.logger.info(f"Updating the action {grounded_action.name}.")
        self._update_action_preconditions(grounded_action, previous_state)
        self._handle_single_agent_action_effects(grounded_action, previous_state, next_state, num_operational_actions)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def handle_multi_agent_trajectory_component(
            self, component: MultiAgentComponent, objects: Dict[str, PDDLObject], agents: List[str]) -> NoReturn:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        :param objects: the objects that were observed in the current trajectory.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state

        number_operational_actions = len([action for action in joint_action.actions if action.name != NOP_ACTION])
        for agent_name, single_agent_action in zip(agents, joint_action.actions):
            if single_agent_action.name == NOP_ACTION:
                self.logger.debug(f"The agent {agent_name} did not perform an action in this joint action triplet.")
                continue

            if self._verify_parameter_duplication(single_agent_action):
                self.logger.warning(f"{str(single_agent_action)} contains duplicated parameters!")
                continue

            if single_agent_action.name not in self.observed_actions:
                self.logger.debug(f"Agent - {agent_name} performed the action - {single_agent_action.name} "
                                  f"which was not observed yet.")
                self.add_new_single_agent_action(single_agent_action, previous_state, next_state, objects,
                                                 number_operational_actions)

            else:
                self.logger.debug(f"Agent - {agent_name} performed the action - {single_agent_action.name}. "
                                  f"Updating the action's data.")
                self.update_single_agent_action(single_agent_action, previous_state, next_state,
                                                number_operational_actions)

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
        learning_report = {action_name: "OK" for action_name in self.partial_domain.actions}
        return self.partial_domain, learning_report

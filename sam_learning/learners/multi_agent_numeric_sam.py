"""basic version of the combination of MA-SAM an N-SAM together."""
import logging
from typing import Dict, List, Tuple, Optional

from pddl_plus_parser.models import MultiAgentComponent, MultiAgentObservation, ActionCall, State, \
    JointActionCall, Domain

from sam_learning.core import LearnerDomain, NotSafeActionError
from sam_learning.learners.numeric_sam import PolynomialSAMLearning


class NumericMultiAgentSAM(PolynomialSAMLearning):
    """Class designated to learning action models from multi-agent trajectories with joint actions."""
    logger: logging.Logger
    safe_actions: List[str]
    agent_names: List[str]

    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
                 polynomial_degree: int = 1, agent_names: List[str] = None, **kwargs):
        super().__init__(partial_domain, preconditions_fluent_map, polynomial_degree, **kwargs)
        self.agent_names = agent_names if agent_names else []

    def are_actions_independent(self, executing_actions: List[ActionCall]) -> bool:
        """Determines whether the actions in a joint action are independent.

        :param executing_actions: the actions that are being executed in the joint action.
        :return: whether the actions are independent.
        """
        self.logger.debug("Computing the set of actions can possibly interact with one another.")
        for action in executing_actions:
            action_parameters = action.parameters
            for other_action in executing_actions:
                if action == other_action:
                    continue

                other_action_parameters = other_action.parameters
                if set(action_parameters).intersection(other_action_parameters):
                    return False
        return True

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
            super().add_new_action(executed_action, previous_state, next_state)
            self.observed_actions.append(observed_action.name)

        else:
            super().update_action(executed_action, previous_state, next_state)

    def update_multiple_executed_actions(
            self, joint_action: JointActionCall, previous_state: State, next_state: State) -> None:
        """Handles the case where more than one action is executed in a single trajectory triplet.

        :param joint_action: the joint action that was executed.
        :param previous_state: the state prior to the joint action's execution.
        :param next_state: the state following the joint action's execution.
        """
        self.logger.info("Learning when multiple actions are executed concurrently.")
        executing_actions = joint_action.operational_actions
        if not self.are_actions_independent(executing_actions):
            return

        for index, executed_action in enumerate(executing_actions):
            self.triplet_snapshot.create_triplet_snapshot(
                previous_state=previous_state, next_state=next_state, current_action=executed_action,
                observation_objects=self.current_trajectory_objects)

            observed_action = self.partial_domain.actions[executed_action.name]
            if executed_action.name not in self.observed_actions:
                super().add_new_action(executed_action, previous_state, next_state)
                self.observed_actions.append(observed_action.name)

            else:
                super().update_action(executed_action, previous_state, next_state)

    def handle_multi_agent_trajectory_component(self, component: MultiAgentComponent) -> None:
        """Handles a single multi-agent triplet in the observed trajectory.

        :param component: the triplet to handle.
        """
        previous_state = component.previous_state
        joint_action = component.grounded_joint_action
        next_state = component.next_state

        if joint_action.action_count == 1:
            executing_action = joint_action.operational_actions[0]
            self.triplet_snapshot.create_triplet_snapshot(
                previous_state=previous_state, next_state=next_state, current_action=executing_action,
                observation_objects=self.current_trajectory_objects)
            self.update_single_agent_executed_action(executing_action, previous_state, next_state)
            return

        self.logger.debug("More than one action is being executed in the current triplet.")
        self.update_multiple_executed_actions(joint_action, previous_state, next_state)

    def learn_action_model(self, observations: List[MultiAgentObservation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input observations.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned and the metadata about the learning.
        """
        self.logger.info("Starting to learn the action model!")
        super().start_measure_learning_time()
        allowed_actions = {}
        learning_metadata = {}
        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                if not super().are_states_different(component.previous_state, component.next_state):
                    continue

                self.handle_multi_agent_trajectory_component(component)

        for action_name, action in self.partial_domain.actions.items():
            if action_name not in self.storage:
                self.logger.debug(f"The action - {action_name} was not observed in the trajectories!")
                continue

            self.storage[action_name].filter_out_inconsistent_state_variables()
            try:
                self._construct_safe_numeric_preconditions(action)
                self._construct_safe_numeric_effects(action)
                allowed_actions[action_name] = action
                learning_metadata[action_name] = "OK"
                self.logger.info(f"Done learning the action - {action_name}!")

            except NotSafeActionError as e:
                self.logger.debug(f"The action - {e.action_name} is not safe for execution, reason - {e.reason}")
                learning_metadata[action_name] = e.solution_type.name

        self.partial_domain.actions = allowed_actions

        super().end_measure_learning_time()
        learning_metadata["learning_time"] = str(self.learning_end_time - self.learning_start_time)
        return self.partial_domain, learning_metadata

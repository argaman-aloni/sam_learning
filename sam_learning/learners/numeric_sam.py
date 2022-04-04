"""Extension to SAM Learning that can learn numeric state variables."""

from typing import List, NoReturn, Dict

from pddl_plus_parser.models import Observation, ActionCall, State, Domain

from sam_learning.core import LearnerDomain, NumericFluentStateStorage
from .sam_learning import SAMLearner


class NumericSAMLearner(SAMLearner):
    """The Extension of SAM that is able to learn numeric state variables."""

    storage: Dict[str, NumericFluentStateStorage]

    def __init__(self, partial_domain: Domain):
        super().__init__(partial_domain)
        self.storage = {}

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        super(NumericSAMLearner, self).add_new_action(grounded_action, previous_state, next_state)
        self.storage[grounded_action.name] = NumericFluentStateStorage(grounded_action.name)
        self.storage[grounded_action.name].add_to_previous_state_storage(previous_state.state_fluents)
        self.storage[grounded_action.name].add_to_next_state_storage(next_state.state_fluents)
        self.logger.debug(f"Done creating the numeric state variable storage for the action - {grounded_action.name}")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> NoReturn:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        action_name = grounded_action.name
        super(NumericSAMLearner, self).update_action(grounded_action, previous_state, next_state)
        self.storage[action_name] = NumericFluentStateStorage(action_name)
        self.storage[action_name].add_to_previous_state_storage(previous_state.state_fluents)
        self.storage[action_name].add_to_next_state_storage(next_state.state_fluents)
        self.logger.debug(f"Done updating the numeric state variable storage for the action - {grounded_action.name}")

    def learn_action_model(self, observations: List[Observation]) -> LearnerDomain:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        for observation in observations:
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        for action_name, action in self.partial_domain.actions.items():
            action.numeric_preconditions = self.storage[action_name].construct_safe_linear_inequalities()
            action.numeric_effects = self.storage[action_name].construct_assignment_equations()

        return self.partial_domain

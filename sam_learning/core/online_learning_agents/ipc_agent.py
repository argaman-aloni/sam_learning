"""An agent for the active learning of IPC models"""
import logging
from typing import Dict, Set, Tuple

from pddl_plus_parser.models import State, Domain, ActionCall, Problem, Operator, evaluate_expression, PDDLFunction, NumericalExpressionTree

from sam_learning.core import VocabularyCreator
from sam_learning.core.online_learning_agents.abstract_agent import AbstractAgent


class IPCAgent(AbstractAgent):
    _domain: Domain
    _problem: Problem
    _vocabulary_creator: VocabularyCreator
    logger: logging.Logger

    def __init__(self, domain: Domain, problem: Problem):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._reward = 0
        self._domain = domain
        self._problem = problem
        self._vocabulary_creator = VocabularyCreator()

    def _assign_state_fluent_value(self, state_fluents: Dict[str, PDDLFunction], goal_required_expressions: Set[NumericalExpressionTree]) -> None:
        """Assigns the values of the state fluents to later verify if the goal was achieved.

        :param state_fluents: the state fluents to be assigned to the goal expressions.
        :param goal_required_expressions: the goal expressions that need to be evaluated
            (not containing actual values).
        """
        self.logger.info("Assigning values to state fluents.")
        for goal_expression in goal_required_expressions:
            expression_functions = [func.value for func in goal_expression if isinstance(func.value, PDDLFunction)]
            for state_fluent in state_fluents.values():
                for expression_function in expression_functions:
                    if state_fluent.untyped_representation == expression_function.untyped_representation:
                        expression_function.set_value(state_fluent.value)

    def _get_reward(self, previous_state: State, next_state: State) -> int:
        """Evaluates the reward for advancing from the previous state to the next state.

        :param previous_state: the previous state.
        :param next_state: the next state.
        :return: the reward for advancing from the previous state to the next state.
        """
        self.logger.debug(f"Checking if the previous state {previous_state} " f"is different from the next state {next_state}")
        if previous_state == next_state:
            self.logger.warning("The previous state is the same as the next state. ")
            return -1

        return 1

    def get_environment_actions(self, state: State) -> Set[ActionCall]:
        """Returns the set of the actions that are legal in the environment in the current state.

        Note:
            This function was added to handle cases where the modeling does not accurately represent the environment.
            In these cases the agent will have to inform the online learning of what actions are actually legal in the
            environment.

        :param state:  the state that the agent is currently in.
        :return: the set of actions that are legal in the environment in the current state.
        """
        self.logger.info("Creating all the grounded actions for the domain given the current possible objects.")
        grounded_action_calls = self._vocabulary_creator.create_grounded_actions_vocabulary(
            domain=self._domain, observed_objects=self._problem.objects
        )
        return grounded_action_calls

    def observe(self, state: State, action: ActionCall) -> Tuple[State, int]:
        """Observes an action being executed on the state and the resulting new state of the environment.

        :param state: the state before the action was executed.
        :param action: the action that was executed.
        :return: the state after the action was executed.
        """
        self.logger.debug("Observing the action %s being executed on the state %s.", str(action), state.serialize())
        operator = Operator(
            action=self._domain.actions[action.name],
            domain=self._domain,
            grounded_action_call=action.parameters,
            problem_objects=self._problem.objects,
        )
        try:
            new_state = operator.apply(state)

        except ValueError:
            self.logger.debug(f"Could not apply the action {str(operator)} to the state.")
            new_state = state.copy()

        reward = self._get_reward(state, new_state)
        return new_state, reward

    def goal_reached(self, state: State) -> bool:
        """Evaluates whether the goal of the PDDL problem has been reached.

        :param state: the state to be evaluated.
        :return: whether the goal has been reached.
        """
        self.logger.info("Evaluating the reward for the state %s.", state.serialize())
        goal_predicates = {p.untyped_representation for p in self._problem.goal_state_predicates}
        goal_fluents = self._problem.goal_state_fluents

        state_predicates = set()
        for grounded_predicates in state.state_predicates.values():
            state_predicates.update([p.untyped_representation for p in grounded_predicates])

        self._assign_state_fluent_value(state.state_fluents, goal_fluents)

        if goal_predicates.issubset(state_predicates) and all([evaluate_expression(fluent.root) for fluent in goal_fluents]):
            self.logger.info("The IPC agent has reached the goal state.")
            return True

        self.logger.debug("Goal has not been reached according to the IPC agent.")
        return False

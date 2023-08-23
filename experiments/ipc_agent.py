"""An agent for the active learning of IPC models"""
import logging
from typing import Dict, Set

from pddl_plus_parser.models import State, Domain, ActionCall, Problem, Operator, evaluate_expression, PDDLFunction, \
    NumericalExpressionTree

from sam_learning.core import AbstractAgent


class IPCAgent(AbstractAgent):
    _domain: Domain
    _problem: Problem
    logger: logging.Logger

    def __init__(self, domain: Domain, problem: Problem):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._reward = 0
        self._domain = domain
        self._problem = problem

    def _assign_state_fluent_value(self, state_fluents: Dict[str, PDDLFunction],
                                   goal_required_expressions: Set[NumericalExpressionTree]) -> None:
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

    def observe(self, state: State, action: ActionCall) -> State:
        """Observes an action being executed on the state and the resulting new state of the environment.

        :param state: the state before the action was executed.
        :param action: the action that was executed.
        :return: the state after the action was executed.
        """
        self.logger.debug(
            "Observing the action %s being executed on the state %s.", str(action), state.serialize())
        operator = Operator(action=self._domain.actions[action.name], domain=self._domain,
                            grounded_action_call=action.parameters, problem_objects=self._problem.objects)
        try:
            return operator.apply(state)

        except ValueError:
            self.logger.debug(f"Could not apply the action {str(operator)} to the state.")
            return state.copy()

    def get_reward(self, state: State) -> float:
        """Evaluates whether the agent has reached the goal state and returns the reward.

        Note:
            At the moment the reward is binary on whether the agent has reached the goal state or not.

        :param state: the state to be evaluated.
        :return: the reward for the state.
        """
        self.logger.info("Evaluating the reward for the state %s.", state.serialize())
        goal_predicates = {p.untyped_representation for p in self._problem.goal_state_predicates}
        goal_fluents = self._problem.goal_state_fluents

        state_predicates = set()
        for grounded_predicates in state.state_predicates.values():
            state_predicates.update([p.untyped_representation for p in grounded_predicates])

        self._assign_state_fluent_value(state.state_fluents, goal_fluents)

        if (goal_predicates.issubset(state_predicates) and
                all([evaluate_expression(fluent.root) for fluent in goal_fluents])):
            self.logger.info("The agent has reached the goal state.")
            return 1

        self.logger.debug("Goal has not been reached.")
        return 0

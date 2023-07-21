"""An online version of the Numeric SAM learner."""
from pddl_plus_parser.models import Domain, State, ActionCall, PDDLObject
from typing import Dict

from sam_learning.core import NumericInformationGainLearner, PropositionalInformationGainLearner
from sam_learning.learners import PolynomialSAMLearning


class OnlineNSAMLearner(PolynomialSAMLearning):
    """"An online version of the Numeric SAM learner."""

    numeric_ig_learner: Dict[str, NumericInformationGainLearner]
    propositional_ig_learner: Dict[str, PropositionalInformationGainLearner]

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 1):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree)

    def _extract_objects_from_state(self, state: State) -> Dict[str, PDDLObject]:
        """

        :param state:
        :return:
        """
        state_objects = {}
        for grounded_predicates_set in state.state_predicates.values():
            for grounded_predicate in grounded_predicates_set:
                for obj_name, obj_type in grounded_predicate.signature.values():
                    state_objects[obj_name] = obj_type

        for grounded_function in state.state_fluents.values():
            for obj_name, obj_type in grounded_function.signature.values():
                state_objects[obj_name] = obj_type

        return state_objects

    def init_online_learning(self) -> None:
        """Initializes the online learning algorithm."""
        for action_name, action_data in self.partial_domain.actions.items():
            self.numeric_ig_learner[action_name] = NumericInformationGainLearner(
                action_name=action_name, domain_functions=self.partial_domain.functions)
            self.propositional_ig_learner[action_name] = PropositionalInformationGainLearner(action_name=action_name)

    def calculate_state_information_gain(self, state: State, action: ActionCall) -> float:
        """Calculates the information gain of a state.

        :param state: the state to calculate the information gain of.
        :param action: the action that is tested on whether should be executed in the state.
        :return: the information gain of the state.
        """
        state_objects = self._extract_objects_from_state(state)
        grounded_state_propositions = self.triplet_snapshot.create_propositional_state_snapshot(
            state, action, state_objects)
        lifted_state_propositions = self.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
        propositional_ig = self.propositional_ig_learner[action.name].calculate_sample_information_gain(
            set(lifted_state_propositions))

        grounded_state_functions = self.triplet_snapshot.create_numeric_state_snapshot(state, action, state_objects)
        lifted_state_functions = self.function_matcher.match_state_functions(action, grounded_state_functions)
        numeric_ig = self.numeric_ig_learner[action.name].calculate_sample_information_gain(lifted_state_functions)
        return numeric_ig + propositional_ig

    def execute_action(
            self, action_to_execute: ActionCall, previous_state: State, next_state: State, action_successfull: bool) -> None:
        """Executes an action in the environment and updates the action model accordingly.

        :param action_to_execute: the action to execute in the environment.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self.logger.info(f"Executing the action {action_to_execute.name} in the environment.")
        self.triplet_snapshot.update_triplet_snapshot(action_to_execute, previous_state, next_state)
        self.update_action(action_to_execute, previous_state, next_state)

"""An online version of the Numeric SAM learner."""
from typing import Dict

from pddl_plus_parser.models import Domain, State, ActionCall, PDDLObject

from sam_learning.core import InformationGainLearner, NotSafeActionError, LearnerDomain
from sam_learning.learners import PolynomialSAMLearning


class OnlineNSAMLearner(PolynomialSAMLearning):
    """"An online version of the Numeric SAM learner."""

    ig_learner: Dict[str, InformationGainLearner]

    def __init__(self, partial_domain: Domain, polynomial_degree: int = 1):
        super().__init__(partial_domain=partial_domain, polynomial_degree=polynomial_degree)

    @staticmethod
    def _extract_objects_from_state(state: State) -> Dict[str, PDDLObject]:
        """Extracts the objects from the state.

        :param state: the state to extract the objects from.
        :return: a dictionary mapping object names to their matching PDDL object .
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
            action_predicate_lifted_vocabulary = self.vocabulary_creator.create_lifted_vocabulary(
                self.partial_domain, action_data.signature)
            lifted_functions = self.vocabulary_creator.create_lifted_functions_vocabulary(
                self.partial_domain, action_data.signature)
            lifted_predicate_names = [p.untyped_representation for p in action_predicate_lifted_vocabulary]
            self.ig_learner[action_name] = InformationGainLearner(
                action_name=action_name, lifted_functions=lifted_functions.keys(),
                lifted_predicates=lifted_predicate_names)

    def calculate_state_information_gain(self, state: State, action: ActionCall) -> float:
        """Calculates the information gain of a state.

        :param state: the state to calculate the information gain of.
        :param action: the action that we calculate the information gain of executing in the state.
        :return: the information gain of the state.
        """
        state_objects = self._extract_objects_from_state(state)
        grounded_state_propositions = self.triplet_snapshot.create_propositional_state_snapshot(
            state, action, state_objects)
        lifted_state_propositions = self.matcher.get_possible_literal_matches(action, list(grounded_state_propositions))
        grounded_state_functions = self.triplet_snapshot.create_numeric_state_snapshot(state, action, state_objects)
        lifted_state_functions = self.function_matcher.match_state_functions(action, grounded_state_functions)
        self.ig_learner[action.name].remove_non_existing_functions(list(lifted_state_functions.keys()))

        numeric_ig = self.ig_learner[action.name].calculate_sample_information_gain(
            lifted_state_functions, lifted_state_propositions)
        return numeric_ig

    def execute_action(
            self, action_to_execute: ActionCall, previous_state: State, next_state: State,
            action_successful: bool) -> None:
        """Executes an action in the environment and updates the action model accordingly.

        :param action_to_execute: the action to execute in the environment.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        :param action_successful: whether the action was successful or not.
        """
        self.logger.info(f"Executing the action {action_to_execute.name} in the environment.")
        observation_objects = self._extract_objects_from_state(next_state)
        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=action_to_execute,
            observation_objects=observation_objects)

        pre_state_lifted_numeric_functions = self.function_matcher.match_state_functions(
            action_to_execute, self.triplet_snapshot.previous_state_functions)
        pre_state_lifted_predicates = self.matcher.get_possible_literal_matches(
            action_to_execute, list(self.triplet_snapshot.previous_state_predicates))

        if not action_successful:
            self.ig_learner[action_to_execute.name].add_negative_sample(
                numeric_negative_sample=pre_state_lifted_numeric_functions,
                negative_propositional_sample=pre_state_lifted_predicates)
            return

        if action_to_execute.name in self.observed_actions:
            super().update_action(action_to_execute, previous_state, next_state)
            return

        self.observed_actions.append(action_to_execute.name)
        super().add_new_action(action_to_execute, previous_state, next_state)

    def create_safe_model(self) -> LearnerDomain:
        """Creates a safe model from the currently learned data."""
        allowed_actions = {}
        for action_name, action in self.partial_domain.actions.items():
            if action_name not in self.storage:
                self.logger.debug(f"The action - {action_name} has not been observed in the trajectories!")
                continue

            self.storage[action_name].filter_out_inconsistent_state_variables()
            try:
                self._construct_safe_numeric_preconditions(action)
                self._construct_safe_numeric_effects(action)
                allowed_actions[action_name] = action
                self.logger.info(f"Done learning the action - {action_name}!")

            except NotSafeActionError as e:
                self.logger.debug(f"The action - {e.action_name} is not safe for execution, reason - {e.reason}")

        self.partial_domain.actions = allowed_actions
        return self.partial_domain

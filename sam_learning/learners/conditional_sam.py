"""Module containing the algorithm to learn action models with conditional effects."""
import logging
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import Domain, State, GroundedPredicate, ActionCall, Observation, \
    ObservedComponent, Predicate, ConditionalEffect, PDDLConstant

from sam_learning.core import DependencySet, LearnerDomain, extract_effects, LearnerAction
from sam_learning.learners import SAMLearner


def _extract_predicate_data(action: LearnerAction, predicate_str: str,
                            domain_constants: Dict[str, PDDLConstant]) -> Predicate:
    """Extracts the lifted bounded predicate from the string.

    :param action: the action that contains the predicate.
    :param predicate_str: the string representation of the predicate.
    :param domain_constants: the constants of the domain if exist.
    :return: the predicate object matching the string.
    """
    predicate_data = predicate_str.replace("(", "").replace(")", "").split(" ")
    predicate_data = [data for data in predicate_data if data != ""]
    predicate_name = predicate_data[0]
    combined_signature = {**action.signature}
    combined_signature.update({constant.name: constant.type for constant in domain_constants.values()})
    predicate_signature = {parameter: combined_signature[parameter] for parameter in predicate_data[1:]}
    return Predicate(predicate_name, predicate_signature)


class ConditionalSAM(SAMLearner):
    """Class dedicated to learning action models with conditional effects."""
    dependency_set: Dict[str, DependencySet]
    logger: logging.Logger
    max_antecedents_size: int

    def __init__(self, partial_domain: Domain, max_antecedents_size: int = 1,
                 preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.safe_actions = []
        self.max_antecedents_size = max_antecedents_size
        self.dependency_set = {}

    def _merge_positive_and_negative_predicates(
            self, positive_state_predicates: Set[GroundedPredicate],
            negative_state_predicates: Set[GroundedPredicate]) -> Set[GroundedPredicate]:
        """Merges the positive and negative predicates into a single set.

        :param positive_state_predicates: the set containing the positive predicates.
        :param negative_state_predicates: the set containing the negative predicates.
        :return: the merged set.
        """
        self.logger.debug("Creating a set of both negative and positive predicates.")
        positive_predicates = positive_state_predicates.copy()
        negative_predicates = negative_state_predicates.copy()
        return positive_predicates.union(negative_predicates)

    def _initialize_actions_dependencies(self, grounded_action: ActionCall) -> None:
        """Initialize the dependency set for a single action.

        :param grounded_action: the action that is currently being observed.
        """
        self.logger.debug("Initializing the dependency set and the effects for the action %s.", grounded_action.name)
        grounded_predicates = self._merge_positive_and_negative_predicates(self.previous_state_positive_predicates,
                                                                           self.previous_state_negative_predicates)
        lifted_predicates = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_predicates))
        dependency_set = DependencySet(self.max_antecedents_size)
        dependency_set.initialize_dependencies(set(lifted_predicates))
        self.dependency_set[grounded_action.name] = dependency_set
        self.partial_domain.actions[grounded_action.name].add_effects.update(lifted_predicates)
        self.partial_domain.actions[grounded_action.name].delete_effects.update(lifted_predicates)

    def _update_action_effects(self, grounded_action: ActionCall) -> None:
        """Set the correct data for the action's effects.

        :param grounded_action: the action that is currently being executed.
        """
        self.logger.debug(f"updating the effects for the action {grounded_action.name}.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        positive_next_state_matches = self.matcher.get_possible_literal_matches(
            grounded_action, list(self.next_state_positive_predicates))
        negative_next_state_matches = self.matcher.get_possible_literal_matches(
            grounded_action, list(self.next_state_negative_predicates))
        observed_action.add_effects.difference_update(negative_next_state_matches)
        observed_action.delete_effects.difference_update(positive_next_state_matches)
        self.logger.debug(f"Done filtering out predicates that cannot be effects.")

    def _find_literals_not_in_state(
            self, grounded_action: ActionCall, positive_predicates: Set[GroundedPredicate],
            negative_predicates: Set[GroundedPredicate]) -> Set[str]:
        """Finds literals that are not present in the current state.

        :param grounded_action: the action that is being executed.
        :param positive_predicates: the positive state predicates.
        :param negative_predicates: the negative state predicates.
        :return: the set of strings representing the literals that are not in the state.
        """
        state_positive_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(positive_predicates))
        state_negative_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(negative_predicates))
        # since we want to capture the literals NOT in s' we will transpose the literals values.
        missing_state_literals_str = [f"(not {literal.untyped_representation})" for literal in
                                      state_positive_literals]
        missing_state_literals_str.extend([literal.untyped_representation for literal in state_negative_literals])
        return set(missing_state_literals_str)

    def _find_literals_existing_in_state(
            self, grounded_action: ActionCall, positive_predicates, negative_predicates) -> Set[str]:
        """Finds the literals present in the current state.

        :param grounded_action: the action that is being executed.
        :param positive_predicates: the positive state predicates.
        :param negative_predicates: the negative state predicates.
        :return: the set of strings representing the literals that are in the state.
        """
        state_positive_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(positive_predicates))
        state_negative_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(negative_predicates))
        # since we want to capture the literals ARE in s' we will transpose the literals values.
        existing_state_literals_str = [literal.untyped_representation for literal in state_positive_literals]
        existing_state_literals_str.extend([f"(not {literal.untyped_representation})" for literal in
                                            state_negative_literals])
        return set(existing_state_literals_str)

    def _remove_existing_previous_state_dependencies(self, grounded_action: ActionCall) -> None:
        """Removes the literals that exist in the previous state from the dependency set of a literal that is
            not in the next state.

        :param grounded_action: the action that is being executed.
        """
        missing_next_state_literals_str = self._find_literals_not_in_state(
            grounded_action, self.next_state_positive_predicates, self.next_state_negative_predicates)
        existing_previous_state_literals_str = self._find_literals_existing_in_state(
            grounded_action, self.previous_state_positive_predicates, self.previous_state_negative_predicates)
        for literal in missing_next_state_literals_str:
            self.dependency_set[grounded_action.name].remove_dependencies(
                literal=literal, literals_to_remove=existing_previous_state_literals_str)

    def _remove_non_existing_previous_state_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removed literals that don't appear in the previous state from the dependency set of a literal that is
            guaranteed as an effect.

        :param grounded_action: the action that is being executed.
        :param previous_state: the state prior to the action execution.
        :param next_state: the state after the action execution.
        :return:
        """
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        lifted_add_effects = self.matcher.get_possible_literal_matches(
            grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(
            grounded_action, list(grounded_del_effects))
        effects_str = [literal.untyped_representation for literal in lifted_add_effects]
        effects_str.extend([f"(not {literal.untyped_representation})" for literal in lifted_delete_effects])
        missing_pre_state_literals_str = self._find_literals_not_in_state(
            grounded_action, self.previous_state_positive_predicates, self.previous_state_negative_predicates)
        for literal in effects_str:
            self.dependency_set[grounded_action.name].remove_dependencies(
                literal=literal, literals_to_remove=missing_pre_state_literals_str)

    def _remove_not_possible_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removes the literals that are not possible as antecedent candidates from the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self._remove_existing_previous_state_dependencies(grounded_action)
        self._remove_non_existing_previous_state_dependencies(
            grounded_action, previous_state, next_state)

    def _update_effects_data(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Updates the literals that cannot be effects as well as updates the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param next_state: the state following the action's execution.
        :param previous_state: the state prior to the action's execution.
        """
        self._update_action_effects(grounded_action)
        self._remove_not_possible_dependencies(grounded_action, previous_state, next_state)

    def _is_action_safe(self, action: LearnerAction, action_dependency_set: DependencySet) -> bool:
        """Checks if the action complies with the safety conditions.

        :param action: the action being tested.
        :param action_dependency_set: the action's dependency set.
        :return: True if the action is safe, False otherwise.
        """
        self.logger.debug(f"Checking if action {action.name} is safe.")
        preconditions_str = {precondition.untyped_representation for precondition in
                             action.positive_preconditions}
        preconditions_str.update([f"(not {precondition})" for precondition in action.negative_preconditions])

        return action_dependency_set.is_safe(preconditions_str)

    def _construct_restrictive_preconditions(
            self, action: LearnerAction, action_dependency_set: DependencySet) -> None:
        """Constructs the additional preconditions that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        """
        self.logger.info(f"Constructing restrictive preconditions for the unsafe action {action.name}.")
        positive_conditions, negative_conditions = action_dependency_set.extract_restrictive_conditions()
        for predicate_str in positive_conditions:
            action.positive_preconditions.add(_extract_predicate_data(
                action, predicate_str, self.partial_domain.constants))

        for predicate_str in negative_conditions:
            action.negative_preconditions.add(_extract_predicate_data(
                action, predicate_str, self.partial_domain.constants))

    def _construct_conditional_effects_from_dependency_set(
            self, action: LearnerAction, action_dependency_set: DependencySet) -> None:
        """Constructs the conditional effects of the action from the data available in the dependency set.

        :param action: the action that is being constructed.
        :param action_dependency_set: the action's dependency set.
        """
        for literal, dependencies in action_dependency_set.dependencies.items():
            if not action_dependency_set.is_safe_conditional_effect(literal):
                self.logger.debug(f"The literal {literal} is not a conditional effect.")
                continue

            self.logger.debug(f"Extracting the conditional effect - {literal} from the dependency set.")
            conditional_effect = ConditionalEffect()
            if literal.startswith("(not"):
                conditional_effect.add_effects.add(_extract_predicate_data(
                    action, f"{literal[5:-1]}", self.partial_domain.constants))

            else:
                conditional_effect.delete_effects.add(_extract_predicate_data(
                    action, literal, self.partial_domain.constants))

            positive_conditions, negative_conditions = action_dependency_set.extract_safe_conditionals(literal)
            for predicate_str in positive_conditions:
                conditional_effect.positive_conditions.add(_extract_predicate_data(
                    action, predicate_str, self.partial_domain.constants))

            for predicate_str in negative_conditions:
                conditional_effect.negative_conditions.add(_extract_predicate_data(
                    action, predicate_str, self.partial_domain.constants))

            action.conditional_effects.add(conditional_effect)

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action)
        self._update_effects_data(grounded_action, previous_state, next_state)
        self.observed_actions.append(observed_action.name)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        super()._update_action_preconditions(grounded_action, previous_state)
        self._update_effects_data(grounded_action, previous_state, next_state)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name
        super()._create_fully_observable_triplet_predicates(grounded_action, previous_state, next_state)
        if action_name not in self.observed_actions:
            self._initialize_actions_dependencies(grounded_action)
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        for action in self.partial_domain.actions.values():
            if action.name not in self.observed_actions:
                continue

            self.logger.debug("Removing preconditions predicates from the action's effects.")
            self._remove_preconditions_from_effects(action)

            if not self._is_action_safe(action, self.dependency_set[action.name]):
                self.logger.warning("Action %s is not safe to execute!", action.name)
                self._construct_restrictive_preconditions(action, self.dependency_set[action.name])
                continue

            self.logger.debug("Action %s is safe to execute.", action.name)
            self._construct_conditional_effects_from_dependency_set(action, self.dependency_set[action.name])
            self.safe_actions.append(action.name)

    def _remove_preconditions_from_effects(self, action: LearnerAction) -> None:
        """Removes the preconditions predicates from the action's effects.

        :param action: the learned action.
        """
        action.add_effects.difference_update(action.positive_preconditions)
        action.delete_effects.difference_update(action.negative_preconditions)

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report

"""Module containing the algorithm to learn action models with conditional effects."""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import Domain, State, GroundedPredicate, ActionCall, Observation, \
    ObservedComponent, ConditionalEffect, PDDLObject, PDDLType

from sam_learning.core import DependencySet, LearnerDomain, extract_effects, LearnerAction, \
    extract_predicate_data
from sam_learning.learners.sam_learning import SAMLearner


class ConditionalSAM(SAMLearner):
    """Class dedicated to learning action models with conditional effects."""
    conditional_antecedents: Dict[str, DependencySet]
    logger: logging.Logger
    max_antecedents_size: int
    current_trajectory_objects: Dict[str, PDDLObject]
    observed_effects: Dict[str, Set[str]]

    def __init__(self, partial_domain: Domain, max_antecedents_size: int = 1,
                 preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.safe_actions = []
        self.max_antecedents_size = max_antecedents_size
        self.conditional_antecedents = {}
        self.additional_parameters = defaultdict(dict)
        self.observed_effects = {action_name: set() for action_name in self.partial_domain.actions}

    def _initialize_actions_dependencies(self, grounded_action: ActionCall) -> None:
        """Initialize the dependency set for a single action.

        :param grounded_action: the action that is currently being observed.
        """
        self.logger.debug("Initializing the dependency set and the effects for the action %s.", grounded_action.name)
        lifted_action_signature = self.partial_domain.actions[grounded_action.name].signature
        vocabulary = self.vocabulary_creator.create_lifted_vocabulary(self.partial_domain, lifted_action_signature)
        dependency_set = DependencySet(
            self.max_antecedents_size, action_signature=self.partial_domain.actions[grounded_action.name].signature,
            domain_constants=self.partial_domain.constants)
        dependency_set.initialize_dependencies(vocabulary)
        self.conditional_antecedents[grounded_action.name] = dependency_set

    def _extract_lifted_conditional_effects(
            self, grounded_action: ActionCall, next_state: State, previous_state: State) -> Set[str]:
        """Extract the simple lifted add and delete effects from the action.

        :param grounded_action: the action that is currently being executed.
        :param next_state: the state after the action was executed.
        :param previous_state: the state before the action was executed.
        :return: the simple add and delete effects.
        """
        self.logger.debug(f"extracting the simple effects for the action {grounded_action.name}.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        lifted_effects = self.matcher.get_possible_literal_matches(
            grounded_action, list(grounded_add_effects.union(grounded_del_effects)))
        return {effect.untyped_representation for effect in lifted_effects}

    def _find_literals(
            self, grounded_action: ActionCall, grounded_predicates: List[GroundedPredicate], should_be_in_state: bool,
            extra_grounded_object: Optional[str] = None, extra_lifted_object: Optional[str] = None) -> Set[str]:
        """Find the *lifted* literals based on the input conditions.

        :param grounded_action: the action that is being executed.
        :param grounded_predicates: the state predicates to iterate over.
        :param should_be_in_state: whether the literals should be in the state or not.
        :param extra_grounded_object: an extra grounded object to add when trying to find quantified state literals.
        :param extra_lifted_object: an extra lifted object that indicates the parameter name of the grounded object.
        :return: the literals that match the input conditions.
        """
        lifted_literals = self.matcher.get_possible_literal_matches(
            grounded_action, grounded_predicates, extra_grounded_object, extra_lifted_object)

        if not should_be_in_state:
            for literal in lifted_literals:
                literal.is_positive = not literal.is_positive

        return set([literal.untyped_representation for literal in lifted_literals])

    def _find_literals_not_in_state(
            self, grounded_action: ActionCall, grounded_predicates: Set[GroundedPredicate],
            extra_grounded_object: Optional[str] = None, extra_lifted_object: Optional[str] = None) -> Set[str]:
        """Finds literals that are not present in the current state.

        :param grounded_action: the action that is being executed.
        :param grounded_predicates: the state predicates to iterate over.
        :param extra_grounded_object: an extra grounded object to add when trying to find quantified state literals.
        :param extra_lifted_object: an extra lifted object that indicates the parameter name of the grounded object.
        :return: the set of strings representing the literals that are not in the state.
        """

        return self._find_literals(grounded_action, list(grounded_predicates), should_be_in_state=False,
                                   extra_grounded_object=extra_grounded_object, extra_lifted_object=extra_lifted_object)

    def _find_literals_existing_in_state(
            self, grounded_action: ActionCall, grounded_predicates: Set[GroundedPredicate],
            extra_grounded_object: Optional[str] = None, extra_lifted_object: Optional[str] = None) -> Set[str]:
        """Finds the literals present in the current state.

        :param grounded_action: the action that is being executed.
        :param grounded_predicates: the state predicates to iterate over.
        :param extra_grounded_object: an extra grounded object to add when trying to find quantified state literals.
        :param extra_lifted_object: an extra lifted object that indicates the parameter name of the grounded object.
        :return: the set of strings representing the literals that are in the state.
        """
        return self._find_literals(grounded_action, list(grounded_predicates), should_be_in_state=True,
                                   extra_grounded_object=extra_grounded_object, extra_lifted_object=extra_lifted_object)

    def _remove_existing_previous_state_dependencies(self, grounded_action: ActionCall) -> None:
        """Removes the literals that exist in the previous state from the dependency set of a literal that is
            not in the next state.

        :param grounded_action: the action that is being executed.
        """
        self.logger.debug("removing existing previous state antecedents from literals not in the post state.")
        missing_next_state_literals = self._find_literals_not_in_state(
            grounded_action, self.triplet_snapshot.next_state_predicates)
        previous_state_literals = self._find_literals_existing_in_state(
            grounded_action, self.triplet_snapshot.previous_state_predicates)
        for literal in missing_next_state_literals:
            self.conditional_antecedents[grounded_action.name].remove_dependencies(
                literal=literal, literals_to_remove=previous_state_literals)
        self.logger.debug(f"Done removing existing previous state dependencies.")

    def _remove_non_existing_previous_state_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removed literals that don't appear in the previous state from the dependency set of a literal that is
            guaranteed as an effect.

        :param grounded_action: the action that is being executed.
        :param previous_state: the state prior to the action execution.
        :param next_state: the state after the action execution.
        :return:
        """
        self.logger.debug("Removing non-existing previous state antecedents from literals observed in s'/ s.")
        effects = self._extract_lifted_conditional_effects(
            grounded_action, previous_state=previous_state, next_state=next_state)
        missing_pre_state_literals = self._find_literals_not_in_state(
            grounded_action, self.triplet_snapshot.previous_state_predicates)
        for literal in effects:
            self.conditional_antecedents[grounded_action.name].remove_dependencies(
                literal=literal, literals_to_remove=missing_pre_state_literals, include_supersets=True)

    def _update_observed_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Set the correct data for the action's effects.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state before the action was executed.
        :param next_state: the state after the action was executed.
        """
        self.logger.debug(f"updating the effects for the action {grounded_action.name}.")
        self.observed_effects[grounded_action.name].update(
            self._extract_lifted_conditional_effects(grounded_action, previous_state=previous_state,
                                                     next_state=next_state))

    def _remove_not_antecedents(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removes the literals that are not possible as antecedent candidates from the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self._remove_existing_previous_state_dependencies(grounded_action)
        self._remove_non_existing_previous_state_dependencies(grounded_action, previous_state, next_state)

    def _apply_inductive_rules(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Updates the literals that cannot be effects as well as updates the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param next_state: the state following the action's execution.
        :param previous_state: the state prior to the action's execution.
        """
        self._update_observed_effects(grounded_action, previous_state, next_state)
        self._remove_not_antecedents(grounded_action, previous_state, next_state)

    def _construct_restrictive_preconditions(
            self, action: LearnerAction, action_dependency_set: DependencySet, literal: str) -> None:
        """Constructs the additional preconditions that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        :param literal: the literal determined to be unsafe.
        """
        self.logger.info(f"Constructing restrictive preconditions for the unsafe action {action.name}.")
        is_effect = literal in self.observed_effects[action.name]
        conservative_preconditions = action_dependency_set.construct_restrictive_preconditions(
            action.preconditions_str_set, literal, is_effect)
        if conservative_preconditions is not None:
            action.preconditions.add_condition(conservative_preconditions)

    def _construct_antecedents(self, action: LearnerAction, action_dependency_set: DependencySet,
                               conditional_effect: ConditionalEffect, literal: str,
                               additional_parameter: Optional[str] = None,
                               additional_parameter_type: Optional[PDDLType] = None) -> None:
        """Constructs the antecedents for the conditional effect.

        :param action: the action that the effect belongs to.
        :param action_dependency_set: the action's dependency set.
        :param conditional_effect: the conditional effect that is being constructed.
        :param literal: the literal representing the effect's result.
        :param additional_parameter: the additional parameter of the effect (if exists).
        :param additional_parameter_type: the type of the additional parameter (if exists).
        """
        antecedents = action_dependency_set.extract_safe_antecedents(literal)
        for antecedent in antecedents:
            conditional_effect.antecedents.add_condition(extract_predicate_data(
                action.signature, antecedent, self.partial_domain.constants, additional_parameter,
                additional_parameter_type))

    def _construct_result(
            self, action: LearnerAction, conditional_effect: ConditionalEffect, literal: str,
            additional_parameter: Optional[str] = None, additional_parameter_type: Optional[PDDLType] = None) -> None:
        """Constructs the result for the conditional effect.

        :param action: the action the effect belongs to.
        :param conditional_effect: the conditional effect that is being constructed.
        :param literal: the literal that represents the effect's result.
        :param additional_parameter: the additional parameter of the effect (if exists).
        :param additional_parameter_type: the type of the additional parameter (if exists).
        """
        conditional_effect.discrete_effects.add(extract_predicate_data(
            action.signature, literal, self.partial_domain.constants, additional_parameter,
            additional_parameter_type))

    def _construct_conditional_effect_data(
            self, action: LearnerAction, action_dependency_set: DependencySet, literal: str,
            additional_parameter: Optional[str] = None,
            additional_parameter_type: Optional[PDDLType] = None) -> ConditionalEffect:
        """Constructs the conditional effects that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        """
        conditional_effect = ConditionalEffect()
        self.logger.info(f"Constructing conditional effect's data for the unsafe action {action.name}.")
        self._construct_result(action, conditional_effect, literal, additional_parameter, additional_parameter_type)
        self._construct_antecedents(action, action_dependency_set, conditional_effect, literal,
                                    additional_parameter, additional_parameter_type)

        return conditional_effect

    def _construct_restrictive_effect(
            self, action: LearnerAction, dependency_set: DependencySet, literal: str,
            quantified_parameter: Optional[str] = None, quantified_type: Optional[str] = None) -> ConditionalEffect:
        """Construct a restrictive conditional effect from the dependency set that includes all possible antecedents.

        :param action: the action to construct the conditional effect for.
        :param dependency_set: the dependency set of the action.
        :param literal: the literal to possibly construct a conditional effect for.
        :return: the constructed conditional effect.
        """
        combined_conditions = set()
        for conditions in dependency_set.possible_antecedents[literal]:
            combined_conditions.update(conditions)

        temp_dependency_set = DependencySet(max_size_antecedents=dependency_set.max_antecedents,
                                            action_signature=action.signature,
                                            domain_constants=self.partial_domain.constants)
        temp_dependency_set.possible_antecedents[literal] = [combined_conditions]
        return self._construct_conditional_effect_data(action, temp_dependency_set, literal, quantified_parameter,
                                                       quantified_type)

    def _construct_simple_effect(self, action: LearnerAction, literal: str) -> None:
        """Constructs a simple effect (non-conditional effect) from the literal.

        :param action: the action to construct the effect for.
        :param literal: the literal that is the result of the effect.
        """
        self.logger.debug(f"The literal {literal} is a simple effect of the action.")
        action.discrete_effects.add(extract_predicate_data(action.signature, literal, self.partial_domain.constants))

    def _compress_conditional_effects(self, conditional_effects: List[ConditionalEffect]) -> List[ConditionalEffect]:
        """Compresses conditional effects that have the same antecedents to contain multiple results.

        :param conditional_effects: the list of conditional effects that are being compressed.
        :return: the compressed list of conditional effects.
        """
        self.logger.debug("Trying to compress the conditional effects based on their antecedents.")
        compressed_effects = []
        effects_to_remove = set()
        while len(conditional_effects) > 0:
            effect = conditional_effects.pop(0)
            compressed_effects.append(effect)
            for other_effect in conditional_effects:
                if not effect.antecedents == other_effect.antecedents:
                    continue

                effect.discrete_effects.update(other_effect.discrete_effects)
                effects_to_remove.add(other_effect)

            for effect_to_remove in effects_to_remove:
                if effect_to_remove in conditional_effects:
                    conditional_effects.remove(effect_to_remove)

        return compressed_effects

    def _verify_and_construct_safe_conditional_effects(self, action: LearnerAction) -> None:
        """Verifies that the action is safe and constructs its effects and preconditions.

        :param action: the action that is being verified.
        """
        self.logger.debug("Removing preconditions predicates from the action's effects.")
        action_dependency = self.conditional_antecedents[action.name]
        conditional_effects = []
        for possible_result in action_dependency.possible_antecedents:
            conditional_effect = self.verify_single_possible_conditional_effect(
                action, action_dependency, possible_result)
            if conditional_effect is not None:
                conditional_effects.append(conditional_effect)

        if len(conditional_effects) == 0:
            return

        compressed_effects = self._compress_conditional_effects(conditional_effects)
        action.conditional_effects = compressed_effects

    def verify_single_possible_conditional_effect(
            self, action: LearnerAction, action_dependency: DependencySet, literal: str) -> Optional[ConditionalEffect]:
        """Verifies whether the literal is a possible conditional effect of the action.

        :param action: the action that is being verified.
        :param action_dependency: the action's dependency set.
        :param literal: the literal considered as a possible conditional effect.
        """
        if not self.conditional_antecedents[action.name].is_safe_literal(literal):
            self.logger.debug(f"The literal {literal} is not considered to be safe for {action.name}.")
            self._construct_restrictive_preconditions(action, self.conditional_antecedents[action.name], literal)
            if literal not in self.observed_effects[action.name]:
                self.logger.debug(f"The literal {literal} was not observed as an effect of the action {action.name}.")
                return

            effect = self._construct_restrictive_effect(action, self.conditional_antecedents[action.name], literal)
            action.conditional_effects.add(effect)
            return

        self.logger.debug(f"The literal {literal} is safe to use in the action {action.name}.")
        if literal not in self.observed_effects[action.name]:
            self.logger.debug(f"The literal {literal} is not affected by the action {action.name}.")
            return

        if not self.conditional_antecedents[action.name].is_safe_conditional_effect(literal):
            self.logger.debug(f"The literal {literal} is not a conditional effect of the action {action.name}, "
                              f"constructing simple effect.")
            self._construct_simple_effect(action, literal)
            return

        self.logger.debug(f"The literal {literal} is a conditional effect of the action {action.name}")
        effect = self._construct_conditional_effect_data(action, action_dependency, literal)
        return effect

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Adds the action that is being observed for the first time.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action)
        self._apply_inductive_rules(grounded_action, previous_state, next_state)
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
        super()._update_action_preconditions(grounded_action)
        self._apply_inductive_rules(grounded_action, previous_state, next_state)
        self.logger.debug(f"Done updating the action - {grounded_action.name}")

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state
        action_name = grounded_action.name

        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=grounded_action,
            observation_objects=self.current_trajectory_objects)
        if action_name not in self.observed_actions:
            self._initialize_actions_dependencies(grounded_action)
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def construct_safe_actions(self) -> None:
        """Constructs the universal effects of the actions or a conservative version of them."""
        self.partial_domain.actions = {
            action_name: self.partial_domain.actions[action_name] for action_name in self.observed_actions}

        for action in self.partial_domain.actions.values():
            self.conditional_antecedents[action.name].remove_preconditions_literals(action.preconditions_str_set)
            self._verify_and_construct_safe_conditional_effects(action)
            self.logger.debug(f"Finished handling action {action.name}.")

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        super().start_measure_learning_time()
        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        super().end_measure_learning_time()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report

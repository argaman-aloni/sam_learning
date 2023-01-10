"""Module containing the algorithm to learn action models with conditional and universal effects."""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import Domain, State, GroundedPredicate, ActionCall, Observation, \
    ObservedComponent, ConditionalEffect, PDDLObject, UniversalQuantifiedEffect, PDDLType

from sam_learning.core import DependencySet, LearnerDomain, extract_effects, LearnerAction
from sam_learning.learners import SAMLearner
from sam_learning.learning_utilities import create_additional_parameter_name, find_unique_objects_by_type, \
    extract_predicate_data, FORALL, NOT_PREFIX, extract_quantified_effects


class ConditionalSAM(SAMLearner):
    """Class dedicated to learning action models with conditional effects."""
    dependency_set: Dict[str, DependencySet]
    quantified_dependency_set: Dict[str, Dict[str, DependencySet]]  # action_name -> type_name -> dependency_set
    additional_parameters: Dict[str, Dict[str, str]]  # action_name -> type_name -> parameter_name
    logger: logging.Logger
    max_antecedents_size: int
    current_trajectory_objects: Dict[str, PDDLObject]
    observed_effects: Dict[str, Set[str]]
    observed_universal_effects: Dict[str, Dict[str, Set[str]]]

    def __init__(self, partial_domain: Domain, max_antecedents_size: int = 1,
                 preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain)
        self.logger = logging.getLogger(__name__)
        self.preconditions_fluent_map = preconditions_fluent_map if preconditions_fluent_map else {}
        self.safe_actions = []
        self.max_antecedents_size = max_antecedents_size
        self.dependency_set = {}
        self.quantified_dependency_set = {action_name: {} for action_name in self.partial_domain.actions}
        self.additional_parameters = defaultdict(dict)
        self.observed_effects = {action_name: set() for action_name in self.partial_domain.actions}
        self.observed_universal_effects = {action_name: {} for action_name in self.partial_domain.actions}

    def _initialize_universal_dependencies(self, ground_action: ActionCall) -> None:
        """Initialize the universal antecedents candidates for a universal effect.

        :param ground_action: the action to initialize the universal dependencies for.
        """
        self.logger.debug("Initializing the universal antecedents candidates for action %s.", ground_action.name)
        lifted_action_signature = self.partial_domain.actions[ground_action.name].signature
        for pddl_type_name, pddl_type in self.partial_domain.types.items():
            if pddl_type_name == "object":
                continue

            additional_parameter_name = create_additional_parameter_name(self.partial_domain, ground_action, pddl_type)
            vocabulary = self.vocabulary_creator.create_lifted_vocabulary(
                domain=self.partial_domain,
                possible_parameters={**lifted_action_signature, additional_parameter_name: pddl_type},
                must_be_parameter=additional_parameter_name)
            dependency_set = DependencySet(self.max_antecedents_size)
            dependency_set.initialize_dependencies(vocabulary)
            self.quantified_dependency_set[ground_action.name][pddl_type_name] = dependency_set
            universal_effect = UniversalQuantifiedEffect(quantified_parameter=additional_parameter_name,
                                                         quantified_type=pddl_type)
            self.additional_parameters[ground_action.name][pddl_type_name] = additional_parameter_name
            self.observed_universal_effects[ground_action.name][pddl_type_name] = set()
            self.partial_domain.actions[ground_action.name].universal_effects.add(universal_effect)

    def _initialize_actions_dependencies(self, grounded_action: ActionCall) -> None:
        """Initialize the dependency set for a single action.

        :param grounded_action: the action that is currently being observed.
        """
        self.logger.debug("Initializing the dependency set and the effects for the action %s.", grounded_action.name)
        lifted_action_signature = self.partial_domain.actions[grounded_action.name].signature
        vocabulary = self.vocabulary_creator.create_lifted_vocabulary(
            domain=self.partial_domain, possible_parameters=lifted_action_signature)
        dependency_set = DependencySet(self.max_antecedents_size)
        dependency_set.initialize_dependencies(vocabulary)
        self.dependency_set[grounded_action.name] = dependency_set

    def _update_observed_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Set the correct data for the action's effects.

        :param grounded_action: the action that is currently being executed.
        """
        self.logger.debug(f"updating the effects for the action {grounded_action.name}.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))
        non_quantified_add_effects = {positive_literal.untyped_representation for positive_literal in add_effects}
        non_quantified_del_effects = {f"{NOT_PREFIX} {negative_literal.untyped_representation})" for negative_literal in
                                      delete_effects}
        self.observed_effects[grounded_action.name].update(non_quantified_add_effects)
        self.observed_effects[grounded_action.name].update(non_quantified_del_effects)

        self.logger.debug("adding observed universal effects.")
        for lifted_add_effects, lifted_delete_effects, pddl_type_name, _ in extract_quantified_effects(
                grounded_action, grounded_add_effects, grounded_del_effects, self.current_trajectory_objects,
                self.matcher, self.additional_parameters[grounded_action.name]):
            self.observed_universal_effects[grounded_action.name][pddl_type_name].update(
                {positive_literal.untyped_representation for positive_literal in lifted_add_effects})
            self.observed_universal_effects[grounded_action.name][pddl_type_name].update(
                {f"{NOT_PREFIX} {negative_literal.untyped_representation})" for negative_literal in delete_effects})

    def _find_literals_not_in_state(
            self, grounded_action: ActionCall, positive_predicates: Set[GroundedPredicate],
            negative_predicates: Set[GroundedPredicate],
            extra_grounded_object: Optional[str] = None, extra_lifted_object: Optional[str] = None) -> Set[str]:
        """Finds literals that are not present in the current state.

        :param grounded_action: the action that is being executed.
        :param positive_predicates: the positive state predicates.
        :param negative_predicates: the negative state predicates.
        :param extra_grounded_object: an extra grounded object to add when trying to find quantified state literals.
        :param extra_lifted_object: an extra lifted object that indicates the parameter name of the grounded object.
        :return: the set of strings representing the literals that are not in the state.
        """
        state_positive_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(positive_predicates), extra_grounded_object, extra_lifted_object)
        state_negative_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(negative_predicates), extra_grounded_object, extra_lifted_object)
        # since we want to capture the literals NOT in that state we will transpose the literals values.
        missing_state_literals_str = [f"{NOT_PREFIX} {literal.untyped_representation})" for literal in
                                      state_positive_literals]
        missing_state_literals_str.extend([literal.untyped_representation for literal in state_negative_literals])
        return set(missing_state_literals_str)

    def _find_literals_existing_in_state(
            self, grounded_action: ActionCall, positive_predicates: Set[GroundedPredicate],
            negative_predicates: Set[GroundedPredicate],
            extra_grounded_object: Optional[str] = None, extra_lifted_object: Optional[str] = None) -> Set[str]:
        """Finds the literals present in the current state.

        :param grounded_action: the action that is being executed.
        :param positive_predicates: the positive state predicates.
        :param negative_predicates: the negative state predicates.
        :param extra_grounded_object: an extra grounded object to add when trying to find quantified state literals.
        :param extra_lifted_object: an extra lifted object that indicates the parameter name of the grounded object.
        :return: the set of strings representing the literals that are in the state.
        """
        state_positive_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(positive_predicates), extra_grounded_object, extra_lifted_object)
        state_negative_literals = self.matcher.get_possible_literal_matches(
            grounded_action, list(negative_predicates), extra_grounded_object, extra_lifted_object)
        existing_state_literals_str = [literal.untyped_representation for literal in state_positive_literals]
        existing_state_literals_str.extend([f"{NOT_PREFIX} {literal.untyped_representation})" for literal in
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
                literal=literal, literals_to_remove=existing_previous_state_literals_str, include_supersets=True)
        self.logger.debug(f"Done removing existing previous state dependencies.")

    def _remove_existing_previous_state_quantified_dependencies(self, grounded_action: ActionCall) -> None:
        """Removes the literals that exist in the previous state from the dependency set of a literal that is
            not in the next state.

        :param grounded_action: the action that is being executed.
        """
        self.logger.debug("Removing existing previous state quantified dependencies.")
        objects_by_type = find_unique_objects_by_type(self.current_trajectory_objects, grounded_action.parameters)
        for pddl_type_name, pddl_objects in objects_by_type.items():
            try:
                additional_parameter_name = self.additional_parameters[grounded_action.name][pddl_type_name]
            except KeyError:
                raise ValueError(
                    f"Could not find additional parameter for action {str(grounded_action)} and type {pddl_type_name}")

            missing_next_state_literals_str = set()
            existing_previous_state_literals_str = set()
            for pddl_object in pddl_objects:
                missing_next_state_literals_str.update(self._find_literals_not_in_state(
                    grounded_action, self.next_state_positive_predicates, self.next_state_negative_predicates,
                    pddl_object.name, additional_parameter_name))
                existing_previous_state_literals_str.update(self._find_literals_existing_in_state(
                    grounded_action, self.next_state_positive_predicates, self.next_state_negative_predicates,
                    pddl_object.name, additional_parameter_name))

            for literal in missing_next_state_literals_str:
                if additional_parameter_name not in literal:
                    continue

                self.quantified_dependency_set[grounded_action.name][pddl_type_name].remove_dependencies(
                    literal=literal, literals_to_remove=existing_previous_state_literals_str, include_supersets=True)

        self.logger.debug(f"Done removing existing previous state quantified dependencies for every object.")

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
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))
        effects_str = [literal.untyped_representation for literal in lifted_add_effects]
        effects_str.extend([f"{NOT_PREFIX} {literal.untyped_representation})" for literal in lifted_delete_effects])
        missing_pre_state_literals_str = self._find_literals_not_in_state(
            grounded_action, self.previous_state_positive_predicates, self.previous_state_negative_predicates)
        for literal in effects_str:
            self.dependency_set[grounded_action.name].remove_dependencies(
                literal=literal, literals_to_remove=missing_pre_state_literals_str)

    def _remove_non_existing_previous_state_quantified_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removed literals that don't appear in the previous state from the dependency set of a literal that is
            guaranteed as an effect.

        :param grounded_action: the action that is being executed.
        :param previous_state: the state prior to the action execution.
        :param next_state: the state after the action execution.
        """
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        for lifted_add_effects, lifted_delete_effects, pddl_type_name, pddl_object in \
                extract_quantified_effects(
                    grounded_action, grounded_add_effects, grounded_del_effects,
                    self.current_trajectory_objects, self.matcher, self.additional_parameters[grounded_action.name]):
            additional_parameter_name = self.additional_parameters[grounded_action.name][pddl_type_name]
            missing_pre_state_literals_str = self._find_literals_not_in_state(
                grounded_action, self.previous_state_positive_predicates, self.previous_state_negative_predicates,
                pddl_object.name, additional_parameter_name)
            effects_str = [literal.untyped_representation for literal in lifted_add_effects]
            effects_str.extend([f"{NOT_PREFIX} {literal.untyped_representation})" for literal in lifted_delete_effects])

            for literal in effects_str:
                self.quantified_dependency_set[grounded_action.name][pddl_type_name].remove_dependencies(
                    literal=literal, literals_to_remove=missing_pre_state_literals_str)

    def _remove_not_possible_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removes the literals that are not possible as antecedent candidates from the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        self._remove_existing_previous_state_dependencies(grounded_action)
        self._remove_existing_previous_state_quantified_dependencies(grounded_action)
        self._remove_non_existing_previous_state_dependencies(grounded_action, previous_state, next_state)
        self._remove_non_existing_previous_state_quantified_dependencies(grounded_action, previous_state, next_state)

    def _update_effects_data(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Updates the literals that cannot be effects as well as updates the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param next_state: the state following the action's execution.
        :param previous_state: the state prior to the action's execution.
        """
        self._update_observed_effects(grounded_action, previous_state, next_state)
        self._remove_not_possible_dependencies(grounded_action, previous_state, next_state)

    def _construct_restrictive_preconditions(
            self, action: LearnerAction, action_dependency_set: DependencySet, literal: str) -> None:
        """Constructs the additional preconditions that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        :param literal: the literal determined to be unsafe.
        """
        self.logger.info(f"Constructing restrictive preconditions for the unsafe action {action.name}.")
        preconditions_str = {precondition.untyped_representation for precondition in self.partial_domain.actions[
            action.name].positive_preconditions}
        preconditions_str.update(f"{NOT_PREFIX} {precondition.untyped_representation})" for precondition in
                                 self.partial_domain.actions[action.name].negative_preconditions)

        is_effect = literal in self.observed_effects[action.name]
        conservative_preconditions = action_dependency_set.extract_restrictive_conditions(
            preconditions_str, literal, is_effect)
        action.manual_preconditions.append(conservative_preconditions)

    def _construct_restrictive_universal_preconditions(
            self, action: LearnerAction, action_dependency_set: DependencySet,
            quantified_type: str, literal: str) -> None:
        """Constructs the additional preconditions that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        :param quantified_type: the quantified type of the literal.
        :param literal: the literal determined to be unsafe.
        """
        self.logger.info(f"Constructing restrictive preconditions for the unsafe action {action.name}.")
        is_effect = literal in self.observed_universal_effects[action.name][quantified_type]
        conservative_conditional_preconditions = action_dependency_set.extract_restrictive_conditions(
            set(), literal, is_effect)
        action.manual_preconditions.append(
            f"({FORALL} ({self.additional_parameters[action.name][quantified_type]} - {quantified_type}) "
            f"{conservative_conditional_preconditions})")

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
        if literal.startswith(NOT_PREFIX):
            conditional_effect.delete_effects.add(extract_predicate_data(
                action, f"{literal[5:-1]}", self.partial_domain.constants,
                additional_parameter, additional_parameter_type))

        else:
            conditional_effect.add_effects.add(extract_predicate_data(
                action, literal, self.partial_domain.constants, additional_parameter, additional_parameter_type))

        positive_conditions, negative_conditions = action_dependency_set.extract_safe_conditionals(literal)
        for predicate_str in positive_conditions:
            conditional_effect.positive_conditions.add(extract_predicate_data(
                action, predicate_str, self.partial_domain.constants, additional_parameter, additional_parameter_type))

        for predicate_str in negative_conditions:
            conditional_effect.negative_conditions.add(extract_predicate_data(
                action, predicate_str, self.partial_domain.constants, additional_parameter, additional_parameter_type))

        return conditional_effect

    def _construct_restrictive_universal_effect(
            self, action: LearnerAction, quantified_type: str, quantified_literal: str) -> None:
        """Removes the universal effect from the action.

        :param action: the action that contains the universal effect.
        :param quantified_type: the type of the quantified variable.
        :param quantified_literal: the literal that is quantified.
        """
        if quantified_literal not in self.observed_universal_effects[action.name][quantified_type]:
            return

        combined_conditions = set()
        dependency_set = self.quantified_dependency_set[action.name][quantified_type]
        for conditions in dependency_set.dependencies[quantified_literal]:
            combined_conditions.update(conditions)

        temp_dependency_set = DependencySet(max_size_antecedents=dependency_set.max_size_antecedents)
        temp_dependency_set.dependencies[quantified_literal] = [combined_conditions]
        additional_parameter_name = self.additional_parameters[action.name][quantified_type]
        conditional_effect = self._construct_conditional_effect_data(
            action, temp_dependency_set, quantified_literal, additional_parameter_name,
            self.partial_domain.types[quantified_type])
        universal_effect = [effect for effect in self.partial_domain.actions[action.name].universal_effects
                            if effect.quantified_type.name == quantified_type][0]
        universal_effect.conditional_effects.add(conditional_effect)

    def _construct_universal_effects_from_dependency_set(
            self, action: LearnerAction, action_dependency_set: DependencySet, quantified_type: str,
            quantified_literal: str) -> None:
        """Constructs the conditional effects of the action from the data available in the dependency set.

        :param action: the action that is being constructed.
        :param action_dependency_set: the action's dependency set.
        :param quantified_type: the type of the quantified variable.
        :param quantified_literal: the literal that is quantified.
        """
        universal_effect = [effect for effect in self.partial_domain.actions[action.name].universal_effects
                            if effect.quantified_type.name == quantified_type][0]

        self.logger.debug(f"Extracting the universal effect - {quantified_literal} from the dependency set.")
        additional_parameter = self.additional_parameters[action.name][quantified_type]
        conditional_effect = self._construct_conditional_effect_data(
            action, action_dependency_set, quantified_literal, additional_parameter=additional_parameter,
            additional_parameter_type=self.partial_domain.types[quantified_type])
        universal_effect.conditional_effects.add(conditional_effect)

    def _remove_preconditions_from_dependency_set(self, action: LearnerAction) -> None:
        """Removes the preconditions predicates from the action's dependency set.

        :param action: the learned action.
        """
        self.logger.debug("Removing the preconditions from the possible conditional effects")
        preconditions_literals = {predicate.untyped_representation for predicate in action.positive_preconditions}
        preconditions_literals.update(
            {f"(not {predicate.untyped_representation})" for predicate in action.negative_preconditions})
        self.dependency_set[action.name].remove_preconditions_literals(preconditions_literals)

    def _construct_restrictive_conditional_effects(
            self, action: LearnerAction, dependency_set: DependencySet, literal: str) -> None:
        """Construct a restrictive conditional effect from the dependency set that includes all possible antecedents.

        :param action: the action to construct the conditional effect for.
        :param dependency_set: the dependency set of the action.
        :param literal: the literal to possibly construct a conditional effect for.
        """
        if literal not in self.observed_effects[action.name]:
            self.logger.debug(f"The literal {literal} was not observed as an effect of the action {action.name}.")
            return

        combined_conditions = set()
        for conditions in dependency_set.dependencies[literal]:
            combined_conditions.update(conditions)

        temp_dependency_set = DependencySet(max_size_antecedents=dependency_set.max_size_antecedents)
        temp_dependency_set.dependencies[literal] = [combined_conditions]
        conditional_effect = self._construct_conditional_effect_data(action, temp_dependency_set, literal)
        action.conditional_effects.add(conditional_effect)

    def _construct_simple_effect(self, action: LearnerAction, literal: str) -> None:
        """Constructs a simple effect (non-conditional effect) from the literal.

        :param action: the action to construct the effect for.
        :param literal: the literal that is the result of the effect.
        """
        self.logger.debug(f"The literal {literal} is a simple effect of the action.")
        constants = self.partial_domain.constants
        if literal.startswith(NOT_PREFIX):
            action.delete_effects.add(extract_predicate_data(action, f"{literal[5:-1]}", constants))
            return

        action.add_effects.add(extract_predicate_data(action, literal, constants))

    def verify_single_possible_conditional_effect(self, action: LearnerAction, action_dependency: DependencySet,
                                                  literal: str) -> None:
        """

        :param action:
        :param action_dependency:
        :param literal:
        :return:
        """
        if not self.dependency_set[action.name].is_safe_literal(literal):
            self.logger.debug(f"The literal {literal} is not considered to be safe for {action.name}.")
            self._construct_restrictive_preconditions(action, self.dependency_set[action.name], literal)
            self._construct_restrictive_conditional_effects(action, self.dependency_set[action.name], literal)
            return

        self.logger.debug(f"The literal {literal} is safe to use in the action {action.name}.")
        if literal in self.observed_effects[action.name]:
            if not self.dependency_set[action.name].is_conditional_effect(literal):
                self._construct_simple_effect(action, literal)

            else:
                conditional_effect = self._construct_conditional_effect_data(action, action_dependency, literal)
                action.conditional_effects.add(conditional_effect)

    def _verify_and_construct_safe_conditional_effects(self, action: LearnerAction) -> None:
        """Verifies that the action is safe and constructs its effects and preconditions.

        :param action: the action that is being verified.
        """
        self.logger.debug("Removing preconditions predicates from the action's effects.")
        self._remove_preconditions_from_dependency_set(action)
        action_dependency = self.dependency_set[action.name]
        for literal in action_dependency.dependencies:
            self.verify_single_possible_conditional_effect(action, action_dependency, literal)

    def _verify_and_construct_safe_universal_effects(self, action: LearnerAction) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        self.logger.info("Constructing the safe universal effects.")
        for quantified_type, dependency_set in self.quantified_dependency_set[action.name].items():
            for quantified_literal in dependency_set.dependencies:
                if not dependency_set.is_safe_literal(quantified_literal):
                    self.logger.debug(f"The quantified literal {quantified_literal} "
                                      f"is not considered to be safe for {action.name}.")
                    self._construct_restrictive_universal_preconditions(action, dependency_set, quantified_type,
                                                                        quantified_literal)
                    self._construct_restrictive_universal_effect(action, quantified_type, quantified_literal)

                if quantified_literal not in self.observed_universal_effects[action.name][quantified_type]:
                    self.logger.debug(f"The literal {quantified_literal} was not observed as a universal effect.")
                    continue

                self.logger.debug(f"The quantified literal {quantified_literal} is safe for action {action.name}.")
                self._construct_universal_effects_from_dependency_set(action, dependency_set, quantified_type,
                                                                      quantified_literal)

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
        super()._create_fully_observable_triplet_predicates(
            grounded_action, previous_state, next_state, should_ignore_action=True)
        if action_name not in self.observed_actions:
            self._initialize_actions_dependencies(grounded_action)
            self._initialize_universal_dependencies(grounded_action)
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def construct_safe_actions(self) -> None:
        """Constructs the universal effects of the actions or a conservative version of them."""
        self.partial_domain.actions = {
            action_name: self.partial_domain.actions[action_name] for action_name in self.observed_actions}

        for action in self.partial_domain.actions.values():
            self._verify_and_construct_safe_conditional_effects(action)
            self._verify_and_construct_safe_universal_effects(action)
            self.logger.debug(f"Finished handling action {action.name}.")

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

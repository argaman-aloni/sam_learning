"""Module containing the algorithm to learn action models with conditional and universal effects."""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.models import Domain, State, ActionCall, Observation, \
    ObservedComponent, PDDLType, Predicate, UniversalEffect, UniversalPrecondition, GroundedPredicate

from sam_learning.core import DependencySet, LearnerDomain, extract_effects, LearnerAction, \
    create_additional_parameter_name, find_unique_objects_by_type
from sam_learning.learners.conditional_sam import ConditionalSAM


class UniversallyConditionalSAM(ConditionalSAM):
    """Class dedicated to learning action models with universal effects.

    Note:
        This class does not support learning universal preconditions.
    """
    logger: logging.Logger
    quantified_antecedents: Dict[str, Dict[str, DependencySet]]  # action_name -> type_name -> dependency_set
    additional_parameters: Dict[str, Dict[str, str]]  # action_name -> type_name -> parameter_name
    observed_universal_effects: Dict[str, Dict[str, Set[str]]]
    universals_map: Dict[str, List[str]]

    def __init__(self, partial_domain: Domain, max_antecedents_size: int = 1,
                 preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
                 universals_map: Dict[str, List[str]] = None):
        super().__init__(partial_domain, max_antecedents_size, preconditions_fluent_map)
        self.logger = logging.getLogger(__name__)
        self.quantified_antecedents = {action_name: {} for action_name in self.partial_domain.actions}
        self.additional_parameters = defaultdict(dict)
        self.observed_universal_effects = {action_name: {
            type_name: set() for type_name in self.partial_domain.types if type_name != "object"}
            for action_name in self.partial_domain.actions}
        self.universals_map = universals_map if universals_map is not None else {}

    def _create_antecedents_and_results_dicts(
            self, additional_parameter_name: str, lifted_action_signature: Dict[str, PDDLType],
            pddl_type: PDDLType) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Create the antecedents and results dictionaries for a given type.

        :param additional_parameter_name: the name of the additional parameter.
        :param lifted_action_signature: the signature of the lifted action.
        :param pddl_type: the type of the additional parameter to create the dictionaries for.
        :return: the antecedents and results dictionaries.
        """
        extended_signature = {**lifted_action_signature, additional_parameter_name: pddl_type}
        self.logger.debug("Creating the dictionary containing the possible results for the action")
        results_dictionary = self.vocabulary_creator.create_lifted_vocabulary(
            self.partial_domain, extended_signature, must_be_parameter=additional_parameter_name)
        self.logger.debug("Creating the dictionary containing the possible antecedents for the action")
        antecedents_dictionary = self.vocabulary_creator.create_lifted_vocabulary(
            self.partial_domain, extended_signature)
        return antecedents_dictionary, results_dictionary

    def _initialize_universal_dependencies(self, ground_action: ActionCall) -> None:
        """Initialize the universal antecedents candidates for a universal effect.

        :param ground_action: the action to initialize the universal dependencies for.
        """
        self.logger.debug("Initializing the universal antecedents candidates for action %s.", ground_action.name)
        action_signature = self.partial_domain.actions[ground_action.name].signature
        for pddl_type_name, pddl_type in self.partial_domain.types.items():
            if pddl_type_name == "object" or pddl_type_name not in self.universals_map[ground_action.name]:
                continue

            additional_param = create_additional_parameter_name(self.partial_domain, ground_action, pddl_type)
            antecedents, results = self._create_antecedents_and_results_dicts(
                additional_param, action_signature, pddl_type)
            extended_signature = {**action_signature, additional_param: pddl_type}
            dependency_set = DependencySet(self.max_antecedents_size, action_signature=extended_signature,
                                           domain_constants=self.partial_domain.constants)
            dependency_set.initialize_dependencies(lifted_literals=results, antecedents=antecedents)
            self.quantified_antecedents[ground_action.name][pddl_type_name] = dependency_set
            self.additional_parameters[ground_action.name][pddl_type_name] = additional_param
            self.partial_domain.actions[ground_action.name].universal_effects.add(
                UniversalEffect(additional_param, pddl_type))

    def _update_observed_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Set the correct data for the action's effects.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state before the action was executed.
        :param next_state: the state after the action was executed.
        """
        self.logger.debug(f"updating the effects for the action {grounded_action.name}.")
        super()._update_observed_effects(grounded_action, previous_state, next_state)
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        grounded_effects = list(grounded_add_effects.union(grounded_del_effects))

        self.logger.debug("adding observed universal effects.")
        action_parameters = grounded_action.parameters
        universal_effects = [eff for eff in grounded_effects if
                             len(set(eff.grounded_objects).difference(action_parameters)) > 0]
        for effect in universal_effects:
            # we assume that there can only be one additional parameter per effect
            additional_object = [obj for obj in effect.grounded_objects if obj not in action_parameters][0]
            pddl_object = self.current_trajectory_objects[additional_object]
            additional_parameter = self.additional_parameters[grounded_action.name][pddl_object.type.name]
            lifted_effect = self.matcher.match_predicate_to_action_literals(
                effect, grounded_action, pddl_object.name, additional_parameter)[0]
            self.observed_universal_effects[grounded_action.name][pddl_object.type.name].add(
                lifted_effect.untyped_representation)

    def _reduce_state_predicates_based_on_types(self, state_predicates: Set[GroundedPredicate],
                                                legal_parameters: List[str]) -> Set[GroundedPredicate]:
        """Reduce the state predicates based on the types of the objects in the state.

        :param state_predicates: the predicates in the state that we need to reduce in number to optimize the search.
        :param legal_parameters: the legal parameters of the action combined with the current searched object.
        :return: the reduced state predicates.
        """
        self.logger.debug("Reducing the state predicates based on the types of the objects being currently searched.")
        reduced_state = set()
        for predicate in state_predicates:
            if set(predicate.grounded_objects).issubset(legal_parameters):
                reduced_state.add(predicate)

        return reduced_state

    def _remove_existing_previous_state_quantified_dependencies(self, grounded_action: ActionCall) -> None:
        """Removes the literals that exist in the previous state from the dependency set of a literal that is
            not in the next state.

        :param grounded_action: the action that is being executed.
        """
        self.logger.debug("Removing existing previous state quantified antecedents from non result literals.")
        objects_by_type = find_unique_objects_by_type(self.current_trajectory_objects, grounded_action.parameters)
        for type_name, additional_param in self.additional_parameters[grounded_action.name].items():
            pddl_objects = objects_by_type[type_name]
            for pddl_object in pddl_objects:
                allowed_objects = [*grounded_action.parameters, pddl_object.name]
                reduced_previous_state = self._reduce_state_predicates_based_on_types(
                    self.triplet_snapshot.previous_state_predicates, allowed_objects)
                reduced_next_state = self._reduce_state_predicates_based_on_types(
                    self.triplet_snapshot.next_state_predicates, allowed_objects)

                not_results = self._find_literals_not_in_state(
                    grounded_action, reduced_next_state, pddl_object.name, additional_param)
                antecedents = self._find_literals_existing_in_state(
                    grounded_action, reduced_previous_state, pddl_object.name, additional_param)

                for literal in not_results:
                    type_dependency = self.quantified_antecedents[grounded_action.name][type_name]
                    if not type_dependency.is_possible_result(literal) or literal in self.observed_universal_effects:
                        continue

                    type_dependency.remove_dependencies(literal, antecedents)

        self.logger.debug(f"Done removing existing previous state quantified dependencies for every object.")

    def _remove_non_existing_previous_state_quantified_dependencies(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removed literals that don't appear in the previous state from the dependency set of a literal that is
            guaranteed as an effect.

        :param grounded_action: the action that is being executed.
        :param previous_state: the state prior to the action execution.
        :param next_state: the state after the action execution.
        """
        self.logger.debug("Removing non-existing previous state quantified antecedents from quantified literals "
                          "observed in s'/ s.")
        grounded_add_effects, grounded_del_effects = extract_effects(previous_state, next_state)
        grounded_effects = list(grounded_add_effects.union(grounded_del_effects))
        action_parameters = grounded_action.parameters
        universal_effects = [eff for eff in grounded_effects if
                             len(set(eff.grounded_objects).difference(action_parameters)) > 0]

        for effect in universal_effects:
            # we assume that there can only be one additional parameter per effect
            additional_object = [obj for obj in effect.grounded_objects if obj not in action_parameters][0]
            allowed_objects = [*action_parameters, additional_object]
            pddl_object = self.current_trajectory_objects[additional_object]
            additional_parameter = self.additional_parameters[grounded_action.name][pddl_object.type.name]
            lifted_effect = self.matcher.match_predicate_to_action_literals(
                effect, grounded_action, pddl_object.name, additional_parameter)[0]
            reduced_previous_state = self._reduce_state_predicates_based_on_types(
                self.triplet_snapshot.previous_state_predicates, allowed_objects)
            missing_pre_state_literals = self._find_literals_not_in_state(
                grounded_action, reduced_previous_state, pddl_object.name, additional_parameter)
            self.quantified_antecedents[grounded_action.name][pddl_object.type.name].remove_dependencies(
                literal=lifted_effect.untyped_representation, literals_to_remove=missing_pre_state_literals,
                include_supersets=True)

    def _remove_not_antecedents(
            self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Removes the literals that are not possible as antecedent candidates from the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param previous_state: the state prior to the action's execution.
        :param next_state: the state following the action's execution.
        """
        super()._remove_not_antecedents(grounded_action, previous_state, next_state)
        self._remove_existing_previous_state_quantified_dependencies(grounded_action)
        self._remove_non_existing_previous_state_quantified_dependencies(grounded_action, previous_state, next_state)

    def _construct_restrictive_universal_effect(
            self, action: LearnerAction, quantified_type: str, quantified_literal: str) -> None:
        """Removes the universal effect from the action.

        :param action: the action that contains the universal effect.
        :param quantified_type: the type of the quantified variable.
        :param quantified_literal: the literal that is quantified.
        """
        self.logger.info(f"Constructing restrictive universal effect for the unsafe action {action.name}.")
        if quantified_literal not in self.observed_universal_effects[action.name][quantified_type]:
            return

        dependency_set = self.quantified_antecedents[action.name][quantified_type]
        param_name = self.additional_parameters[action.name][quantified_type]
        effect = super()._construct_restrictive_effect(
            action, dependency_set, quantified_literal, param_name, quantified_type)
        universal_effect = [effect for effect in self.partial_domain.actions[action.name].universal_effects
                            if effect.quantified_type.name == quantified_type][0]
        universal_effect.conditional_effects.add(effect)

    def _construct_universal_effects_from_dependency_set(
            self, action: LearnerAction, action_dependency_set: DependencySet, quantified_type: str,
            quantified_literal: str) -> None:
        """Constructs the conditional effects of the action from the data available in the dependency set.

        :param action: the action that is being constructed.
        :param action_dependency_set: the action's dependency set.
        :param quantified_type: the type of the quantified variable.
        :param quantified_literal: the literal that is quantified.
        """
        if quantified_literal not in self.observed_universal_effects[action.name][quantified_type]:
            self.logger.debug(f"The literal {quantified_literal} was not observed as a universal effect.")
            return

        universal_effect = [effect for effect in self.partial_domain.actions[action.name].universal_effects
                            if effect.quantified_type.name == quantified_type][0]

        self.logger.debug(f"Extracting the universal effect - {quantified_literal} from the dependency set.")
        additional_parameter = self.additional_parameters[action.name][quantified_type]
        conditional_effect = self._construct_conditional_effect_data(
            action, action_dependency_set, quantified_literal, additional_parameter=additional_parameter,
            additional_parameter_type=self.partial_domain.types[quantified_type])
        universal_effect.conditional_effects.add(conditional_effect)

    def _remove_preconditions_from_antecedents(self, action: LearnerAction) -> None:
        """Removes the preconditions predicates from the action's dependency set.

        :param action: the learned action.
        """
        self.logger.debug("Removing the preconditions from the possible conditional effects")
        for quantified_type in self.quantified_antecedents[action.name]:
            self.quantified_antecedents[action.name][quantified_type].remove_preconditions_literals(
                action.preconditions_str_set)

    def _construct_restrictive_preconditions(
            self, action: LearnerAction, action_dependency_set: DependencySet, literal: str,
            quantified_type: Optional[str] = None) -> None:
        """Constructs the additional preconditions that are required for the action to be safe.

        :param action: the action that contains unsafe literals in the effects.
        :param action_dependency_set: the action's dependency set.
        :param literal: the literal determined to be unsafe.
        :param quantified_type: the type of the quantified literal (if exists).
        """
        self.logger.info(f"Constructing restrictive preconditions for the unsafe action {action.name}.")
        if quantified_type is None:
            super()._construct_restrictive_preconditions(action, action_dependency_set, literal)
            return

        is_effect = literal in self.observed_universal_effects[action.name][quantified_type]
        conservative_preconditions = action_dependency_set.construct_restrictive_preconditions(
            action.preconditions_str_set, literal, is_effect)
        universal_precondition = UniversalPrecondition(
            quantified_parameter=self.additional_parameters[action.name][quantified_type],
            quantified_type=self.partial_domain.types[quantified_type])
        if conservative_preconditions is not None:
            universal_precondition.add_condition(conservative_preconditions)
        action.preconditions.add_condition(universal_precondition)

    def _verify_and_construct_safe_universal_effects(self, action: LearnerAction) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        self.logger.info("Constructing the safe universal effects.")
        for quantified_type, dependency_set in self.quantified_antecedents[action.name].items():
            for possible_result in dependency_set.possible_antecedents:
                if not dependency_set.is_safe_literal(possible_result):
                    self._construct_restrictive_preconditions(action, dependency_set, possible_result, quantified_type)
                    self._construct_restrictive_universal_effect(action, quantified_type, possible_result)
                    continue

                self.logger.info(f"The quantified literal {possible_result} is safe for action {action.name}.")
                self._construct_universal_effects_from_dependency_set(action, dependency_set, quantified_type,
                                                                      possible_result)

        for universal_effect in action.universal_effects:
            compressed_effects = self._compress_conditional_effects(list(universal_effect.conditional_effects))
            universal_effect.conditional_effects = set(compressed_effects)

    def _apply_inductive_rules(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Updates the literals that cannot be effects as well as updates the dependency set.

        :param grounded_action: the action that is currently being executed.
        :param next_state: the state following the action's execution.
        :param previous_state: the state prior to the action's execution.
        """
        self._update_observed_effects(grounded_action, previous_state, next_state)
        self._remove_not_antecedents(grounded_action, previous_state, next_state)

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        super()._add_new_action_preconditions(grounded_action)
        if grounded_action.name not in self.universals_map or len(self.universals_map[grounded_action.name]) == 0:
            super().add_new_action(grounded_action, previous_state, next_state)
            return

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
        self.logger.info(f"Updating the action {grounded_action.name} preconditions and effects.")
        super()._update_action_preconditions(grounded_action)
        if grounded_action.name not in self.universals_map or len(self.universals_map[grounded_action.name]) == 0:
            super().update_action(grounded_action, previous_state, next_state)
            return

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

        specific_types = self.universals_map[action_name] if action_name in self.universals_map and len(
            self.universals_map[action_name]) > 0 else []
        self.triplet_snapshot.create_snapshot(
            previous_state=previous_state, next_state=next_state, current_action=grounded_action,
            observation_objects=self.current_trajectory_objects,
            specific_types=specific_types)
        if action_name not in self.observed_actions:
            self._initialize_actions_dependencies(grounded_action)
            self._initialize_universal_dependencies(grounded_action)
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

        self.logger.debug(f"Finished handling the action {action_name}.")

    def construct_safe_actions(self) -> None:
        """Constructs the universal effects of the actions or a conservative version of them."""
        self.partial_domain.actions = {
            action_name: self.partial_domain.actions[action_name] for action_name in self.observed_actions}

        for action in self.partial_domain.actions.values():
            self.conditional_antecedents[action.name].remove_preconditions_literals(action.preconditions_str_set)
            super()._verify_and_construct_safe_conditional_effects(action)
            if action.name not in self.universals_map or len(self.universals_map[action.name]) == 0:
                continue

            self._remove_preconditions_from_antecedents(action)
            self._verify_and_construct_safe_universal_effects(action)
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

"""Module to learn action models from multi-agent trajectories with joint actions."""
import logging
import re
from typing import Dict, List, Tuple, Set, Optional

from pddl_plus_parser.models import Predicate, Domain, MultiAgentComponent, MultiAgentObservation, ActionCall, State, \
    GroundedPredicate, JointActionCall, CompoundPrecondition, SignatureType, PDDLType

from sam_learning.core import (LearnerDomain, extract_effects, LiteralCNF, LearnerAction, extract_predicate_data,
                               group_params_from_clause)
from sam_learning.learners.multi_agent_sam import MultiAgentSAM

from itertools import chain, combinations
from utilities import powerset, combine_groupings

BindingType = Dict[tuple[str, str], str]


class MASAMPlus(MultiAgentSAM):
    """Class designated to learning action models with macro actions
        from multi-agent trajectories with joint actions."""
    mapping: Dict[str,  BindingType]
    unsafe_actions_preconditions_map: Dict[str, CompoundPrecondition]
    def __init__(self, partial_domain: Domain, preconditions_fluent_map: Optional[Dict[str, List[str]]] = None):
        super().__init__(partial_domain, preconditions_fluent_map)
        self.mapping = {}
        self.unsafe_actions_preconditions_map = {}

    def _extract_predicate_from_clause(self, clause_element, mapping):
        """Helper function to extract predicate data and adapt it to the macro signature.

        :param clause_element: A tuple of (action_name, lifted_fluent).
        :param mapping: The macro-action signature.
        :return: Adapted predicate.
        """
        action_name, fluent = clause_element
        action_signature = self.partial_domain.actions[action_name].signature
        predicate = extract_predicate_data(action_signature, fluent, self.partial_domain.constants)
        return MacroActionParser.adapt_predicate_to_macro_mapping(mapping, predicate, action_name)

    def extract_relevant_lmas2(self) -> List[set[LearnerAction]]:
        """Extracts relevant action groups

        :return: a list of action groups
        """
        actions_set = set(self.partial_domain.actions.values())
        actions_powerset = powerset(actions_set)
        unsafe_actions = set(self.unsafe_actions_preconditions_map.keys())
        all_lmas = [lma for lma in actions_powerset if unsafe_actions.intersection(lma) and len(lma) > 1]
        relevant_lmas = []

        for lma in all_lmas:
            lma_action_names = list(map(lambda x: x.name, lma))
            found = False

            for fluent, fluent_cnf in self.literals_cnf.items():
                for clause in fluent_cnf.possible_lifted_effects:
                    all_actions_act = all(any(action == clause_action for clause_action, _ in clause)
                                          for action in lma_action_names)
                    all_atoms_act = all([action in lma_action_names for (action, _) in clause])

                    if all_actions_act and all_atoms_act:
                        relevant_lmas.append(lma)
                        found = True
                        break

                if found:
                    break

        return relevant_lmas

    def extract_relevant_lmas(self) -> List[set[LearnerAction]]:
        """Extracts relevant action groups

        :return: a list of action groups
        """
        action_list = [action for action in self.partial_domain.actions.values() if action.name in self.observed_actions]
        actions_set = set(action_list)
        action_names = set(self.observed_actions)
        unsafe_actions = action_names.difference(set(self.safe_actions))
        relevant_lmas = []

        for fluent, fluent_cnf in self.literals_cnf.items():
            for clause in fluent_cnf.possible_lifted_effects:
                clause_actions = {action_name for action_name, _ in clause}

                if len(clause) > 1 and not unsafe_actions.isdisjoint(clause_actions):
                    lma = {action for action in actions_set if action.name in clause_actions}

                    if lma not in relevant_lmas:
                        relevant_lmas.append(lma)

        return relevant_lmas

    def generate_possible_binding(self, lma_names: list[str]) -> List[set]:
        all_param_groups = [
            group_params_from_clause(clause)
            for fluent_cnf in self.literals_cnf.values()
            for clause in fluent_cnf.possible_lifted_effects
            if all(action in lma_names for action, _ in clause)
        ]

        flattened_groups = combine_groupings(all_param_groups)

        return flattened_groups

    def extract_effects_for_macro_from_cnf(self, lma_set: set[LearnerAction], param_grouping, mapping):
        lma_names = [lma.name for lma in lma_set]
        cnf_effects = []
        relevant_preconditions_str = {precondition.untyped_representation for action in lma_set for precondition
                                      in action.preconditions if isinstance(precondition, Predicate)}

        for fluent, fluent_cnf in self.literals_cnf.items():
            effects = fluent_cnf.extract_macro_action_effects(lma_names, relevant_preconditions_str, param_grouping)
            for effect_element in effects:
                cnf_effects.append(self._extract_predicate_from_clause(effect_element, mapping))

        # TODO find a neater way to remove duplicates
        unique_representations = {}
        unique_cnf_effects = []

        for effect in cnf_effects:
            if effect.untyped_representation not in unique_representations:
                unique_representations[effect.untyped_representation] = True
                unique_cnf_effects.append(effect)

        return unique_cnf_effects

    def extract_preconditions_for_macro_from_cnf(self, lma: set[LearnerAction], param_grouping, mapping):
        cnf_preconditions = []
        lma_names = [action.name for action in lma]

        for fluent, fluent_cnf in self.literals_cnf.items():
            preconditions = fluent_cnf.extract_macro_action_preconditions(lma_names, param_grouping)
            for precondition_element in preconditions:
                cnf_preconditions.append(self._extract_predicate_from_clause(precondition_element, mapping))

        new_precondition = CompoundPrecondition()
        for action in lma:
            preconditions = action.preconditions if action.name not in self.unsafe_actions_preconditions_map \
                else self.unsafe_actions_preconditions_map[action.name]

            for _, precondition in preconditions:
                if isinstance(precondition, Predicate):
                    cnf_preconditions.append(
                        self._extract_predicate_from_clause((action.name,
                                                            precondition.untyped_representation),
                                                            mapping))

            else:
                for _, precondition in action.preconditions:
                    if isinstance(precondition, Predicate):
                        cnf_preconditions.append(self._extract_predicate_from_clause((action.name,
                                                                                  precondition.untyped_representation),
                                                                                 mapping))

        for predicate in cnf_preconditions:
            new_precondition.add_condition(predicate)

        return new_precondition


    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        super()._remove_unobserved_actions_from_partial_domain()
        for action in self.partial_domain.actions.values():
            self.logger.debug("Constructing safe action for %s", action.name)
            action_preconditions = {precondition for precondition in
                                    action.preconditions if isinstance(precondition, Predicate)}
            if not self._is_action_safe(action, action_preconditions):
                self.logger.warning("Action %s is not safe to execute!", action.name)
                #TODO need to check if its copy or overriden next line
                self.unsafe_actions_preconditions_map[action.name] = action.preconditions
                action.preconditions = CompoundPrecondition()
                continue

            self.logger.debug("Action %s is safe to execute.", action.name)
            self.safe_actions.append(action.name)
            self.extract_effects_from_cnf(action, action_preconditions)


    def construct_safe_macro_actions(self) -> None:
        """Constructs the multi-agent actions that are safe to execute."""
        relevant_lmas = self.extract_relevant_lmas()

        for lma in relevant_lmas:
            lma_names = [action.name for action in lma]
            binding = self.generate_possible_binding(lma_names)
            mapper = MacroActionParser.generate_macro_mappings(binding, lma)

            macro_action_name = MacroActionParser.generate_macro_action_name(lma_names)
            macro_action_signature = MacroActionParser.generate_macro_action_signature(lma, mapper)
            macro_action_preconditions = self.extract_preconditions_for_macro_from_cnf(lma, binding, mapper)
            macro_action_effects = self.extract_effects_for_macro_from_cnf(lma, binding, mapper)

            macro_action = LearnerAction(macro_action_name, macro_action_signature)
            macro_action.preconditions = macro_action_preconditions
            macro_action.discrete_effects = macro_action_effects

            self.partial_domain.actions[macro_action.name] = macro_action
            self.safe_actions.append(macro_action.name)
            self.observed_actions.append(macro_action.name)
            self.mapping[macro_action.name] = mapper

    def learn_combined_action_model_with_macro_actions(
            self, observations: List[MultiAgentObservation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model with macro actions!")
        super().start_measure_learning_time()
        self._initialize_cnfs()

        super().deduce_initial_inequality_preconditions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                self.handle_multi_agent_trajectory_component(component)

        self.construct_safe_actions()
        self.construct_safe_macro_actions()
        self.logger.info("Finished learning the action model!")
        super().end_measure_learning_time()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report

    def extract_actions_from_macro_action(self, action_line: str) -> set[str]:
        param_pattern = re.compile(r'[^\s()]+')
        macro_name = param_pattern.findall(action_line)[0]

        if macro_name not in self.mapping:
            return {action_line}

        mapping = self.mapping[macro_name]
        action_names = {name for (name, param) in mapping}
        actions_dict = {}
        for action in action_names:
            actions_dict[action] = f'({action}'

        params_bound = param_pattern.findall(action_line)[1:]
        params_name = self.partial_domain.actions[macro_name].parameter_names

        for action in action_names:
            for param in self.partial_domain.actions[action].parameter_names:
                for param_name, param_bound in zip(params_name, params_bound):
                    if mapping[(action, param)] == param_name:
                        actions_dict[action] += f' {param_bound}'

        for action in action_names:
            actions_dict[action] += ')'

        return set(actions_dict.values())


class MacroActionParser:
    # Maybe should be a util class

    @staticmethod
    def _return_sub_type(type1: PDDLType, type2: PDDLType) -> PDDLType:
        if not type1:
            return type2

        if not type2:
            return type1

        if type1.is_sub_type(type2):
            return type1

        return type2

    @staticmethod
    def generate_macro_action_name(action_names: list[str]) -> str:
        return "-".join(action_names)

    @staticmethod
    def generate_macro_action_signature(actions: set[LearnerAction], mapping) -> SignatureType:
        all_params_dict = {}
        for action in actions:
            for param_name, param_type in action.signature.items():
                all_params_dict[(action.name, param_name)] = param_type

        action_signature: SignatureType = {}

        for key, param_type in all_params_dict.items():
            if key in mapping:
                param_name = mapping[key]
                if param_name not in action_signature:
                    action_signature[param_name] = param_type
                else:
                    action_signature[param_name] = MacroActionParser._return_sub_type(param_type,
                                                                                      action_signature[param_name])

        return action_signature

    @staticmethod
    def adapt_predicate_to_macro_mapping(mapping, predicate: Predicate, relevant_action):
        # Ensure the string is properly trimmed
        predicate_copy = predicate.copy()
        new_signature = {}
        for param_name, param_type in predicate.signature.items():
            new_name = mapping[(relevant_action, param_name)]
            new_signature[new_name] = param_type

        predicate_copy.signature = new_signature
        return predicate_copy

    # TODO should see what it gets. maybe Action rather than LearnerAction, but idk how the process goes.
    @staticmethod
    def extract_actions_from_macro_action(action_line: str, mapper) -> set[str]:
        param_pattern = re.compile(r'[^\s()]+')
        macro_name = param_pattern.findall(action_line)[0]

        if macro_name not in mapper:
            return {action_line}

        macro_action_rep, mapping = mapper[macro_name]
        action_names = {name for (name, param) in mapping}
        actions_dict = {}
        for action in action_names:
            actions_dict[action] = f'({action}'

        params_bound = param_pattern.findall(action_line)
        params_name = param_pattern.findall(macro_action_rep)

        for param_name, param_bound in zip(params_name[1:], params_bound[1:]):
            for (action, param) in mapping:
                if mapping[(action, param)] == param_name:
                    actions_dict[action] += f' {param_bound}'

        for action in action_names:
            actions_dict[action] += ')'

        return set(actions_dict.values())

    @staticmethod
    def adapt_fluent_str_to_macro_signature(signature, fluent_str, relevant_action):
        # Ensure the string is properly trimmed
        fluent_str = fluent_str.strip()

        # Improved regex to account for surrounding parentheses and spaces
        action_match = re.match(r'^\((\w+)\s+(.+)\)$', fluent_str)

        if not action_match:
            raise ValueError(f"Invalid fluent string format: '{fluent_str}'")

        action_name, params_str = action_match.groups()

        # Split the parameters by whitespace
        param_list = params_str.split()

        # Replace each parameter in the list according to the signature
        adapted_params = []
        for param in param_list:
            adapted_params.append(f'?{signature[(relevant_action, param[1:])]}')

        # Construct the adapted fluent string
        adapted_fluent = f"({action_name} {' '.join(adapted_params)})"

        return adapted_fluent

    @staticmethod
    def generate_macro_mappings(groupings: List[set], lma_set: set[LearnerAction]) -> BindingType:
        lma_names = [action.name for action in lma_set]
        param_bindings = {
            (action.name, param_name): f"{param_name}_{lma_names.index(action.name)}"
            for action in lma_set
            for param_name in action.parameter_names
        }

        for group in groupings:
            if len(group) > 1:
                new_param_name = '?' + ''.join([param[1:] for _, param in group])
                for action_name, param in group:
                    param_bindings[(action_name, param)] = new_param_name

        return param_bindings

"""Module to learn action models with macro actions from multi-agent trajectories with joint actions."""
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import networkx as nx
from pddl_plus_parser.models import Predicate, Domain, MultiAgentObservation, CompoundPrecondition

from sam_learning.core import LearnerDomain, LearnerAction, extract_predicate_data, PGType, group_params_from_clause
from sam_learning.learners.multi_agent_sam import MultiAgentSAM
from utilities import NegativePreconditionPolicy, MacroActionParser, BindingType, MappingElement


def combine_groupings(grouping: List[set]) -> List[set]:
    """
    Combine sets that share common elements across all groupings into new groupings.
    """
    grouping_graph = nx.Graph()

    # For each group, add edges between all tuples in that group
    for group in grouping:
        if len(group) > 1:
            for node1, node2 in combinations(group, 2):
                grouping_graph.add_edge(node1, node2)  # Connect nodes (tuples)
        else:
            grouping_graph.add_node(next(iter(group)))  # takes first element

    # Find the connected components (merged sets)
    maximal_grouping = list(nx.connected_components(grouping_graph))

    # Convert the set of tuples back to a list of sets
    return maximal_grouping
    # return [set(component) for component in maximal_grouping]


class MASAMPlus(MultiAgentSAM):
    """Class designated to learning action models with macro actions
        from multi-agent trajectories with joint actions."""

    mapping: Dict[str, MappingElement]
    unsafe_actions_preconditions_map: Dict[str, CompoundPrecondition]

    def __init__(
        self,
        partial_domain: Domain,
        preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.no_remove,
    ):
        super().__init__(partial_domain, preconditions_fluent_map, negative_precondition_policy=negative_precondition_policy)
        self.mapping = {}
        self.unsafe_actions_preconditions_map = {}

    def _remove_unsafe_actions_from_partial_domain(self):
        """Removes the actions that were not observed from the partial domain."""
        self.logger.debug("Removing unobserved actions from the partial domain")
        actions_to_remove = [action for action in self.partial_domain.actions if action in self.unsafe_actions_preconditions_map]
        for action in actions_to_remove:
            self.partial_domain.actions.pop(action)

    def _extract_predicate_from_clause_and_adapt_to_macro(self, clause_element: Tuple[str, str], mapping: BindingType) -> Predicate:
        """Helper function to extract predicate data and adapt it to the macro signature.

        :param clause_element: A tuple of (action_name, lifted_fluent).
        :param mapping: The macro-action signature.
        :return: Adapted predicate.
        """
        action_name, fluent = clause_element
        action_signature = self.partial_domain.actions[action_name].signature
        predicate = extract_predicate_data(action_signature, fluent, self.partial_domain.constants)
        return MacroActionParser.adapt_predicate_to_macro_mapping(mapping, predicate, action_name)

    def extract_relevant_action_groups(self) -> List[set[LearnerAction]]:
        """Extracts relevant action groups

        :return: a list of action groups
        """
        action_list = [action for action in self.partial_domain.actions.values() if action.name in self.observed_actions]
        actions_set = set(action_list)
        action_names = set(self.observed_actions)
        unsafe_actions = action_names.difference(set(self.safe_actions))
        action_groups = []

        for fluent, fluent_cnf in self.literals_cnf.items():
            for clause in fluent_cnf.possible_lifted_effects:
                clause_actions = {action_name for action_name, _ in clause}

                if len(clause) > 1 and not unsafe_actions.isdisjoint(clause_actions):
                    action_group = {action for action in actions_set if action.name in clause_actions}

                    if action_group not in action_groups:
                        action_groups.append(action_group)

        return action_groups

    def extract_relevant_parameter_groupings(self, action_group_names: list[str]) -> List[PGType]:
        """Extracts relevant parameter groups
            This implementation only extracts one such possible parameter groups
        """
        all_param_groups = [
            param_set
            for fluent_cnf in self.literals_cnf.values()
            for clause in fluent_cnf.possible_lifted_effects
            if all(action in action_group_names for action, _ in clause)
            for param_set in group_params_from_clause(clause)
        ]

        flattened_groups = combine_groupings(all_param_groups)

        return [flattened_groups]

    def extract_effects_for_macro_from_cnf(self, lma_set: set[LearnerAction], param_grouping: PGType, mapping: BindingType):
        lma_names = [lma.name for lma in lma_set]
        cnf_effects = []
        relevant_preconditions_str = {
            precondition.untyped_representation for action in lma_set for precondition in action.preconditions if isinstance(precondition, Predicate)
        }

        for fluent, fluent_cnf in self.literals_cnf.items():
            effects = fluent_cnf.extract_macro_action_effects(lma_names, relevant_preconditions_str, param_grouping)
            for effect_element in effects:
                cnf_effects.append(self._extract_predicate_from_clause_and_adapt_to_macro(effect_element, mapping))

        # TODO find a neater way to remove duplicates
        unique_representations = set()
        unique_cnf_effects = []

        for effect in cnf_effects:
            if effect.untyped_representation not in unique_representations:
                unique_representations.add(effect.untyped_representation)
                unique_cnf_effects.append(effect)

        return unique_cnf_effects

    def extract_preconditions_for_macro_from_cnf(self, action_group: set[LearnerAction], param_grouping: PGType, mapping: BindingType):
        new_precondition = CompoundPrecondition()
        lma_names = [action.name for action in action_group]

        # extracting preconditions from the cnfs, including vague effects that become preconditions
        for fluent, fluent_cnf in self.literals_cnf.items():
            preconditions = fluent_cnf.extract_macro_action_preconditions(lma_names, param_grouping)
            for precondition_element in preconditions:
                precondition = self._extract_predicate_from_clause_and_adapt_to_macro(precondition_element, mapping)
                new_precondition.add_condition(precondition)

        # taking the already existing learned preconditions into account
        for action in action_group:
            preconditions = (
                action.preconditions
                if action.name not in self.unsafe_actions_preconditions_map
                else self.unsafe_actions_preconditions_map[action.name]
            )

            for _, precondition in preconditions:
                if isinstance(precondition, Predicate):
                    precondition_to_add = self._extract_predicate_from_clause_and_adapt_to_macro(
                        clause_element=(action.name, precondition.untyped_representation), mapping=mapping
                    )
                    new_precondition.add_condition(precondition_to_add)

            # Todo remove the comment after seeing all experiments prove good results
            # else:
            #     for _, precondition in action.preconditions:
            #         if isinstance(precondition, Predicate):
            #             cnf_preconditions.append(self._extract_predicate_from_clause((action.name,
            #                                                                       precondition.untyped_representation),
            #                                                                      mapping))

        return new_precondition

    def construct_safe_actions(self) -> None:
        """Constructs the single-agent actions that are safe to execute."""
        super()._remove_unobserved_actions_from_partial_domain()
        for action in self.partial_domain.actions.values():
            self.logger.debug("Constructing safe action for %s", action.name)
            action_preconditions = {precondition for precondition in action.preconditions if isinstance(precondition, Predicate)}
            if not self._is_action_safe(action, action_preconditions):
                self.logger.warning("Action %s is not safe to execute!", action.name)
                self.unsafe_actions_preconditions_map[action.name] = action.preconditions
                action.preconditions = CompoundPrecondition()
                continue

            self.logger.debug("Action %s is safe to execute.", action.name)
            self.safe_actions.append(action.name)
            self.extract_effects_from_cnf(action, action_preconditions)

    def construct_safe_macro_actions(self) -> None:
        """Constructs the multi-agent-plus macro actions that are safe to execute."""
        action_groups = self.extract_relevant_action_groups()

        for action_group in action_groups:
            action_group_names = list(sorted(action.name for action in action_group))
            parameter_groupings = self.extract_relevant_parameter_groupings(action_group_names)
            for parameter_grouping in parameter_groupings:
                mapper = MacroActionParser.generate_macro_mappings(parameter_grouping, action_group)

                macro_action_name = MacroActionParser.generate_macro_action_name(action_group_names)
                macro_action_signature = MacroActionParser.generate_macro_action_signature(action_group, mapper)
                macro_action_preconditions = self.extract_preconditions_for_macro_from_cnf(action_group, parameter_grouping, mapper)
                macro_action_effects = self.extract_effects_for_macro_from_cnf(action_group, parameter_grouping, mapper)

                macro_action = LearnerAction(macro_action_name, macro_action_signature)
                macro_action.preconditions = macro_action_preconditions
                macro_action.discrete_effects = macro_action_effects

                self.partial_domain.actions[macro_action.name] = macro_action
                self.safe_actions.append(macro_action.name)
                self.observed_actions.append(macro_action.name)
                self.mapping[macro_action.name] = (macro_action.parameter_names, mapper)

    def learn_combined_action_model_with_macro_actions(
        self, observations: List[MultiAgentObservation]
    ) -> Tuple[LearnerDomain, Dict[str, str], Dict[str, MappingElement]]:
        """Learn the SAFE action model from the input multi-agent trajectories.

        :param observations: the multi-agent observations.
        :return: a domain containing the actions that were learned, including macro actions.
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
        self._remove_unsafe_actions_from_partial_domain()
        self.handle_negative_preconditions_policy()
        self.logger.info("Finished learning the action model!")
        super().end_measure_learning_time()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report, self.mapping

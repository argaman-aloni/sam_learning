"""Module to learn action models with macro actions from multi-agent trajectories with joint actions."""

from itertools import combinations, chain
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
from pddl_plus_parser.models import Predicate, Domain, MultiAgentObservation, CompoundPrecondition

from sam_learning.core import LearnerDomain, LearnerAction, extract_predicate_data, PGType, group_params_from_clause
from sam_learning.learners.multi_agent_sam import MultiAgentSAM
from utilities import NegativePreconditionPolicy, MacroActionParser, BindingType, MappingElement
import matplotlib.pyplot as plt


def visualize_grouping_graph(grouping_graph: nx.Graph) -> None:
    """Plots the binding graph.

    :param grouping_graph: the graph containing the binding groups.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(grouping_graph)  # Layout for visualization
    nx.draw(grouping_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500, font_size=12)
    plt.title("Graph Representation of Groupings")
    plt.show()


def combine_groupings(binding_groups_sets: PGType, should_plot: bool = False) -> PGType:
    """
    Combine sets that share common elements across all groupings into new groupings.
    """
    grouping_graph = nx.Graph()

    # For each group, add edges between all tuples in that group
    for action_binding_set in binding_groups_sets:
        if len(action_binding_set) > 1:
            for node1, node2 in combinations(action_binding_set, 2):
                grouping_graph.add_edge(node1, node2)  # Connect nodes (tuples)
        else:
            grouping_graph.add_node(next(iter(action_binding_set)))  # takes first element

    if should_plot:
        visualize_grouping_graph(grouping_graph)

    # Find the connected components (merged sets)
    maximal_grouping = list(nx.connected_components(grouping_graph))
    # Convert the set of tuples back to a list of sets
    return maximal_grouping


def generate_supersets_of_actions(action_groups: List[Set[str]]) -> List[Set[str]]:
    """Generate the superset of all possible action combinations for the macro-action construction.

    :param action_groups: the action groups containing the actions that are relevant for the macro-action construction.
    :return: a list of action groups and their supersets.
    """
    power_sets = []
    for r in range(1, len(action_groups) + 1):
        action_combinations = list(combinations(action_groups, r))
        power_sets.extend([set(chain.from_iterable(a)) for a in action_combinations])

    return power_sets


class MASAMPlus(MultiAgentSAM):
    """Class designated to learning action models with macro actions
    from multi-agent trajectories with joint actions."""

    mapping: Dict[str, MappingElement]

    def __init__(
        self,
        partial_domain: Domain,
        preconditions_fluent_map: Optional[Dict[str, List[str]]] = None,
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.no_remove,
    ):
        super().__init__(partial_domain, preconditions_fluent_map, negative_precondition_policy=negative_precondition_policy)
        self.mapping = {}

    def _extract_predicate_from_clause_and_adapt_to_macro(self, action_name: str, fluent: str, mapping: BindingType) -> Predicate:
        """Helper function to extract predicate data and adapt it to the macro signature.

        :param action_name: The name of the action in the CNF.
        fluent: the parameter bound literal that belongs to the action
        :param mapping: The macro-action signature.
        :return: Adapted predicate.
        """
        action_signature = self.partial_domain.actions[action_name].signature
        predicate = extract_predicate_data(action_signature, fluent, self.partial_domain.constants)
        return MacroActionParser.adapt_predicate_to_macro_mapping(mapping, predicate, action_name)

    def extract_relevant_action_groups(self) -> List[Set[LearnerAction]]:
        """Extracts the action sets containing at least one unsafe action.

        These action groups (sets) are the ones that are relevant for the macro-action construction.

        :return: a list of action groups.
        """
        action_groups = []
        for lifted_literal, cnf in self.literals_cnf.items():
            for clause in cnf.possible_lifted_effects:
                clause_actions = {action_name for action_name, _ in clause}

                if len(clause) > 1 and not self._unsafe_actions.isdisjoint(clause_actions):
                    action_group = {action.name for action in self.partial_domain.actions.values() if action.name in clause_actions}

                    if action_group not in action_groups:
                        action_groups.append(action_group)

        super_set_groups = generate_supersets_of_actions(action_groups)

        action_candidates = [{self.partial_domain.actions[action] for action in group} for group in super_set_groups]
        return action_candidates

    def extract_relevant_parameter_groupings(self, action_group_names: List[str]) -> List[PGType]:
        """Extracts relevant parameter groups, that is, the parameters of the actions in the action group.

        Note:
            This implementation only extracts one such possible parameter groups

        :param action_group_names: the names of the actions in the action group.
        """
        all_param_groups = [
            param_set
            for fluent_cnf in self.literals_cnf.values()
            for clause in fluent_cnf.possible_lifted_effects
            if all([action in action_group_names for action, _ in clause])
            for param_set in group_params_from_clause(clause)
        ]

        flattened_groups = combine_groupings(all_param_groups)

        return [flattened_groups]

    def extract_effects_for_macro_from_cnf(
        self, lma_set: Set[LearnerAction], param_grouping: PGType, mapping: BindingType
    ) -> Set[Predicate]:
        """Extract the effects of the macro action containing the input single-agent actions.

        :param lma_set: the single agent actions contained in the macro action.
        :param param_grouping: the parameter grouping of the macro action.
        :param mapping: the mapping between the macro action and the single-agent actions.
        :return: the set of effects of the macro action.
        """
        lma_names = [lma.name for lma in lma_set]
        cnf_effects = set()
        relevant_preconditions_str = set()
        for action in lma_set:
            relevant_preconditions_str.update(
                {precondition.untyped_representation for _, precondition in action.preconditions if isinstance(precondition, Predicate)}
            )

        for fluent, fluent_cnf in self.literals_cnf.items():
            effects = fluent_cnf.extract_macro_action_effects(lma_names, relevant_preconditions_str, param_grouping)
            for action, pb_literal in effects:
                cnf_effects.add(self._extract_predicate_from_clause_and_adapt_to_macro(action, pb_literal, mapping))

        return cnf_effects

    def extract_preconditions_for_macro_from_cnf(
        self, action_group: Set[LearnerAction], param_grouping: PGType, mapping: BindingType
    ) -> CompoundPrecondition:
        """Extracts the preconditions for the newly constructed macro action.

        :param action_group: the actions composing the macro action.
        :param param_grouping: the parameters that are grouped together and represent the joint execution of the actions.
        :param mapping: the mapping between the macro action and the single-agent actions.
        :return:
        """
        new_precondition = CompoundPrecondition()
        lma_names = [action.name for action in action_group]

        # extracting preconditions from the CNFs, including vague effects that become preconditions
        for fluent, fluent_cnf in self.literals_cnf.items():
            preconditions = fluent_cnf.extract_macro_action_preconditions(lma_names, param_grouping)
            for action, pb_literal in preconditions:
                precondition = self._extract_predicate_from_clause_and_adapt_to_macro(action, pb_literal, mapping)
                new_precondition.add_condition(precondition)

        # taking the already existing learned preconditions into account
        for action in action_group:
            for _, precondition in action.preconditions:
                if isinstance(precondition, Predicate):
                    precondition_to_add = self._extract_predicate_from_clause_and_adapt_to_macro(
                        action_name=action.name, fluent=precondition.untyped_representation, mapping=mapping
                    )
                    new_precondition.add_condition(precondition_to_add)

        return new_precondition

    def construct_safe_macro_actions(self) -> None:
        """Constructs the multi-agent-plus macro actions that are safe to execute."""
        action_groups = self.extract_relevant_action_groups()

        macro_action_names = []
        for action_group in action_groups:
            action_group_names = sorted([action.name for action in action_group])
            parameter_groupings = self.extract_relevant_parameter_groupings(action_group_names)
            for parameter_grouping in parameter_groupings:
                mapper = MacroActionParser.generate_macro_mappings(parameter_grouping, action_group)

                macro_action_name = MacroActionParser.generate_macro_action_name(action_group_names, macro_action_names)
                macro_action_names.append(macro_action_name)
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

        for unsafe_action in self._unsafe_actions:
            self.partial_domain.actions.pop(unsafe_action)

    def learn_combined_action_model_with_macro_actions(
        self, observations: List[MultiAgentObservation]
    ) -> Tuple[Domain, Dict[str, str], Dict[str, MappingElement]]:
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

        self.construct_safe_actions(should_remove_actions=False)
        self.construct_safe_macro_actions()
        self.handle_negative_preconditions_policy()
        self.logger.info("Finished learning the action model!")
        super().end_measure_learning_time()
        learning_report = super()._construct_learning_report()
        return self.partial_domain, learning_report, self.mapping

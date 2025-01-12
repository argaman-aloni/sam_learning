import re
from typing import List, Set, Dict, Tuple

from pddl_plus_parser.models import Predicate, SignatureType, PDDLType

from sam_learning.core import LearnerAction

BindingType = Dict[Tuple[str, str], str]
MappingElement = Tuple[List[str], BindingType]


class MacroActionParser:
    # Maybe should be a util class

    @staticmethod
    def _return_sub_type(type1: PDDLType, type2: PDDLType) -> PDDLType:
        """
        function to return the subtype between two types
        """
        if not type1:
            return type2

        if not type2:
            return type1

        if type1.is_sub_type(type2):
            return type1

        return type2

    @staticmethod
    def generate_macro_action_name(action_names: List[str]) -> str:
        """ function to generate a name for macro actions given the names of the actions """
        return "-".join(action_names)

    @staticmethod
    def generate_macro_action_signature(actions: Set[LearnerAction], mapping: BindingType) -> SignatureType:
        """
        generates a signature for a macro action, taking care of the types of the parameters, and their names

        :param actions: a set of actions consisting the macro action
        :param mapping: maps between micro actions param names to macro action param names
        NOTE:, in cases of 2 or more micro actions parameters converging in 1 macro parameter, the macro param type
                will be the subtype of all micro parameters.
        """

        # For orders sake we shall sort the actions by name so the parameters are ordered the same way always.
        actions_list = sorted(actions, key=lambda action: action.name)
        all_params_dict = {}
        for action in actions_list:
            for param_name, param_type in action.signature.items():
                all_params_dict[(action.name, param_name)] = param_type

        macro_action_signature: SignatureType = {}

        for key, param_type in all_params_dict.items():
            if key in mapping:
                macro_action_param_name = mapping[key]
                if macro_action_param_name not in macro_action_signature:
                    macro_action_signature[macro_action_param_name] = param_type
                else:
                    macro_action_signature[macro_action_param_name] = MacroActionParser._return_sub_type(
                        param_type, macro_action_signature[macro_action_param_name]
                    )

        return macro_action_signature

    @staticmethod
    def adapt_predicate_to_macro_mapping(mapping: BindingType, predicate: Predicate, relevant_action) -> Predicate:
        # Ensure the string is properly trimmed
        predicate_copy = predicate.copy()
        new_signature = {}
        for param_name, param_type in predicate.signature.items():
            new_name = mapping[(relevant_action, param_name)]
            new_signature[new_name] = param_type

        predicate_copy.signature = new_signature
        return predicate_copy

    @staticmethod
    def generate_macro_mappings(groupings: List[set], lma_set: Set[LearnerAction]) -> BindingType:
        """
        returns a mapping between macro action names and macro action names
        the orders of the keys conveys important information:
        1)ordered according to action name
        2)parameters ordered according to micro action parameters order.
        """
        lma_names = list(sorted(action.name for action in lma_set))
        lma_set_ordered = sorted(lma_set, key=lambda action: action.name)
        param_bindings = {
            (action.name, param_name): f"{param_name}_{lma_names.index(action.name)}"
            for action in lma_set_ordered
            for param_name in action.parameter_names
        }

        for group in groupings:
            if len(group) > 1:
                new_param_name = "?" + "".join(sorted(param[1:] for _, param in group))
                for action_name, param in group:
                    param_bindings[(action_name, param)] = new_param_name

        return param_bindings

    @staticmethod
    def extract_actions_from_macro_action(action_line: str, mapper: Dict[str, MappingElement]) -> Set[str]:
        """
        This function replaces a single line consisting of macro action, with several micro actions.

        :param action_line: the macro action line from the solution file to be replaced
        :param mapper: the macro actions mapping of the MA-SAM+ learned domain.
        """
        param_pattern = re.compile(r"[^\s()]+")
        macro_name = param_pattern.findall(action_line)[0]

        if macro_name not in mapper:
            return {action_line}

        params_name, mapping = mapper[macro_name]
        action_names = {name for (name, param) in mapping}
        actions_dict = {}
        for action in action_names:
            actions_dict[action] = f"({action}"

        params_bound = param_pattern.findall(action_line)[1:]

        for action in action_names:
            for (action_name, param) in mapping:
                if action_name != action:
                    continue

                for param_name, param_bound in zip(params_name, params_bound):
                    if mapping[(action, param)] == param_name:
                        actions_dict[action] += f" {param_bound}"
                        break

        for action in action_names:
            actions_dict[action] += ")"

        return set(actions_dict.values())

"""Module containing the datatype of the output domain that the learning algorithms return."""
from collections import defaultdict
from typing import Set, List, Dict, Tuple

from pddl_plus_parser.models import SignatureType, Predicate, PDDLType, PDDLConstant, PDDLFunction, Domain

from .learning_types import ConditionType

DISJUNCTIVE_PRECONDITIONS_REQ = ":disjunctive-preconditions"


class LearnerAction:
    """Class representing an action that the learning algorithm outputs."""

    name: str
    signature: SignatureType
    positive_preconditions: Set[Predicate]
    negative_preconditions: Set[Predicate]
    inequality_preconditions: Set[Tuple[str, str]]  # set of parameters names that should not be equal.
    numeric_preconditions: Tuple[List[str], ConditionType]  # tuple mapping the numeric preconditions to their type.
    add_effects: Set[Predicate]
    delete_effects: Set[Predicate]
    numeric_effects: List[str]  # set of the strings representing the equations creating the numeric effect.

    def __init__(self, name: str, signature: SignatureType):
        self.name = name
        self.signature = signature
        self.positive_preconditions = set()
        self.negative_preconditions = set()
        self.inequality_preconditions = set()
        self.numeric_preconditions = tuple()
        self.add_effects = set()
        self.delete_effects = set()
        self.numeric_effects = []

    def __str__(self):
        signature_str_items = []
        for parameter_name, parameter_type in self.signature.items():
            signature_str_items.append(f"{parameter_name} - {str(parameter_type)}")

        signature_str = " ".join(signature_str_items)
        return f"({self.name} {signature_str})"

    @property
    def parameter_names(self) -> List[str]:
        return list(self.signature.keys())

    def _signature_to_pddl(self) -> str:
        """Converts the action's signature to the PDDL format.

        :return: the PDDL format of the signature.
        """
        signature_str_items = []
        for parameter_name, parameter_type in self.signature.items():
            signature_str_items.append(f"{parameter_name} - {str(parameter_type)}")

        signature_str = " ".join(signature_str_items)
        return f"({signature_str})"

    def _extract_inequality_preconditions(self) -> str:
        """Extracts the inequality preconditions from the learned action.

        :return: the inequality precondition of the action.
        """
        inequality_precondition_str = ""
        if len(self.inequality_preconditions) > 0:
            inequality_precondition_str = " ".join(f"(not (= {obj[0]} {obj[1]}))" for obj in
                                                   self.inequality_preconditions)
            inequality_precondition_str += "\n"
        return inequality_precondition_str

    def _extract_numeric_preconditions(self, positive_preconditions, precondition_str) -> str:
        """Extract the numeric preconditions from the action.

        :param positive_preconditions: the positive predicates to append to the string.
        :param precondition_str: the precondition string up to this point.
        :return: the string containing the numeric preconditions.
        """
        numeric_preconditions = self.numeric_preconditions[0]
        conditions_type = self.numeric_preconditions[1]
        numeric_preconditions_str = "\t\t\n".join(numeric_preconditions)

        if conditions_type == ConditionType.disjunctive:
            numeric_preconditions_str = f"(or {numeric_preconditions_str})"

        return f"(and {' '.join(positive_preconditions)}\n" \
               f"\t\t{precondition_str}" \
               f"\t\t{numeric_preconditions_str})"

    def _preconditions_to_pddl(self) -> str:
        """Converts the action's preconditions to the needed PDDL format.

        :return: the preconditions in PDDL format.
        """
        positive_preconditions = [precond.untyped_representation for precond in self.positive_preconditions]
        precondition_str = self._extract_inequality_preconditions()
        if len(self.numeric_preconditions) > 0:
            return self._extract_numeric_preconditions(positive_preconditions, precondition_str)

        return f"(and {' '.join(positive_preconditions)} {precondition_str})"

    def _effects_to_pddl(self) -> str:
        """Converts the effects to the needed PDDL format.

        :return: the PDDL format of the effects.
        """
        add_effects = [effect.untyped_representation for effect in self.add_effects]
        delete_effects = [effect.untyped_representation for effect in self.delete_effects]
        delete_effects_str = ""
        if len(delete_effects) > 0:
            delete_effects_str = " ".join([f"(not {effect})" for effect in delete_effects])

        if len(self.numeric_effects) > 0:
            numeric_effects = "\t\t\n".join([effect for effect in self.numeric_effects])
            return f"(and {' '.join(add_effects)} {delete_effects_str}\n" \
                   f"\t\t{numeric_effects}\n)"

        return f"(and {' '.join(add_effects)} {delete_effects_str})"

    def to_pddl(self) -> str:
        """Returns the PDDL string representation of the action.

        :return: the PDDL string representing the action.
        """
        return f"(:action {self.name}\n" \
               f"\t:parameters {self._signature_to_pddl()}\n" \
               f"\t:precondition {self._preconditions_to_pddl()}\n" \
               f"\t:effect {self._effects_to_pddl()})\n"


class LearnerDomain:
    """Class representing the domain that is to be learned by the action model learning algorithm."""

    name: str
    requirements: List[str]
    types: Dict[str, PDDLType]
    constants: Dict[str, PDDLConstant]
    predicates: Dict[str, Predicate]
    functions: Dict[str, PDDLFunction]
    actions: Dict[str, LearnerAction]

    # processes: Dict[str, Action] - TBD
    # events: Dict[str, Action] - TBD

    def __init__(self, domain: Domain):
        self.name = domain.name
        self.requirements = domain.requirements
        if DISJUNCTIVE_PRECONDITIONS_REQ not in self.requirements:
            self.requirements.append(DISJUNCTIVE_PRECONDITIONS_REQ)

        self.types = domain.types
        self.constants = domain.constants
        self.predicates = domain.predicates
        self.functions = domain.functions
        self.actions = {}
        for action_name, action_object in domain.actions.items():
            self.actions[action_name] = LearnerAction(name=action_name, signature=action_object.signature)

    def __str__(self):
        return (
                "< Domain definition: %s\n Requirements: %s\n Predicates: %s\n Functions: %s\n Actions: %s\n "
                "Constants: %s >"
                % (
                    self.name,
                    [req for req in self.requirements],
                    [str(p) for p in self.predicates.values()],
                    [str(f) for f in self.functions.values()],
                    [str(a) for a in self.actions.values()],
                    [str(c) for c in self.constants],
                )
        )

    def _types_to_pddl(self) -> str:
        """Converts the types to a PDDL string.

        :return: the PDDL string representing the types.
        """
        parent_child_map = defaultdict(list)
        for type_name, type_obj in self.types.items():
            if type_name == "object":
                continue

            parent_child_map[type_obj.parent.name].append(type_name)

        types_strs = []
        for parent_type, children_types in parent_child_map.items():
            types_strs.append(f"\t{' '.join(children_types)} - {parent_type}")

        return "\n".join(types_strs)

    def _constants_to_pddl(self) -> str:
        """Converts the constants to a PDDL string.

        :return: the PDDL string representing the constants.
        """
        same_type_constant = defaultdict(list)
        for const_name, constant in self.constants.items():
            if const_name == "object":
                continue

            same_type_constant[constant.type.name].append(const_name)

        types_strs = []
        for constant_type_name, constant_objects in same_type_constant.items():
            types_strs.append(f"\t{' '.join(constant_objects)} - {constant_type_name}")

        return "\n".join(types_strs)

    def _functions_to_pddl(self) -> str:
        """Converts the functions to PDDL format.

        :return: the PDDL format of the functions.
        """
        return "\n\t".join([str(f) for f in self.functions.values()])

    def to_pddl(self) -> str:
        """Converts the domain into a PDDL string format.

        :return: the PDDL string representing the domain.
        """
        predicates = "\n\t".join([str(p) for p in self.predicates.values()])
        actions = "\n".join(action.to_pddl() for action in self.actions.values())
        constants = f"(:constants {self._constants_to_pddl()}\n)\n\n" if len(self.constants) > 0 else ""
        functions = f"(:functions {self._functions_to_pddl()}\n)\n\n" if len(self.functions) > 0 else ""
        return f"(define (domain {self.name})\n" \
               f"(:requirements {' '.join(self.requirements)})\n" \
               f"(:types {self._types_to_pddl()}\n)\n\n" \
               f"(:predicates {predicates}\n)\n\n" \
               f"{constants}" \
               f"{functions}" \
               f"{actions}\n)"

"""class representing the datatype of the learner domain since it differs from the original domain in terms of the actions."""
from typing import Set, List, Dict, Tuple

from pddl_plus_parser.models import SignatureType, Predicate, PDDLType, PDDLConstant, PDDLFunction, Domain

from sam_learning.core import ConditionType


class LearnerAction:
    """Class representing an instantaneous action in a PDDL+ problems."""

    name: str
    signature: SignatureType
    positive_preconditions: Set[Predicate]
    negative_preconditions: Set[Predicate]
    numeric_preconditions: Tuple[List[str], ConditionType]
    add_effects: Set[Predicate]
    delete_effects: Set[Predicate]
    numeric_effects: List[str]

    def __init__(self, name: str, signature: SignatureType):
        self.name = name
        self.signature = signature
        self.positive_preconditions = set()
        self.negative_preconditions = set()
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
        """

        :return:
        """
        signature_str_items = []
        for parameter_name, parameter_type in self.signature.items():
            signature_str_items.append(f"{parameter_name} - {str(parameter_type)}")

        signature_str = " ".join(signature_str_items)
        return f"({signature_str})"

    def _preconditions_to_pddl(self) -> str:
        """

        :return:
        """
        positive_preconditions = [precond.untyped_representation for precond in self.positive_preconditions]
        # TODO: Handle numeric preconditions.
        return f"(and {' '.join(positive_preconditions)})"

    def _effects_to_pddl(self) -> str:
        add_effects = [effect.untyped_representation for effect in self.add_effects]
        delete_effects = [effect.untyped_representation for effect in self.delete_effects]
        delete_effects_str = ""
        if len(delete_effects) > 0:
            delete_effects_str = f"(not {' '.join(delete_effects)})"

        # TODO: Handle numeric effects.
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

    def to_pddl(self) -> str:
        predicates = "\n\t".join([p.untyped_representation for p in self.predicates.values()])
        actions = "\n".join(action.to_pddl() for action in self.actions.values())
        return f"(define (domain: {self.name})\n" \
               f"(:requirements {' '.join(self.requirements)})\n" \
               f"(:predicates {predicates}\n)\n\n" \
               f"{actions}\n)"

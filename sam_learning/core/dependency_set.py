"""Module representing the dependency set of an action."""
import itertools
import logging
from typing import Set, Dict, List, Optional, Union

from pddl_plus_parser.models import Predicate, Precondition, SignatureType, PDDLConstant

from sam_learning.core.conditional_sam_utilities import extract_predicate_data

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1


def create_antecedents_combination(antecedents: Set[str], max_antecedents_size: int,
                                   exclude_literals: Optional[Set[str]] = None) -> List[Set[str]]:
    """Creates all possible subset combinations of antecedents.

    :param antecedents: the list of antecedents that may be trigger for conditional effects.
    :param max_antecedents_size: the maximal size of the antecedents' combination.
    :param exclude_literals: the literals to exclude from the antecedents combinations.
    :return: all possible subsets of the antecedents up to the given size.
    """
    antecedents_combinations = []
    antecedents_to_use = antecedents - exclude_literals if exclude_literals is not None else antecedents
    for subset_size in range(1, max_antecedents_size + 1):
        possible_combinations = [set(combination) for combination in itertools.combinations(
            antecedents_to_use, subset_size)]
        for combination in possible_combinations:
            antecedents_combinations.append(combination)

    return antecedents_combinations


class DependencySet:
    """Class representing the dependency set of an action."""
    possible_antecedents: Dict[str, List[Set[str]]]
    max_antecedents: int
    action_signature: SignatureType
    domain_constants: Dict[str, PDDLConstant]
    logger: logging.Logger

    def __init__(self, max_size_antecedents: int, action_signature: SignatureType,
                 domain_constants: Dict[str, PDDLConstant]):
        self.possible_antecedents = {}
        self.max_antecedents = max_size_antecedents
        self.action_signature = action_signature
        self.domain_constants = domain_constants
        self.logger = logging.getLogger(__name__)

    def _extract_superset_dependencies(self, literal: str, dependencies_to_remove: Set[str]) -> List[Set[str]]:
        """Extracts the superset dependencies of the given literal.

        :param literal: the literal to check.
        :param dependencies_to_remove: the dependencies to remove.
        :return: all supersets that include a member of the dependencies to remove.
        """
        self.logger.debug(f"Extracting superset dependencies for literal {literal}")
        superset_dependencies = []
        for antecedents_conjunction in self.possible_antecedents[literal]:
            for dependency_to_remove in dependencies_to_remove:
                if dependency_to_remove in antecedents_conjunction:
                    superset_dependencies.append(antecedents_conjunction)

        return superset_dependencies

    @staticmethod
    def _negate_predicates(antecedents: Set[str]) -> Union[Set[str], str]:
        """Negates the given predicates.

        :param antecedents: the predicates to negate.
        :return: the negated predicates.
        """
        negated_predicates = set()
        for antecedent in antecedents:
            negated_predicates.add(antecedent[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX] if
                                   antecedent.startswith(NOT_PREFIX) else f"{NOT_PREFIX} {antecedent})")

        if len(negated_predicates) == 1:
            return negated_predicates.pop()

        return negated_predicates

    def _create_negated_antecedents(self, antecedents: Set[str]) -> Union[Predicate, Precondition]:
        """Creates the negated antecedents.

        :param antecedents: the antecedents to negate.
        :return: the negated antecedents as a combined string.
        """
        negated_predicates = self._negate_predicates(antecedents)
        if isinstance(negated_predicates, str):
            return extract_predicate_data(action_signature=self.action_signature,
                                          predicate_str=negated_predicates,
                                          domain_constants=self.domain_constants)

        antecedents_disjunction = Precondition("or")
        for antecedent in negated_predicates:
            lifted_antecedent = extract_predicate_data(action_signature=self.action_signature,
                                                       predicate_str=antecedent,
                                                       domain_constants=self.domain_constants)
            antecedents_disjunction.add_condition(lifted_antecedent)

        return antecedents_disjunction

    def initialize_dependencies(self, lifted_literals: Set[Predicate],
                                antecedents: Optional[Set[Predicate]] = None) -> None:
        """Initialize the dependencies with positive and negative literals.

        :param lifted_literals: the lifted bounded literals matching the action.
        :param antecedents: in case that the antecedents may contain different literals from the results
            (e.g. in the case of universal effects).
        """
        literals_str = {literal.untyped_representation for literal in lifted_literals}
        for literal in literals_str:
            if antecedents is not None:
                antecedents_literals = {antecedent.untyped_representation for antecedent in antecedents}
                self.possible_antecedents[literal] = create_antecedents_combination(
                    antecedents_literals, self.max_antecedents)

            else:
                self.possible_antecedents[literal] = create_antecedents_combination(literals_str, self.max_antecedents)

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str], include_supersets: bool = False) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        :param include_supersets: whether to include supersets of the dependencies to remove.
        """
        self.logger.info(f"Removing dependencies {literals_to_remove} for literal {literal}")
        dependencies_to_remove = create_antecedents_combination(literals_to_remove, self.max_antecedents)
        if include_supersets:
            dependencies_to_remove = self._extract_superset_dependencies(literal, literals_to_remove)

        for dependency in dependencies_to_remove:
            if dependency in self.possible_antecedents[literal]:
                self.possible_antecedents[literal].remove(dependency)

    def remove_preconditions_literals(self, preconditions_literals: Set[str]) -> None:
        """Removes the preconditions literals from the dependency set (from both the antecedents and the results).

        :param preconditions_literals: the preconditions of the action.
        """
        for literal in preconditions_literals:
            self.possible_antecedents.pop(literal, None)

        for literal in self.possible_antecedents:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

    def is_safe_literal(self, literal: str, preconditions_literals: Optional[Set[str]] = None) -> bool:
        """Determines whether the literal is safe in terms of number of antecedents.

        :param literal: the literal to check.
        :param preconditions_literals: the preconditions of the action.
        :return: True if the dependency set is safe, False otherwise.
        """
        if preconditions_literals is not None:
            self.remove_dependencies(literal, preconditions_literals, include_supersets=True)

        return len(self.possible_antecedents[literal]) <= 1

    def is_safe_conditional_effect(self, literal: str) -> bool:
        """Determines whether the literal is a conditional effect with safe number of antecedents.

        :param literal: the literal to check.
        :return: True if the dependency set is safe, False otherwise.
        """
        self.logger.info(f"Determining whether the literal {literal} is a conditional effect "
                         f"with safe number of antecedents")
        return len(self.possible_antecedents[literal]) == 1 and self.possible_antecedents[literal][0] != {literal}

    def is_possible_result(self, literal: str) -> bool:
        """Determines whether the literal is a possible result.

        :param literal: the literal to check.
        :return: True if the literal is a key in the dependency set, False otherwise.
        """
        return literal in self.possible_antecedents

    def extract_safe_antecedents(self, literal: str) -> Set[str]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the safe antecedents for the literals which is the result of the conditional effect.
        """
        self.logger.info("Extracting the tuple of the safe antecedents for the literal %s", literal)
        antecedents_copy = self.possible_antecedents[literal].copy()
        if len(antecedents_copy) == 0:
            return set()

        safe_antecedents = antecedents_copy.pop()
        return safe_antecedents

    def construct_restrictive_preconditions(
            self, preconditions: Set[str], literal: str, is_effect: bool = False) -> Optional[Precondition]:
        """Extracts the safe conditional effects from the dependency set.

        :param preconditions: the preconditions of the action.
        :param literal: the literal to create restrictive preconditions for.
        :param is_effect: whether the literal is an effect.
        :return: the negative and positive conditions that need to be added or None.
        """
        if literal in preconditions:
            return None

        unsafe_antecedents = self.possible_antecedents[literal]
        self.logger.debug("Constructing restrictive preconditions from the unsafe antecedents for literal %s", literal)
        negated_conditions_statement = []
        positive_antecedents = Precondition("and")
        negated_antecedents_conjunction = Precondition("and")
        for conjunction in unsafe_antecedents:
            negated_conditions_statement.append(self._negate_predicates(conjunction))
            negated_antecedents_conjunction.add_condition(self._create_negated_antecedents(conjunction))
            for antecedent_literal in conjunction:
                positive_antecedents.add_condition(extract_predicate_data(action_signature=self.action_signature,
                                                                          predicate_str=antecedent_literal,
                                                                          domain_constants=self.domain_constants))

        if all([isinstance(cond, str) for cond in negated_conditions_statement]) and \
                set(negated_conditions_statement) == preconditions:
            self.logger.debug(f"The negation of the antecedents for {literal} is the same as the preconditions"
                              f" so it can never be triggered!")
            return None

        negated_result = f"{NOT_PREFIX} {literal})"
        lifted_result_predicate = extract_predicate_data(action_signature=self.action_signature,
                                                         predicate_str=literal,
                                                         domain_constants=self.domain_constants)
        if is_effect:
            restrictive_precondition = Precondition("or")
            restrictive_precondition.add_condition(positive_antecedents)
            restrictive_precondition.add_condition(negated_antecedents_conjunction)
            if negated_result not in preconditions:
                restrictive_precondition.add_condition(lifted_result_predicate)

            return restrictive_precondition

        if negated_result in preconditions:
            return negated_antecedents_conjunction

        not_result_precondition = Precondition("or")
        not_result_precondition.add_condition(lifted_result_predicate)
        not_result_precondition.add_condition(negated_antecedents_conjunction)
        return not_result_precondition

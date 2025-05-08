"""Module representing the dependency set of an action."""
import logging
from typing import Set, Dict, List, Optional, Union

import math
from pddl_plus_parser.models import Predicate, Precondition, SignatureType, PDDLConstant

from sam_learning.core.propositional_operations.discrete_utilities import extract_predicate_data
from sam_learning.core.propositional_operations.logical_expression_operations import (
    create_cnf_combination,
    minimize_cnf_clauses,
    _flip_single_predicate,
    check_complementary_literals,
)

NOT_PREFIX = "(not"
AFTER_NOT_PREFIX_INDEX = 5
RIGHT_BRACKET_INDEX = -1


class DependencySet:
    """Class representing the dependency set of an action."""

    possible_antecedents: Dict[str, List[Set[str]]]
    max_antecedents: int
    action_signature: SignatureType
    domain_constants: Dict[str, PDDLConstant]
    logger: logging.Logger

    def __init__(self, max_size_antecedents: int, action_signature: SignatureType, domain_constants: Dict[str, PDDLConstant]):
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

    def _extract_negated_minimized_antecedents(self, unsafe_antecedents: List[Set[str]], preconditions: Set[str]) -> Union[List[Set[str]], bool]:
        """Tries to minimize the negated antecedents' size to a more compact representation.

        :param unsafe_antecedents: the unsafe antecedents to minimize.
        :param preconditions: the preconditions of the action to remove from the CNFs
        :return: the minimized CNFs.
        """
        negated_conditions_statement = []
        for conjunction in unsafe_antecedents:
            negated_conditions_statement.append({_flip_single_predicate(predicate) for predicate in conjunction})

        try:
            minimized_clauses = negated_conditions_statement
            previous_size = math.inf
            new_size = sum([len(clause) for clause in minimized_clauses])
            while previous_size > new_size:
                minimized_clauses = minimize_cnf_clauses(minimized_clauses, preconditions.copy())
                if len(minimized_clauses) == 0:
                    return True

                previous_size = new_size
                new_size = sum([len(clause) for clause in minimized_clauses])

            return minimized_clauses

        except ValueError:
            self.logger.warning("The CNF unit clauses are contradicting one another!")
            return False

    def _post_process_preconditions(self, precondition: Precondition) -> Union[Precondition, bool]:
        """Post processes the preconditions to remove unnecessary conditions.

        :param precondition: the precondition to post process.
        :return: the post processed precondition.
        """
        self.logger.debug(f"Post processing precondition {precondition}")
        if precondition.binary_operator == "or":
            same_level_predicates = [operand for operand in precondition.operands if isinstance(operand, Predicate)]
            if check_complementary_literals({predicate.untyped_representation for predicate in same_level_predicates}):
                return True

        return precondition

    def _create_negated_antecedent_preconditions(self, negated_minimized_antecedents: Union[List[Set[str]], bool]) -> Union[Precondition, Predicate]:
        """Creates the preconditions for the negated antecedents.

        :param negated_minimized_antecedents: the literals set representing the negated antecedents.
        :return: the preconditions for the negated antecedents.
        """
        negated_antecedents_preconditions = Precondition("and")
        if len(negated_minimized_antecedents) == 1:
            disjunction = next(iter(negated_minimized_antecedents))
            if len(disjunction) == 1:
                return extract_predicate_data(
                    action_signature=self.action_signature, predicate_str=disjunction.pop(), domain_constants=self.domain_constants
                )

            or_condition = Precondition("or")
            for antecedent_literal in disjunction:
                or_condition.add_condition(
                    extract_predicate_data(
                        action_signature=self.action_signature, predicate_str=antecedent_literal, domain_constants=self.domain_constants
                    )
                )
            return or_condition

        for disjunction in negated_minimized_antecedents:
            if len(disjunction) == 1:
                negated_antecedents_preconditions.add_condition(
                    extract_predicate_data(
                        action_signature=self.action_signature, predicate_str=disjunction.pop(), domain_constants=self.domain_constants
                    )
                )
                continue

            or_condition = Precondition("or")
            for antecedent_literal in disjunction:
                or_condition.add_condition(
                    extract_predicate_data(
                        action_signature=self.action_signature, predicate_str=antecedent_literal, domain_constants=self.domain_constants
                    )
                )

            negated_antecedents_preconditions.add_condition(or_condition)

        return negated_antecedents_preconditions

    def _handle_contradicting_antecedents(
        self, is_effect: bool, literal: str, positive_minimized_antecedents: Set[str]
    ) -> Union[Predicate, Precondition]:
        self.logger.debug(f"The negation of the antecedents for {literal} contains contradiction, " f"so the rest of the DNF must be true")
        if not is_effect:
            lifted_result_predicate = extract_predicate_data(
                action_signature=self.action_signature, predicate_str=literal, domain_constants=self.domain_constants
            )
            return lifted_result_predicate
        # the literal was observed as an effect of the action, so either the positive clauses are
        # true or the literal itself is true
        if check_complementary_literals(positive_minimized_antecedents):
            lifted_result_predicate = extract_predicate_data(
                action_signature=self.action_signature, predicate_str=literal, domain_constants=self.domain_constants
            )
            return lifted_result_predicate
        positive_antecedents = Precondition("and")
        for antecedent_literal in positive_minimized_antecedents:
            positive_antecedents.add_condition(
                extract_predicate_data(
                    action_signature=self.action_signature, predicate_str=antecedent_literal, domain_constants=self.domain_constants
                )
            )
        resulting_preconditions = Precondition("or")
        lifted_result_predicate = extract_predicate_data(
            action_signature=self.action_signature, predicate_str=literal, domain_constants=self.domain_constants
        )
        resulting_preconditions.add_condition(lifted_result_predicate)
        resulting_preconditions.add_condition(positive_antecedents)
        return resulting_preconditions

    def _handle_effect_literal(
        self,
        lifted_result_predicate: Predicate,
        negated_antecedents_preconditions: Union[Precondition, Predicate],
        negated_result: str,
        positive_minimized_antecedents: Set[str],
        preconditions: Set[str],
    ) -> Optional[Precondition]:
        """Handles the creation of the conditional's effect preconditions.

        :param lifted_result_predicate: the lifted result predicate.
        :param negated_antecedents_preconditions: the negated antecedents preconditions.
        :param negated_result: the negated result.
        :param positive_minimized_antecedents: the positive minimized antecedents.
        :param preconditions: the preconditions.
        :return: the conditional's effect preconditions or None if the antecedents return a tautology.
        """
        restrictive_precondition = Precondition("or")
        positive_antecedents = Precondition("and")
        for antecedent_literal in positive_minimized_antecedents:
            positive_antecedents.add_condition(
                extract_predicate_data(
                    action_signature=self.action_signature, predicate_str=antecedent_literal, domain_constants=self.domain_constants
                )
            )
        restrictive_precondition.add_condition(positive_antecedents)
        restrictive_precondition.add_condition(negated_antecedents_preconditions)
        if negated_result not in preconditions:
            restrictive_precondition.add_condition(lifted_result_predicate)

        final_precondition = self._post_process_preconditions(restrictive_precondition)
        if final_precondition is True:
            return None

        return restrictive_precondition

    def initialize_dependencies(self, lifted_literals: Set[Predicate], antecedents: Optional[Set[Predicate]] = None) -> None:
        """Initialize the dependencies with positive and negative literals.

        :param lifted_literals: the lifted bounded literals matching the action.
        :param antecedents: in case that the antecedents may contain different literals from the results
            (e.g. in the case of universal effects).
        """

        literals_str = {literal.untyped_representation for literal in lifted_literals}
        for literal in literals_str:
            antecedents_literals = {antecedent.untyped_representation for antecedent in antecedents} if antecedents is not None else literals_str
            self.possible_antecedents[literal] = create_cnf_combination(antecedents_literals, self.max_antecedents)

    def remove_dependencies(self, literal: str, literals_to_remove: Set[str], include_supersets: bool = False) -> None:
        """Remove a dependency from the dependency set.

        :param literal: the literal that is dependent on the dependency.
        :param literals_to_remove: the literals that cannot be trigger candidates for the literal.
        :param include_supersets: whether to include supersets of the dependencies to remove.
        """
        self.logger.debug(f"Removing dependencies {literals_to_remove} for literal {literal}")
        if literal not in self.possible_antecedents:
            self.logger.warning(f"The literal {literal} is not in the dependency set!")
            return

        conjunctions_to_remove = create_cnf_combination(literals_to_remove, self.max_antecedents)
        if include_supersets:
            conjunctions_to_remove = self._extract_superset_dependencies(literal, literals_to_remove)

        for dependency in conjunctions_to_remove:
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
        self.logger.info(f"Determining whether the literal {literal} is a conditional effect " f"with safe number of antecedents")
        return len(self.possible_antecedents[literal]) == 1

    def is_possible_result(self, literal: str) -> bool:
        """Determines whether the literal is a possible result.

        :param literal: the literal to check.
        :return: True if the literal is a key in the dependency set, False otherwise.
        """
        return literal in self.possible_antecedents

    def extract_safe_antecedents(self, literal: str) -> Union[Set[str], List[List[str]]]:
        """Extracts the safe conditional effects from the dependency set.

        :return: the safe antecedents for the literals which is the result of the conditional effect.
        """
        self.logger.info("Extracting the tuple of the safe antecedents for the literal %s", literal)
        antecedents_copy = self.possible_antecedents[literal].copy()
        if len(antecedents_copy) > 1:
            raise ValueError("The literal has more than one antecedent so it is unsafe!")

        if len(antecedents_copy) == 0:
            return set()

        safe_antecedents = antecedents_copy.pop()
        return safe_antecedents

    def construct_restrictive_preconditions(
        self, preconditions: Set[str], literal: str, is_effect: bool = False
    ) -> Optional[Union[Precondition, Predicate]]:
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
        negated_minimized_antecedents = self._extract_negated_minimized_antecedents(unsafe_antecedents, preconditions)
        positive_minimized_antecedents = set()
        for disjunction in unsafe_antecedents:
            positive_minimized_antecedents.update(disjunction)

        if isinstance(negated_minimized_antecedents, bool) and negated_minimized_antecedents:
            self.logger.debug(f"The negation of the antecedents for {literal} is the same as the preconditions " f"so it can never be triggered!")
            return None

        if isinstance(negated_minimized_antecedents, bool) and not negated_minimized_antecedents:
            return self._handle_contradicting_antecedents(is_effect, literal, positive_minimized_antecedents)

        # the negated antecedents are the minimized set of literals with no contradictions
        negated_antecedents_preconditions = self._create_negated_antecedent_preconditions(negated_minimized_antecedents)

        negated_result = f"{NOT_PREFIX} {literal})" if not literal.startswith(NOT_PREFIX) else literal[AFTER_NOT_PREFIX_INDEX:RIGHT_BRACKET_INDEX]
        lifted_result_predicate = extract_predicate_data(
            action_signature=self.action_signature, predicate_str=literal, domain_constants=self.domain_constants
        )

        if is_effect:
            return self._handle_effect_literal(
                lifted_result_predicate, negated_antecedents_preconditions, negated_result, positive_minimized_antecedents, preconditions
            )

        if negated_result in preconditions:
            return negated_antecedents_preconditions

        not_result_precondition = Precondition("or")
        not_result_precondition.add_condition(lifted_result_predicate)
        not_result_precondition.add_condition(negated_antecedents_preconditions)
        final_result = self._post_process_preconditions(not_result_precondition)
        if final_result is True:
            return None

        return final_result

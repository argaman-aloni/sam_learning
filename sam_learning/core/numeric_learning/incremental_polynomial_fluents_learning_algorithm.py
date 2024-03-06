"""Module that learns polynomial preconditions and effects from a domain."""
import itertools
from typing import Dict, List, Optional

import numpy
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Precondition

from sam_learning.core.numeric_learning import IncrementalNumericFluentStateStorage, IncrementalConvexHullLearner


class IncrementalPolynomialFluentsLearningAlgorithm(IncrementalNumericFluentStateStorage):
    """Class that learns polynomial preconditions and effects from a domain.

    Note:
        If the polynom degree is 0 the algorithm reverts to its linear version.
        degree of 1 is the multiplication of each couple of state fluents.
        degree 2 and above is the maximal degree of the polynomial.
    """

    polynom_degree: int

    def __init__(self, action_name: str, polynom_degree: int, domain_functions: Dict[str, PDDLFunction]):
        super().__init__(action_name, domain_functions)
        self.monomials = []
        self.polynom_degree = polynom_degree

    def _create_polynomial_string_recursive(self, fluents: List[str]) -> str:
        """Creates the polynomial string representing the equation recursively.

        :param fluents: the numeric fluents to create the polynomial string from.
        :return: the polynomial string representing the equation.
        """
        if len(fluents) == 1:
            return fluents[0]

        return f"(* {fluents[0]} {self._create_polynomial_string_recursive(fluents[1:])})"

    def _create_polynomial_string(self, fluents: List[str]) -> str:
        """The auxiliary function that creates the polynomial string representing the equation.

        :param fluents: the numeric fluents to create the polynomial string from.
        :return: the polynomial string representing the equation.
        """
        return self._create_polynomial_string_recursive(fluents)

    def _create_monomials(self, domain_functions: List[str]) -> None:
        """Creates the monomials from the state fluents.

        :return: the monomials from the state fluents.
        """
        self.monomials = list([item] for item in domain_functions)
        if self.polynom_degree == 0:
            return

        if self.polynom_degree == 1:
            for first_fluent, second_fluent in itertools.combinations(domain_functions, r=2):
                monomial = [first_fluent, second_fluent]
                self.monomials.append(monomial)

        else:
            for degree in range(2, self.polynom_degree + 1):
                for fluent_combination in itertools.combinations_with_replacement(domain_functions, r=degree):
                    monomial = list(fluent_combination)
                    self.monomials.append(monomial)

    def init_numeric_datasets(self) -> None:
        """Initializes the convex hull learner."""
        if self.polynom_degree == 0:
            return

        self._create_monomials(list([function.untyped_representation for function in self.domain_functions.values()]))
        monomial_strings = [self._create_polynomial_string(monomial) for monomial in self.monomials]
        self.convex_hull_learner.data = DataFrame(columns=monomial_strings)
        self.linear_regression_learner.previous_state_data = DataFrame(columns=monomial_strings)

    def add_to_previous_state_storage(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds the matched lifted state fluents to the previous state storage.

        :param state_fluents: the lifted state fluents that were matched for the action.
        """
        if self.polynom_degree == 0:
            super().add_to_previous_state_storage(state_fluents)
            return

        sample_dataset = {}
        for monomial in self.monomials:
            sample_dataset[self._create_polynomial_string(monomial)] = numpy.prod([state_fluents[fluent].value for fluent in monomial])

        self.convex_hull_learner.add_new_point(sample_dataset)
        self.linear_regression_learner.add_new_observation(sample_dataset, store_in_prev_state=True)

    def construct_safe_linear_inequalities(self, relevant_fluents: Optional[List[str]] = None) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        return self.convex_hull_learner.construct_convex_hull_inequalities()

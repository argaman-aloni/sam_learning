"""Module representing a data strudture that manages the matching of lifted predicates to their possible executing actions."""
from typing import List, Dict, NoReturn

from pddl_plus_parser.models import Predicate


class MultiActionPredicateMatching:
    """Class that manages the matching of lifted predicates to their possible executing actions."""

    positive_predicates: List[Dict[Predicate, List[Predicate]]]
    negative_predicates: List[Dict[Predicate, List[Predicate]]]

    def __init__(self,):
        """Initialize the class."""
        self.positive_predicates = []
        self.negative_predicates = []

    def add_positive_predicates(self, predicate: Predicate, bounded_lifted_predicates: List[Predicate]) -> NoReturn:
        """Add a positive predicates to the list of predicates.

        :param predicate: the domain predicate.
        :param bounded_lifted_predicates: the bounded action lifted predicates.
        """
        self.positive_predicates.append({predicate: bounded_lifted_predicates})

    def add_negative_predicate(self, predicate: Predicate, bounded_lifted_predicates: List[Predicate]) -> NoReturn:
        """Add a negative predicates to the list of predicates.

        :param predicate: the domain predicate.
        :param bounded_lifted_predicates: the bounded action lifted predicates.
        """
        self.negative_predicates.append({predicate: bounded_lifted_predicates})


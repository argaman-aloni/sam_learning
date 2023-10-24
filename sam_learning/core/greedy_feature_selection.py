"""A greedy algorithm for feature selection."""
import logging
from typing import List, Dict, Set

from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction


class GreedyFeatureSelector:
    """A greedy algorithm for feature selection."""

    logger: logging.Logger
    action_name: str
    open_list: List[Set[str]]
    closed_list: List[Set[str]]
    positive_samples: DataFrame
    negative_samples: DataFrame
    _function_vocabulary: List[str]

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.open_list = []
        self.closed_list = []
        self.positive_samples = None
        self.negative_samples = None
        self._function_vocabulary = None

    # def

    def init_search_data_structures(self, lifted_functions: List[str]) -> None:
        """

        :param lifted_functions:
        :return:
        """
        self.positive_samples = DataFrame(columns=lifted_functions)
        self.negative_samples = DataFrame(columns=lifted_functions)
        self._function_vocabulary = lifted_functions
        self.open_list.append(set())

    def add_positive_sample(self, positive_numeric_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param positive_numeric_sample: the numeric functions representing the positive sample.
        """
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           positive_numeric_sample.items()}
        self.positive_samples.loc[len(self.positive_samples)] = new_sample_data

    def add_negative_sample(self, numeric_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param numeric_negative_sample: the numeric functions representing the negative sample.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           numeric_negative_sample.items()}
        self.negative_samples.loc[len(self.negative_samples)] = new_sample_data

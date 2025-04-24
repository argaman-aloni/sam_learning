"""A greedy algorithm for feature selection."""
import itertools
import logging
from typing import List

import pandas as pd
from pandas import DataFrame, Series


class BFSFeatureSelector:
    """A greedy algorithm for feature selection.

    The algorithm works as a Breadth-First Search (BFS) algorithm, where each node in the search tree represents
    the current set of features that are being considered.
    The algorithm starts with an optimistic assumption, i.e., empty set of features, and at each iteration,
    it tries new sets of features. Each level of the tree represents the size of the features set examined.
    The tree increases until the leaves contain all the features of the action.
    """

    logger: logging.Logger
    action_name: str
    open_list: List[List[str]]
    _relevant_monomials: List[str]
    _observations: DataFrame

    def __init__(self, action_name: str, pb_monomials: List[str], pb_predicates: List[str]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.open_list = []
        self._relevant_monomials = pb_monomials
        self._pb_predicates = pb_predicates
        self._observations = None

    def _find_matching_successful_transition_row(self, reference_df: DataFrame, single_row_df: DataFrame, columns: list[str]) -> bool:
        """Searches for a row in the dataframe that matches the values for the specified columns.

        :param reference_df: The DataFrame to search in.
        :param single_row_df: The DataFrame containing the row to match.
        :param columns: List of columns to compare.
        :return: True if a successful match is found, False otherwise.
        """
        self.logger.debug("Searching for a matching successful transition row based on the input transition.")
        target_values = single_row_df.iloc[0][columns]
        matches = reference_df[reference_df[columns].eq(target_values).all(axis=1)]
        if matches.empty:
            return False

        for _, match in matches.iterrows():
            if match["is_successful"]:
                self.logger.debug("Found a successful match.")
                return True

        self.logger.debug("No successful match found.")
        return False

    def initialize_open_list(self) -> None:
        """Initialize the open list to contain all possible combinations of the relevant monomials.

        The open list is a list of lists, where each inner list contains a combination of monomials.
        Example: [['a'], ['b'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'c'], ['a', 'b', 'c']]
        """
        self.logger.debug("Initializing the open list.")
        combinations = [
            list(combo) for r in range(0, len(self._relevant_monomials) + 1) for combo in itertools.combinations(self._relevant_monomials, r)
        ]
        self.open_list = sorted(combinations, key=len)

    def add_new_observation(self, observation: DataFrame, is_successful: bool) -> List[str]:
        """Adds a new observation and determines whether to update the selected features.
        
        :param observation:  the observation to add.
        :param is_successful: whether the observation is successful or not.
        :return: the next set of features to use for the active learning.
        """
        if self._observations is None:
            self.logger.debug("Since this is the first observation, we assume optimistically that there are not numeric features.")
            self._observations = observation.copy()
            self._observations["is_successful"] = is_successful
            return self.open_list[0]

        self.logger.debug("Adding new observation to the search data structures.")
        new_labeled_observation = observation.copy()
        new_labeled_observation["is_successful"] = is_successful

        # check if the observation is already in the observations dataset
        if ((self._observations == new_labeled_observation).all(axis=1)).any():
            self.logger.debug("The observation already exists in the observations dataset.")
            return self.open_list[0]

        self._observations = pd.concat([self._observations, new_labeled_observation], ignore_index=True)
        if not is_successful:
            self.logger.debug("The transition is not successful, we need to check if the selected features need to be updated.")
            # search for the predicate part of the observation in the observations dataset and see if there exists a successful transition with the same predicates
            # if there is, we update the selected features, otherwise we return the same features.
            if self._find_matching_successful_transition_row(self._observations.iloc[:-1], new_labeled_observation, self._pb_predicates):
                self.open_list.pop(0)
                return self.open_list[0]

        return self.open_list[0]

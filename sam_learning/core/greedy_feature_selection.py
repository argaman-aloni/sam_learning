"""A greedy algorithm for feature selection."""
import logging
from typing import List, Dict, Set

from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction

from sam_learning.core.consistent_model_validator import NumericConsistencyValidator


class GreedyFeatureSelector(NumericConsistencyValidator):
    """A greedy algorithm for feature selection.

    The algorithm works as a Breadth-First Search (BFS) algorithm, where each node in the search tree represents
    the current set of features that are being considered.
    The algorithm starts with an optimistic assumption, i.e., empty set of features, and at each iteration,
    it tries new sets of features. Each level of the tree represents the size of the features set examined.
    The tree increases until the leaves contain all the features of the action.
    """

    logger: logging.Logger
    action_name: str
    open_list: List[Set[str]]
    closed_list: List[Set[str]]
    _function_vocabulary: List[str]

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        super().__init__(action_name)
        self.action_name = action_name
        self.open_list = []
        self.closed_list = []
        self._function_vocabulary = None

    # def

    def init_search_data_structures(self, lifted_functions: List[str]) -> None:
        """

        :param lifted_functions:
        :return:
        """
        super().init_numeric_dataframes(lifted_functions)
        self._function_vocabulary = lifted_functions
        self.open_list.append(set())

    def _expand_node(self) -> Set[str]:
        """The function executed the node expansion step of the algorithm.

        :return: the next set of features to examine.
        """
        self.logger.info("Expanding the new node in the features search tree.")
        if len(self.open_list) == 0:
            raise ValueError("The open list is empty. This should not happen.")

        next_set_of_features = self.open_list.pop(0)
        while next_set_of_features in self.closed_list:
            self.logger.debug(f"The node {next_set_of_features} was already expanded. Skipping.")
            next_set_of_features = self.open_list.pop(0)

        self.logger.debug("Adding the next level of features to the open list.")
        for function_name in self._function_vocabulary:
            if function_name in next_set_of_features:
                continue

            new_set_of_features = next_set_of_features.union({function_name})
            if new_set_of_features in self.closed_list or new_set_of_features in self.open_list:
                continue

            self.open_list.append(new_set_of_features)

        self.closed_list.append(next_set_of_features)
        self.logger.debug(f"Done expanding node {next_set_of_features}.")

        return next_set_of_features

    def apply_feature_selection(self) -> List[str]:
        """Applies feature selection and selects the next set of features to use for the active learning.

        Note:
            The algorithm expands the new node in the tree and validates that the features selected are consistent with
            the observation the algorithm currently has.
            If the features are consistent, the algorithm returns the new set of features.
            Otherwise, the algorithm expands the next node in the tree.
            We define consistency by validating that no negative sample is consistent with the model created by the set
            of features selected by the algorithm.

        :return: the next set of features to use for the active learning.
        """
        self.logger.info("Applying feature selection.")
        next_set_of_features = list(self._expand_node())
        self.logger.debug(f"Validating the consistency of the features {next_set_of_features}.")
        while self._in_hull(points_to_test=self.numeric_negative_samples[next_set_of_features],
                            hull_df=self.numeric_positive_samples[next_set_of_features],
                            use_cached_ch=True):
            self.logger.debug(f"The features {next_set_of_features} are consistent with the negative samples "
                              f"so cannot be used as features for the active learning.")
            next_set_of_features = list(self._expand_node())

        self.logger.info(f"The features {next_set_of_features} are not consistent with any negative sample.")
        return next_set_of_features

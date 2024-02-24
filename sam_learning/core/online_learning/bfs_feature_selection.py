"""A greedy algorithm for feature selection."""
import logging
from typing import List, Set

from sam_learning.core.online_learning.consistent_model_validator import NumericConsistencyValidator


class BFSFeatureSelector(NumericConsistencyValidator):
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

    def _expand_node(self) -> Set[str]:
        """The function executed the node expansion step of the algorithm.

        :return: the next set of features to examine.
        """
        self.logger.info("Expanding the new node in the features search tree.")
        if len(self.open_list) == 0:
            raise ValueError("The open list is empty. This should not happen.")

        selected_feature_set = self.open_list.pop(0)
        while selected_feature_set in self.closed_list:
            self.logger.debug(f"The node {selected_feature_set} was already expanded. Skipping.")
            selected_feature_set = self.open_list.pop(0)

        self.logger.debug("Adding the next level of features to the open list.")
        features_not_in_selected_feature_set = set(self._function_vocabulary).difference(selected_feature_set)
        for function_name in features_not_in_selected_feature_set:
            new_set_of_features = selected_feature_set.union({function_name})
            if new_set_of_features in self.closed_list or new_set_of_features in self.open_list:
                continue

            self.open_list.append(new_set_of_features)

        self.closed_list.append(selected_feature_set)
        self.logger.debug(f"Done expanding node {selected_feature_set}.")

        return selected_feature_set

    def init_search_data_structures(self, lifted_functions: List[str]) -> None:
        """Initializes the data structures used by the algorithm.

        :param lifted_functions: the names of the lifted numeric functions.
        """
        super().init_numeric_dataframes(lifted_functions)
        self._function_vocabulary = lifted_functions
        self.open_list.append(set())

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

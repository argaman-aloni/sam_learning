"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pddl_plus_parser.models import PDDLFunction, Predicate
from scipy.spatial import Delaunay, QhullError, delaunay_plot_2d


class InformationGainLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    lifted_functions: List[str]
    lifted_predicates: List[str]
    positive_samples_df: DataFrame
    negative_samples_df: DataFrame

    def __init__(self, action_name: str, lifted_functions: List[str], lifted_predicates: List[str]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.lifted_functions = lifted_functions
        self.lifted_predicates = lifted_predicates
        self.positive_samples_df = DataFrame(columns=lifted_functions + lifted_predicates)
        self.negative_samples_df = DataFrame(columns=lifted_functions + lifted_predicates)

    def _locate_sample_in_df(self, sample_to_locate: List[float], df: DataFrame) -> int:
        """Locates the sample in the data frame.

        :param sample_to_locate: the sample to locate in the data frame.
        :param df: the data frame to locate the sample in.
        :return: the index of the sample in the data frame.
        """
        for index, row in df.iterrows():
            if row.values.tolist() == sample_to_locate:
                self.logger.debug("Found the matching row.")
                return index

        return -1

    def _display_delaunay_graph(self, hull: Delaunay, num_dimensions: int) -> None:
        """Displays the convex hull in as a plot.

        :param hull: the convex hull to display.
        :param num_dimensions: the number of dimensions of the original data.
        """
        if num_dimensions == 2:
            _ = delaunay_plot_2d(hull)
            plt.title(f"{self.action_name} - delaunay graph")
            plt.show()

    def _in_hull(self, points_to_test: np.ndarray, hull: np.ndarray, debug_mode: bool = True) -> bool:
        """
        Test if the points are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.

        It returns true if any of the points lies inside the hull.

        :param points_to_test: the points to test whether they are inside the convex hull.
        :param hull: the points composing the positive samples convex hull.
        :param debug_mode: whether to display the convex hull.
        :return: whether any of the negative samples is inside the convex hull.
        """
        if hull.shape[1] == 1:
            return all([hull.min() <= point <= hull.max() for point in points_to_test])

        delaunay_hull = Delaunay(hull)
        if debug_mode:
            self._display_delaunay_graph(delaunay_hull, hull.shape[1])

        result = delaunay_hull.find_simplex(points_to_test) >= 0
        if isinstance(result, np.bool_):
            return result

        return any(result)

    def _remove_features(self, features_to_keep: List[str], features_list: List[str]) -> List[str]:
        """Removes features from the data frames.

        :param features_to_keep: the features to keep.
        :param features_list: the features list to remove the features from.
        :return:
        """
        columns_to_drop = [feature for feature in features_list if feature not in features_to_keep]
        self.positive_samples_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
        self.negative_samples_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
        return columns_to_drop

    def _remove_redundant_propositional_features(self, positive_propositional_sample: List[Predicate]) -> List[str]:
        """Removes features that are not needed for the calculation of the information gain.

        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        :return: the list of the propositional predicates that are needed for the calculation.
        """
        self.logger.info("Removing propositional features that are not needed for the calculation.")
        state_predicates_names = [predicate.untyped_representation for predicate in positive_propositional_sample]
        columns_to_drop = self._remove_features(state_predicates_names, self.lifted_predicates)
        for column in columns_to_drop:
            if column in self.lifted_predicates:
                self.lifted_predicates.remove(column)

        return self.lifted_predicates

    def _validate_negative_sample_in_state_predicates(
            self, negative_sample: Series, state_predicates_names: List[str]) -> bool:
        """Check whether the negative sample is relevant to the state predicates.

        :param negative_sample:  the negative sample to validate.
        :param state_predicates_names:  the state predicates names.
        :return:  whether the negative sample is relevant to the state predicates.
        """
        for predicate in self.lifted_predicates:
            if negative_sample[predicate] == 0.0 and predicate not in state_predicates_names or \
                    negative_sample[predicate] == 1.0 and predicate in state_predicates_names:
                continue

            return False

        return True

    def _is_non_informative_safe(self, new_numeric_sample: Dict[str, PDDLFunction],
                                 new_propositional_sample: List[Predicate]) -> bool:
        """Validate whether a new sample is non-informative according to the safe model.

        :param new_numeric_sample: the numeric functions representing the new sample.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: whether the new sample is non-informative according to the safe model.
        """
        state_predicates_names = [predicate.untyped_representation for predicate in new_propositional_sample]
        # checking if the state contains the predicates appearing in the preconditions.
        if len(self.lifted_predicates) > 0 and \
                not all([precondition_predicate in state_predicates_names for precondition_predicate
                         in self.lifted_predicates]):
            self.logger.debug("Not all of the discrete preconditions hold in the new sample. "
                              "It is not applicable according to the safe model")
            return False

        positive_points_data = self.positive_samples_df[self.lifted_functions]
        positive_points = positive_points_data.to_numpy()
        new_point_data = [new_numeric_sample[col].value for col in self.lifted_functions]
        new_point_array = np.array(new_point_data)
        try:
            return self._in_hull(new_point_array, positive_points)

        except (QhullError, ValueError):
            return self._locate_sample_in_df(new_point_data, positive_points_data) != -1

    def _is_non_informative_unsafe(self, new_numeric_sample: Dict[str, PDDLFunction],
                                   new_propositional_sample: List[Predicate]) -> bool:
        """Tests whether a new sample is non-informative according to the unsafe model.

        Note:
            To validate this, we apply two tests:
                1. We check whether none of the predicates required for the action to be applicable are in the new
                sample.
                2. We check whether the new sample combined with the positive samples creates a convex hull that
                contains a negative sample.


        :param new_numeric_sample: the numeric functions representing the new sample.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: whether the new sample is non-informative according to the unsafe model.
        """
        state_predicates_names = [predicate.untyped_representation for predicate in new_propositional_sample]
        if len(self.lifted_predicates) > 0 and \
                not any([precondition_predicate in state_predicates_names for precondition_predicate
                         in self.lifted_predicates]):
            self.logger.debug("None of the existing preconditions hold in the new sample. "
                              "It is not informative since it will never be applicable.")
            return True

        self.logger.debug("Creating a new model from the new sample and validating if the new model "
                          "contains a negative sample.")
        new_model_data = self.positive_samples_df[self.lifted_functions].copy()
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           new_numeric_sample.items()}
        new_model_data.loc[len(new_model_data)] = new_sample_data

        for _, negative_sample in self.negative_samples_df.iterrows():
            if not self._validate_negative_sample_in_state_predicates(negative_sample, state_predicates_names):
                continue

            try:
                if self._in_hull(negative_sample[self.lifted_functions].to_numpy(),
                                 new_model_data.to_numpy()):
                    self.logger.debug("The new sample is not informative since it contains a negative sample.")
                    return True

            except (QhullError, ValueError):
                if self._locate_sample_in_df(negative_sample.values.tolist(), new_model_data) != -1:
                    return True

        return False

    def remove_non_existing_functions(self, functions_to_keep: List[str]) -> None:
        """Removes functions that do not exist in the state and thus irrelevant to the action.

        :param functions_to_keep: the functions to keep.
        """
        self.logger.info("Removing functions that do not exist in the state.")
        columns_to_drop = self._remove_features(functions_to_keep, self.lifted_functions)
        self.logger.debug(f"Found the following columns to drop - {columns_to_drop}")
        for column in columns_to_drop:
            if column in self.lifted_functions:
                self.lifted_functions.remove(column)

    def add_positive_sample(self, positive_numeric_sample: Dict[str, PDDLFunction],
                            positive_propositional_sample: List[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param positive_numeric_sample: the numeric functions representing the positive sample.
        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        """
        filtered_predicates_names = self._remove_redundant_propositional_features(positive_propositional_sample)
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           positive_numeric_sample.items()}
        for predicate in filtered_predicates_names:
            new_sample_data[predicate] = 1.0

        self.positive_samples_df.loc[len(self.positive_samples_df)] = new_sample_data

    def add_negative_sample(self, numeric_negative_sample: Dict[str, PDDLFunction],
                            negative_propositional_sample: List[Predicate]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param numeric_negative_sample: the numeric functions representing the negative sample.
        :param negative_propositional_sample: the propositional predicates representing the negative sample.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           numeric_negative_sample.items()}
        relevant_predicates = [predicate.untyped_representation for predicate in negative_propositional_sample if
                               predicate.untyped_representation in self.lifted_predicates]
        for predicate in self.lifted_predicates:
            if predicate not in relevant_predicates:
                new_sample_data[predicate] = 0.0
            else:
                new_sample_data[predicate] = 1.0

        self.negative_samples_df.loc[len(self.negative_samples_df)] = new_sample_data

    def calculate_sample_information_gain(
            self, new_numeric_sample: Dict[str, PDDLFunction],
            new_propositional_sample: List[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_numeric_sample: the new sample to calculate the information gain for.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: the information gain of the new sample.
        """
        self.logger.info("Calculating the information gain of a new sample.")
        # this way we maintain the order of the columns in the data frame.
        if len(self.positive_samples_df) == 0 and len(self.negative_samples_df) == 0:
            return 1  # TODO: calculate the information gain.

        is_non_informative_safe = self._is_non_informative_safe(new_numeric_sample, new_propositional_sample)
        is_non_informative_unsafe = self._is_non_informative_unsafe(new_numeric_sample, new_propositional_sample)
        if is_non_informative_safe or is_non_informative_unsafe:
            return 0

        self.logger.debug("The point is informative, calculating the information gain.")
        return 1  # TODO: calculate the information gain.

"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List

import numpy as np
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction
from scipy.spatial import Delaunay, QhullError


class NumericInformationGainLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    domain_functions: Dict[str, PDDLFunction]
    positive_samples_df: DataFrame
    negative_samples_df: DataFrame

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.domain_functions = domain_functions
        self.positive_samples_df = DataFrame()
        self.negative_samples_df = DataFrame()

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

    @staticmethod
    def _in_hull(points_to_test: np.ndarray, hull: np.ndarray) -> bool:
        """
        Test if the points are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.

        It returns true if any of the points lies inside the hull.

        :param hull: the points composing the positive samples convex hull.
        :return: whether any of the negative samples is inside the convex hull.
        """
        hull = Delaunay(hull)
        result = hull.find_simplex(points_to_test) >= 0
        if isinstance(result, np.bool_):
            return result

        return any(result)

    def add_positive_sample(self, lifted_positive_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param lifted_positive_sample: the numeric functions representing the positive sample.
        """
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           lifted_positive_sample.items()}
        self.positive_samples_df.loc[len(self.positive_samples_df)] = new_sample_data

    def add_negative_sample(self, lifted_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param lifted_negative_sample: the numeric functions representing the negative sample.
        """
        self.logger.info(f"Adding a new (possibly) negative sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           lifted_negative_sample.items()}
        self.negative_samples_df.loc[len(self.negative_samples_df)] = new_sample_data

    def remove_false_negative_sample(self, lifted_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Removes a sample that was considered as negative after it was observed in an applicable state.

        :param lifted_negative_sample: the numeric functions representing the falsely negative sample.
        """
        self.logger.info(f"Removing a false negative sample for the action {self.action_name}.")
        sample_values = [lifted_negative_sample[fluent].value for fluent in self.negative_samples_df.columns]
        matching_row_index = self._locate_sample_in_df(sample_values, self.negative_samples_df)
        if matching_row_index == -1:
            return

        self.negative_samples_df.drop(index=matching_row_index, axis=0, inplace=True)

    def calculate_sample_information_gain(self, new_lifted_sample: Dict[str, PDDLFunction]) -> float:
        """Calculates the information gain of a new sample.

        :param new_lifted_sample: the new sample to calculate the information gain for.
        :return: the information gain of the new sample.
        """
        self.logger.info("Calculating the information gain of a new sample.")
        positive_points = self.positive_samples_df.to_numpy()
        negative_points = self.negative_samples_df.to_numpy()
        # this way we maintain the order of the columns in the data frame.
        new_point_data = [new_lifted_sample[col].value for col in self.positive_samples_df.columns]
        new_point_array = np.array(new_point_data)
        points_combined = np.vstack((positive_points, new_point_array))
        try:
            is_non_informative = self._in_hull(negative_points, points_combined) or \
                                 self._in_hull(new_point_array, positive_points)
            if is_non_informative:
                return 0

            self.logger.debug("The point is informative, calculating the information gain.")
            return 1  # TODO: calculate the information gain.

        except (QhullError, ValueError):
            self.logger.debug("could not compile a convex hull from the given points.")
            if self._locate_sample_in_df(new_point_data, self.negative_samples_df) != -1 or \
                    self._locate_sample_in_df(new_point_data, self.positive_samples_df) != -1:
                self.logger.debug("The point is not informative.")
                return 0

            return 1

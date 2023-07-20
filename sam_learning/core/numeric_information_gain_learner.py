"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List

import numpy as np
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction
from scipy.spatial import Delaunay, QhullError


class NumericInformationGainLearner:
    """Online information gain learning algorithm."""

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

    @staticmethod
    def _convert_df_to_dict_list(df: DataFrame) -> Dict[str, List[float]]:
        """Converts a data frame to a dictionary of lists.

        :param df: the data frame to convert.
        :return: the dictionary of lists.
        """
        return {col: df[col].tolist() for col in df.columns}

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
        return any(hull.find_simplex(points_to_test) >= 0)

    def add_positive_sample(self, lifted_positive_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param lifted_positive_sample: the numeric functions representing the positive sample.
        """
        df_data = self._convert_df_to_dict_list(self.positive_samples_df)
        for col, val in df_data.items():
            df_data[col].append(lifted_positive_sample[col].value)

        self.positive_samples_df = DataFrame(df_data)

    def add_negative_sample(self, lifted_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param lifted_negative_sample: the numeric functions representing the negative sample.
        """
        df_data = self._convert_df_to_dict_list(self.negative_samples_df)
        for col, val in df_data.items():
            df_data[col].append(lifted_negative_sample[col].value)

        self.negative_samples_df = DataFrame(df_data)

    def remove_negative_sample(self, lifted_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Removes a sample that was considered as negative after it was observed in an applicable state.

        :param lifted_negative_sample: the numeric functions representing the falsely negative sample.
        """
        false_negative_sample = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                                 lifted_negative_sample.items()}
        self.negative_samples_df = self.negative_samples_df.drop(
            self.negative_samples_df.query(' & '.join([f'{col} == {val}' for col, val in
                                                       false_negative_sample.items()])).index)

    def calculate_sample_information_gain(self, new_sample: Dict[str, PDDLFunction]) -> float:
        """Calculates the information gain of a new sample.

        :param new_sample: the new sample to calculate the information gain for.
        :return: the information gain of the new sample.
        """
        positive_points = self.positive_samples_df.to_numpy()
        negative_points = self.negative_samples_df.to_numpy()
        # this way we maintain the order of the columns in the data frame.
        new_point = np.array([new_sample[col].value for col in self.positive_samples_df.columns])
        points_combined = np.vstack((positive_points, new_point))
        try:
            is_non_informative = self._in_hull(negative_points, points_combined) or \
                                 self._in_hull(new_point, positive_points)
            if is_non_informative:
                return 0

            self.logger.debug("The point is informative, calculating the information gain.")
            return 1  # TODO: calculate the information gain.

        except (QhullError, ValueError):
            self.logger.debug("could not compile a convex hull from the given points.")
            if new_point in positive_points or new_point in negative_points:
                return 0

            return 1

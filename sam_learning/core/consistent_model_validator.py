"""Module to use to validate whether the model is consistent with the observations."""
import logging
from typing import Dict, Optional, List

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction
from scipy.spatial import Delaunay, delaunay_plot_2d

from sam_learning.core.numeric_utils import get_num_independent_equations, filter_constant_features, \
    detect_linear_dependent_features, extended_gram_schmidt, EPSILON


class NumericConsistencyValidator:
    numeric_positive_samples: DataFrame
    numeric_negative_samples: DataFrame
    _effects_learned_perfectly: bool
    _cached_convex_hull: Optional[Delaunay]
    _orthonormal_span: Optional[List[List[float]]]
    _projected_base: Optional[List[List[float]]]

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.numeric_positive_samples = None
        self.numeric_negative_samples = None
        self._effects_learned_perfectly = False
        self._cached_convex_hull = None
        self._orthonormal_span = None
        self._projected_base = None

    def _are_numeric_dataframes_empty(self) -> bool:
        """Determines whether the numeric data frames are empty.

        :return: whether the numeric data frames are empty.
        """
        return len(self.numeric_positive_samples) == 0 and len(self.numeric_negative_samples) == 0

    def _locate_sample_in_df(self, sample_to_locate: List[float], df: DataFrame) -> int:
        """Locates the sample in the data frame.

        :param sample_to_locate: the sample to locate in the data frame.
        :param df: the data frame to locate the sample in.
        :return: the index of the sample in the data frame.
        """
        for index, row in df.iterrows():
            if row.values.tolist() == sample_to_locate:
                self.logger.debug("Found the matching row.")
                return int(index)

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

    def _calculate_whether_in_delanauy_hull(
            self, convex_hull_points: np.ndarray,
            new_point: np.ndarray, debug_mode: bool = False, use_cached_ch: bool = False) -> bool:
        """Calculates whether the new point is inside the convex hull using the delanauy algorith.

        :param convex_hull_points: the points composing the convex hull.
        :param new_point: the new point to test whether it is inside the convex hull.
        :param debug_mode: whether to display the convex hull.
        :param use_cached_ch: whether to use the cached convex hull. This reduces runtime.
        :return: whether the new point is inside the convex hull.
        """
        delaunay_hull = self._cached_convex_hull if self._cached_convex_hull is not None and use_cached_ch \
            else Delaunay(convex_hull_points)
        if use_cached_ch:
            self._cached_convex_hull = delaunay_hull

        if debug_mode:
            self._display_delaunay_graph(delaunay_hull, convex_hull_points.shape[1])

        result = delaunay_hull.find_simplex(new_point) >= 0
        if isinstance(result, np.bool_):
            return result.astype(bool)

        return any(result)

    def _in_hull(
            self, points_to_test: DataFrame, hull_df: DataFrame, debug_mode: bool = False,
            use_cached_ch: bool = False) -> bool:
        """
        Test if the points are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.

        It returns true if any of the points lies inside the hull.

        :param points_to_test: the points to test whether they are inside the convex hull.
        :param hull_df: the points composing the positive samples convex hull.
        :param debug_mode: whether to display the convex hull.
        :param use_cached_ch: whether to use the cached convex hull. This reduces runtime.
        :return: whether any of the negative samples is inside the convex hull.
        """
        self.logger.debug("Validating whether the input samples are inside the convex hull.")
        if len(hull_df.columns.tolist()) == 0:
            # If there is nothing to explore the points are by default in the convex hull.
            return True

        if len(points_to_test) == 0:
            return False

        shifted_hull_points = hull_df.to_numpy() - hull_df.to_numpy()[0]
        diagonal_eye = [list(vector) for vector in np.eye(shifted_hull_points.shape[1])]

        if self._projected_base is not None and use_cached_ch:
            projection_basis = self._projected_base
            orthnormal_span = self._orthonormal_span

        else:
            projection_basis = extended_gram_schmidt(shifted_hull_points)
            orthnormal_span = extended_gram_schmidt(diagonal_eye, projection_basis)
            if use_cached_ch:
                self._projected_base = projection_basis
                self._orthonormal_span = orthnormal_span

        projected_ch_points = np.dot(shifted_hull_points, np.array(projection_basis).T)
        shifted_new_points = points_to_test.to_numpy() - hull_df.to_numpy()[0]
        projected_new_point = np.dot(shifted_new_points, np.array(projection_basis).T)
        if (len(orthnormal_span) > 0 and
                (np.absolute(np.dot(np.array(orthnormal_span), np.array(shifted_new_points).T)) > EPSILON).any()):
            self.logger.debug("The new points are not in the span of the input points.")
            return False

        if projected_ch_points.shape[1] == 1:
            return all([projected_ch_points.min() <= point <= projected_ch_points.max()
                        for point in projected_new_point])

        return self._calculate_whether_in_delanauy_hull(
            projected_ch_points, projected_new_point, debug_mode, use_cached_ch)

    def init_numeric_dataframes(self, valid_lifted_functions: List[str]) -> None:
        """Initializes the data frames used to calculate the information gain.

        :param valid_lifted_functions: the lifted functions matching the action (used to avoid adding redundant ones).
        """

        self.numeric_positive_samples = DataFrame(columns=valid_lifted_functions)
        self.numeric_negative_samples = DataFrame(columns=valid_lifted_functions)

    def can_determine_numeric_effects_effects_perfectly(self) -> bool:
        """Determines whether the effects of the action can be predicted perfectly.

        :return: whether the effects of the action can be predicted perfectly.
        """
        if self._effects_learned_perfectly:
            # This is to prevent redundant calculations when the effects are already learned perfectly.
            return True

        if len(self.numeric_positive_samples) <= 1:
            return False

        filtered_df, _, _ = filter_constant_features(self.numeric_positive_samples)
        regression_df, _, _ = detect_linear_dependent_features(filtered_df)
        num_dimensions = len(regression_df.columns.tolist()) + 1  # +1 for the bias.
        num_independent_rows = get_num_independent_equations(regression_df)
        if num_independent_rows >= num_dimensions:
            self._effects_learned_perfectly = True
            return True

        return False

    def add_positive_numeric_sample(self, positive_numeric_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param positive_numeric_sample: the numeric functions representing the positive sample.
        """
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           positive_numeric_sample.items()}
        self.numeric_positive_samples.loc[len(self.numeric_positive_samples)] = new_sample_data

    def add_negative_numeric_sample(self, numeric_negative_sample: Dict[str, PDDLFunction]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param numeric_negative_sample: the numeric functions representing the negative sample.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           numeric_negative_sample.items()}
        self.numeric_negative_samples.loc[len(self.numeric_negative_samples)] = new_sample_data

    def clear_convex_hull_cache(self) -> None:
        """Clears the cached convex hull."""
        self._cached_convex_hull = None

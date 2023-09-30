"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from pddl_plus_parser.models import PDDLFunction, Predicate
from scipy.spatial import Delaunay, QhullError, delaunay_plot_2d
from sklearn.decomposition import PCA

from sam_learning.core.numeric_utils import get_num_independent_equations, filter_constant_features, \
    detect_linear_dependent_features, extended_gram_schmidt


class InformationGainLearner:
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    lifted_functions: List[str]
    lifted_predicates: List[str]
    positive_samples_df: DataFrame
    negative_samples_df: DataFrame
    _effects_learned_perfectly: bool
    _cached_convex_hull: Optional[Delaunay]
    _pca_model: Optional[PCA]

    def __init__(self, action_name: str, lifted_functions: List[str], lifted_predicates: List[str]):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.lifted_functions = lifted_functions
        self.lifted_predicates = lifted_predicates
        self.positive_samples_df = DataFrame(columns=lifted_functions + lifted_predicates)
        self.negative_samples_df = DataFrame(columns=lifted_functions + lifted_predicates)
        self._effects_learned_perfectly = False
        self._cached_convex_hull = None
        self._pca_model = None

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

    def _can_determine_effects_perfectly(self) -> bool:
        """Determines whether the effects of the action can be predicted perfectly.

        :return: whether the effects of the action can be predicted perfectly.
        """
        if self._effects_learned_perfectly:
            # This is to prevent redundant calculations when the effects are already learned perfectly.
            return True

        if len(self.positive_samples_df) == 1:
            return False

        filtered_df, _, _ = filter_constant_features(self.positive_samples_df)
        regression_df, _, _ = detect_linear_dependent_features(filtered_df)
        num_dimensions = len(regression_df.columns.tolist()) + 1  # +1 for the bias.
        num_independent_rows = get_num_independent_equations(regression_df)
        if num_independent_rows >= num_dimensions:
            self._effects_learned_perfectly = True
            return True

        return False

    @staticmethod
    def _validate_consts_match(
            points_to_test: DataFrame, hull_df: DataFrame, constant_features: List[str]) -> bool:
        """Validates that the points to test contain the same constant features as the hull.

        :param points_to_test: the points to test whether they are inside the convex hull.
        :param hull_df: the points composing the positive samples convex hull.
        :param constant_features: the constant features to validate.
        :return: whether the points to test contain the same constant features as the hull.
        """
        for constant_feature in constant_features:
            if len(points_to_test[constant_feature].unique().tolist()) != 1 or \
                    points_to_test[constant_feature].unique().tolist()[0] != hull_df[constant_feature].iloc[0]:
                return False

        return True

    def _calculate_whether_in_delanauy_hull(
            self, convex_hull_points: np.ndarray,
            new_point: np.ndarray, debug_mode: bool = False, use_cached_ch: bool = False, ) -> bool:
        """Calculates whether the new point is inside the convex hull using the delanauy algorith.

        :param convex_hull_points: the points composing the convex hull.
        :param new_point: the new point to test whether it is inside the convex hull.
        :param debug_mode: whether to display the convex hull.
        :param use_cached_ch: whether to use the cached convex hull. This reduces runtime.
        :return: whether the new point is inside the convex hull.
        """
        if self._cached_convex_hull is not None and use_cached_ch:
            delaunay_hull = self._cached_convex_hull
            relevant_sample = new_point if not self._pca_model is None else self._pca_model.transform(new_point)

        else:
            delaunay_hull = Delaunay(convex_hull_points)
            self._pca_model = None
            relevant_sample = new_point
            if use_cached_ch:
                self._cached_convex_hull = delaunay_hull

        if debug_mode:
            self._display_delaunay_graph(delaunay_hull, convex_hull_points.shape[1])

        result = delaunay_hull.find_simplex(relevant_sample) >= 0
        if isinstance(result, np.bool_):
            return result

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
        shifted_new_points = points_to_test.to_numpy() - hull_df.to_numpy()[0]
        shifted_hull_points = hull_df.to_numpy() - hull_df.to_numpy()[0]
        projection_basis = extended_gram_schmidt(shifted_hull_points)
        projected_points = np.dot(shifted_hull_points, np.array(projection_basis).T)
        projected_new_point = np.dot(shifted_new_points, np.array(projection_basis).T)
        diagonal_eye = [list(vector) for vector in np.eye(shifted_hull_points.shape[1])]
        orthnormal_span = extended_gram_schmidt(diagonal_eye, projection_basis)
        if len(orthnormal_span) > 0 and np.dot(np.array(projected_new_point), np.array(orthnormal_span)).any() != 0:
            self.logger.debug("The new points are not in the span of the input points.")
            return False

        if projected_points.shape[1] == 1:
            return all([projected_points.min() <= point <= projected_points.max() for point in projected_new_point])

        return self._calculate_whether_in_delanauy_hull(
            projected_points, projected_new_point, debug_mode, use_cached_ch)

    def _remove_features(self, features_to_keep: List[str], features_list: List[str]) -> List[str]:
        """Removes features from the data frames.

        :param features_to_keep: the features to keep.
        :param features_list: the features list to remove the features from.
        :return: the list of the removed features.
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

    def _validate_action_discrete_preconditions_hold_in_state(self, new_propositional_sample: List[Predicate]) -> bool:
        """Validate whether the bounded lifted predicates in the preconditions hold in the new sample.

        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: whether the bounded lifted predicates in the preconditions hold in the new sample.
        """
        state_bounded_lifted_predicates = {predicate.untyped_representation for predicate in new_propositional_sample}
        if len(self.lifted_predicates) > 0 and \
                not set(state_bounded_lifted_predicates).issuperset(self.lifted_predicates):
            self.logger.debug("Not all of the discrete preconditions hold in the new sample. "
                              "It is not applicable according to the safe model")
            return False

        return True

    def _is_non_informative_safe(
            self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: List[Predicate],
            relevant_numeric_features: Optional[List[str]] = None, use_cache: bool = False) -> bool:
        """Validate whether a new sample is non-informative according to the safe model.

        Note:
            To validate if the new sample is non-informative, we first check if all the discrete preconditions hold
            if so we continue to validate whether the new sample is inside the convex hull of the positive samples.
            If both of the conditions hold, the new sample is non-informative.

        :param new_numeric_sample: the numeric functions representing the new sample.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :param relevant_numeric_features:
        :param use_cache: whether to use the cached convex hull. This prevents redundant calculations and reduces
        :param relevant_numeric_features: the relevant numeric features to calculate the information gain for. If None,
            all the numeric features are used. This is used to reduce the runtime and reduce the dimensionality of
            the calculations.
            runtime.
        :return: whether the new sample is non-informative according to the safe model.
        """
        self.logger.info("Validating whether the new sample is non-informative according to the safe model.")
        if not self._validate_action_discrete_preconditions_hold_in_state(new_propositional_sample):
            return False

        functions_to_explore = self.lifted_functions if relevant_numeric_features is None else relevant_numeric_features
        if len(functions_to_explore) == 0:
            return True

        positive_points_data = self.positive_samples_df[functions_to_explore]
        new_point_data = DataFrame({col: [new_numeric_sample[col].value] for col in functions_to_explore})
        try:
            return self._in_hull(new_point_data,
                                 self.positive_samples_df[functions_to_explore], use_cached_ch=use_cache)

        except (QhullError, ValueError):
            sample_values = [new_numeric_sample[col].value for col in functions_to_explore]
            return self._locate_sample_in_df(sample_values, positive_points_data) != -1

    def _is_non_informative_unsafe(
            self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: List[Predicate],
            relevant_numeric_features: Optional[List[str]] = None) -> bool:
        """Tests whether a new sample is non-informative according to the unsafe model.

        Note:
            To validate this, we apply two tests:
                1. We check whether none of the predicates required for the action to be applicable are in the new
                sample.
                2. We check whether the new sample combined with the positive samples creates a convex hull that
                contains a negative sample.


        :param new_numeric_sample: the numeric functions representing the new sample.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :param relevant_numeric_features: the relevant numeric features to calculate the information gain for. If None,
            all the numeric features are used. This is used to reduce the runtime and reduce the dimensionality of
            the calculations.
        :return: whether the new sample is non-informative according to the unsafe model.
        """
        new_sample_predicates = {predicate.untyped_representation for predicate in new_propositional_sample}
        if len(self.lifted_predicates) > 0 and len(new_sample_predicates.intersection(self.lifted_predicates)) == 0:
            self.logger.debug("None of the existing preconditions hold in the new sample. "
                              "It is not informative since it will never be applicable.")
            return True

        self.logger.debug("Creating a new model from the new sample and validating if the new model "
                          "contains a negative sample.")
        functions_to_explore = self.lifted_functions if relevant_numeric_features is None else relevant_numeric_features
        if len(functions_to_explore) == 0:
            return True

        new_model_data = self.positive_samples_df[functions_to_explore].copy()
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           new_numeric_sample.items()}
        new_model_data.loc[len(new_model_data)] = new_sample_data

        for _, negative_sample in self.negative_samples_df.iterrows():
            if not self._validate_negative_sample_in_state_predicates(negative_sample, list(new_sample_predicates)):
                continue

            try:
                if self._in_hull(negative_sample[functions_to_explore].to_frame().T, new_model_data):
                    self.logger.debug("The new sample is not informative since it contains a negative sample.")
                    return True

            except (QhullError, ValueError):
                if self._locate_sample_in_df(list(new_sample_data.values()),
                                             negative_sample[functions_to_explore].to_frame().T) != -1:
                    return True

        return False

    def clear_convex_hull_cache(self) -> None:
        """Clears the cached convex hull."""
        self._cached_convex_hull = None

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

    def is_applicable_and_new_state(
            self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: List[Predicate]) -> bool:
        """Checks whether the new sample corresponds with the action's precondition and whether it is a new state.

        :param new_numeric_sample: the numeric functions representing the new sample.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: whether the state can be used to apply the action and whether this state has already been observed.
        """
        self.logger.info("Validating whether the new sample is applicable and a new state.")
        if not self._validate_action_discrete_preconditions_hold_in_state(new_propositional_sample):
            return False

        positive_points_data = self.positive_samples_df[self.lifted_functions]
        new_point_data = DataFrame({col: [new_numeric_sample[col].value] for col in self.lifted_functions})
        sample_values = [new_numeric_sample[col].value for col in self.lifted_functions]
        try:
            return self._in_hull(new_point_data, self.positive_samples_df[self.lifted_functions]) and \
                self._locate_sample_in_df(sample_values, positive_points_data) == -1

        except (QhullError, ValueError):
            return False

    def is_sample_informative(
            self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: List[Predicate],
            use_cache: bool = False, relevant_numeric_features: Optional[List[str]] = None) -> bool:
        """Checks whether the sample is informative.

        :param new_numeric_sample: the new sample to calculate whether it is informative.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :param use_cache: whether to use the cached convex hull. This prevents redundant calculations and reduces
            runtime.
        :param relevant_numeric_features: the relevant numeric features to calculate the information gain for. If None,
            all the numeric features are used. This is used to reduce the runtime and reduce the dimensionality of
            the calculations.
        :return: whether the sample is informative.
        """
        self.logger.info("Calculating the information gain of a new sample.")
        # this way we maintain the order of the columns in the data frame.
        if len(self.positive_samples_df) == 0 and len(self.negative_samples_df) == 0:
            self.logger.debug("There are no samples to calculate the information gain from - action not observed yet.")
            return True

        is_non_informative_safe = self._is_non_informative_safe(
            new_numeric_sample, new_propositional_sample, use_cache=use_cache,
            relevant_numeric_features=relevant_numeric_features)
        is_non_informative_unsafe = self._is_non_informative_unsafe(
            new_numeric_sample, new_propositional_sample, relevant_numeric_features=relevant_numeric_features)
        if is_non_informative_safe or is_non_informative_unsafe:
            return False

        self.logger.debug("The point is informative, calculating the information gain.")
        return True

    def calculate_information_gain(
            self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: List[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_numeric_sample: the numeric part of the sample to calculate the information gain for.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return:
        """
        return 1  # TODO: calculate the information gain.

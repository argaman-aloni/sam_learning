"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List, Set

import numpy as np
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Predicate
from scipy.spatial import Delaunay, QhullError


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

    def _remove_redundant_features(self, positive_propositional_sample: Set[Predicate]) -> List[str]:
        """Removes features that are not needed for the calculation of the information gain.

        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        :return: the list of the propositional predicates that are needed for the calculation.
        """
        self.logger.info("Removing propositional features that are not needed for the calculation.")
        state_predicates_names = [predicate.untyped_representation for predicate in positive_propositional_sample]
        columns_to_drop = [predicate for predicate in self.lifted_predicates if predicate not in state_predicates_names]
        self.positive_samples_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
        self.negative_samples_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
        for column in columns_to_drop:
            if column in self.lifted_predicates:
                self.lifted_predicates.remove(column)

        return self.lifted_predicates

    def add_positive_sample(self, positive_numeric_sample: Dict[str, PDDLFunction],
                            positive_propositional_sample: Set[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param positive_numeric_sample: the numeric functions representing the positive sample.
        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        """
        filtered_predicates_names = self._remove_redundant_features(positive_propositional_sample)

        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           positive_numeric_sample.items()}
        for predicate in filtered_predicates_names:
            new_sample_data[predicate] = 1.0

        self.positive_samples_df.loc[len(self.positive_samples_df)] = new_sample_data

    def add_negative_sample(self, lifted_negative_sample: Dict[str, PDDLFunction],
                            negative_propositional_sample: Set[Predicate]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param lifted_negative_sample: the numeric functions representing the negative sample.
        :param negative_propositional_sample: the propositional predicates representing the negative sample.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           lifted_negative_sample.items()}
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
            new_propositional_sample: Set[Predicate]) -> float:
        """Calculates the information gain of a new sample.

        :param new_numeric_sample: the new sample to calculate the information gain for.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: the information gain of the new sample.
        """
        self.logger.info("Calculating the information gain of a new sample.")
        positive_points = self.positive_samples_df.to_numpy()
        negative_points = self.negative_samples_df.to_numpy()
        # this way we maintain the order of the columns in the data frame.
        new_point_data = [new_numeric_sample[col].value for col in self.lifted_functions]
        lifted_predicate_names = [predicate.untyped_representation for predicate in new_propositional_sample]
        for predicate in self.lifted_predicates:
            if predicate in lifted_predicate_names:
                new_point_data.append(1.0)
            else:
                new_point_data.append(0.0)

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

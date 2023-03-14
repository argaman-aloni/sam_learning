"""This module contains the ConvexHullLearner class."""
import logging
import os
from pathlib import Path

from typing import Dict, List, Tuple, Optional

import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import EquationSolutionType, ConditionType
from sam_learning.core.numeric_utils import construct_multiplication_strings, construct_linear_equation_string, \
    detect_linear_dependent_features, prettify_coefficients


class ConvexHullLearner:

    def __init__(self, action_name: str):
        self.logger = logging.getLogger(__name__)
        self.previous_state_df = None
        self.convex_hull_error_file_path = Path(os.environ["CONVEX_HULL_ERROR_PATH"])
        self.action_name = action_name

    def _construct_pddl_inequality_scheme(
            self, coefficient_matrix: np.ndarray, border_points: np.ndarray, headers: List[str]) -> List[str]:
        """Construct the inequality strings in the appropriate PDDL format.

        :param coefficient_matrix: the matrix containing the coefficient vectors for each inequality.
        :param border_points: the convex hull point which ensures that Ax <= b.
        :param relevant_fluents: the fluents relevant to the creation of the preconditions if exists, if not,
            should be ALL the previous state variables.
        :return: the inequalities PDDL formatted strings.
        """
        inequalities = set()
        for inequality_coefficients, border_point in zip(coefficient_matrix, border_points):
            multiplication_functions = construct_multiplication_strings(inequality_coefficients, headers)
            constructed_left_side = construct_linear_equation_string(multiplication_functions)
            inequalities.add(f"(<= {constructed_left_side} {border_point})")

        return list(inequalities)

    def _create_disjunctive_preconditions(
            self, previous_state_matrix: DataFrame, equality_conditions=None) -> Tuple[List[str], ConditionType]:
        """Create the disjunctive representation of the preconditions.

        :param previous_state_matrix: the matrix containing the previous state values.
        :return: a disjunctive representation for the precondition in case a convex hull cannot be created.
        """
        if equality_conditions is None:
            equality_conditions = []

        injunctive_conditions = []
        functions_equality_strings = []
        for _, row in previous_state_matrix.iterrows():
            for column in previous_state_matrix.columns:
                functions_equality_strings.append(f"(= {column} {row[column]})")

            concatenated_str = " ".join(functions_equality_strings)
            injunctive_conditions.append(f"(and {concatenated_str} {' '.join(equality_conditions)})")

        if previous_state_matrix.shape[0] == 1:
            return [concatenated_str], ConditionType.conjunctive

        return injunctive_conditions, ConditionType.disjunctive

    def _create_convex_hull_linear_inequalities(self, points_df: DataFrame,
                                                display_mode: bool = False) -> Tuple[List[List[float]], List[float]]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :param points: the points that represent the values of the function in the states of the observations.
        :return: the matrix representing the inequalities of the planes created by the convex hull.

        Note: the returned values represents the linear inequalities of the convex hull, i.e.,  Ax <= b.
        """
        points = points_df.to_numpy()
        try:
            hull = ConvexHull(points)
            num_dimensions = points.shape[1]
            self._display_convex_hull(display_mode, hull, num_dimensions)
            A = hull.equations[:, :num_dimensions]
            b = -hull.equations[:, num_dimensions]
            return [prettify_coefficients(row) for row in A], prettify_coefficients(b)

        except (QhullError, ValueError) as e:
            with open(self.convex_hull_error_file_path, "at") as error_file:
                error_file.write(f"{e}\n")

            failure_reason = "Convex hull encountered an error condition and no solution was found"
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason, EquationSolutionType.convex_hull_not_found)

    def _display_convex_hull(self, display_mode: bool, hull: ConvexHull, num_dimensions: int) -> None:
        """Displays the convex hull in as a plot.

        :param display_mode: whether to display the plot.
        :param hull: the convex hull to display.
        :param num_dimensions: the number of dimensions of the original data.
        """
        if num_dimensions == 2 and display_mode:
            _ = convex_hull_plot_2d(hull)
            plt.title(f"{self.action_name} - convex hull")
            plt.show()

    def _construct_single_dimension_inequalities(self, single_column_df: Series, equality_strs: List[str] = []) -> \
            Tuple[List[str], ConditionType]:
        """Construct a single dimension precondition representation.

        :param relevant_fluent: the fluent only fluent that is relevant to the preconditions' creation.
        :param equality_strs: the equality conditions that are already present in the preconditions.
        :return: the preconditions string and the condition type.
        """
        min_value = single_column_df.min()
        max_value = single_column_df.max()
        relevant_fluent = single_column_df.name
        if min_value == max_value:
            conditions = [f"(= {relevant_fluent} {min_value})"]
        else:
            conditions = [f"(>= {relevant_fluent} {min_value})", f"(<= {relevant_fluent} {max_value})"]

        conditions.extend(equality_strs)
        return conditions, ConditionType.conjunctive

    def _filter_all_convex_hull_inconsistencies(self, previous_state_matrix: DataFrame) -> Tuple[List[str], DataFrame]:
        """Filters out features that might prevent from the convex hull algorithm to work properly.

        :param previous_state_matrix: the matrix of the previous state containing numeric values.
        :return: the equality strings, the filtered matrix and the remaining fluents that will be used to create
         the convex hull.
        """
        self.logger.debug(f"Now checking to see if there are columns that linearly dependent.")
        if previous_state_matrix.shape[0] == 1:
            self.logger.debug("There is only one sample of information to learn from, "
                              "not enough to learn of linear dependency!")
            return [], previous_state_matrix

        filtered_matrix, added_conditions, _ = detect_linear_dependent_features(previous_state_matrix)
        return added_conditions, filtered_matrix

    def construct_safe_linear_inequalities(
            self, state_storge: Dict[str, List[float]],
            relevant_fluents: Optional[List[str]] = None) -> Tuple[List[str], ConditionType]:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        state_data = DataFrame(state_storge).drop_duplicates().drop(
            columns=[fluent for fluent in state_storge.keys() if fluent not in relevant_fluents])
        if relevant_fluents is not None and len(relevant_fluents) == 1:
            self.logger.debug("Only one dimension is needed in the preconditions!")
            return self._construct_single_dimension_inequalities(state_data.loc[:, relevant_fluents[0]])

        equality_conditions, filtered_pre_state_df = self._filter_all_convex_hull_inconsistencies(state_data)
        if len(filtered_pre_state_df.columns) == 0:
            self.logger.debug("After filtering, no dimensions remained in the preconditions!")
            return equality_conditions, ConditionType.conjunctive

        needed_dimensions = len(filtered_pre_state_df.columns) + 1

        if filtered_pre_state_df.shape[1] < 2:
            self.logger.debug("After filtering, only one dimension remained in the preconditions!")
            return self._construct_single_dimension_inequalities(
                filtered_pre_state_df.loc[:, filtered_pre_state_df.columns[0]], equality_conditions)

        if filtered_pre_state_df.shape[0] < needed_dimensions:
            return self._create_disjunctive_preconditions(filtered_pre_state_df, equality_conditions)

        A, b = self._create_convex_hull_linear_inequalities(filtered_pre_state_df, display_mode=False)
        inequalities_strs = self._construct_pddl_inequality_scheme(A, b, filtered_pre_state_df.columns)
        inequalities_strs.extend(equality_conditions)
        return inequalities_strs, ConditionType.conjunctive

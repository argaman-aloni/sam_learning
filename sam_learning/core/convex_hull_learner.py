"""This module contains the ConvexHullLearner class."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from pddl_plus_parser.models import Precondition, PDDLFunction
from scipy.spatial import ConvexHull, convex_hull_plot_2d, QhullError

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import EquationSolutionType, ConditionType
from sam_learning.core.numeric_utils import construct_multiplication_strings, construct_linear_equation_string, \
    detect_linear_dependent_features, prettify_coefficients, construct_numeric_conditions, filter_constant_features, \
    get_num_independent_equations


class ConvexHullLearner:
    """Class that learns the convex hull of the preconditions of an action."""
    logger: logging.Logger
    action_name: str
    domain_functions: Dict[str, PDDLFunction]
    convex_hull_error_file_path: Path

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction]):
        self.logger = logging.getLogger(__name__)
        self.convex_hull_error_file_path = Path(os.environ["CONVEX_HULL_ERROR_PATH"])
        self.action_name = action_name
        self.domain_functions = domain_functions

    def _create_convex_hull_linear_inequalities(self, points_df: DataFrame,
                                                display_mode: bool = False) -> Tuple[List[List[float]], List[float]]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :param points_df: the dataframe containing the points that represent the values of the function in the states
            of the observations prior to the action's execution.
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

            failure_reason = f"Convex hull encountered an error condition and no solution was found " \
                             f"for action {self.action_name}"
            self.logger.warning(failure_reason)
            raise NotSafeActionError(self.action_name, failure_reason, EquationSolutionType.convex_hull_not_found)

    @staticmethod
    def _construct_pddl_inequality_scheme(
            coefficient_matrix: np.ndarray, border_points: np.ndarray, headers: List[str]) -> List[str]:
        """Construct the inequality strings in the appropriate PDDL format.

        :param coefficient_matrix: the matrix containing the coefficient vectors for each inequality.
        :param border_points: the convex hull point which ensures that Ax <= b.
        :param headers: the headers of the columns in the coefficient matrix.
        :return: the inequalities PDDL formatted strings.
        """
        inequalities = set()
        for inequality_coefficients, border_point in zip(coefficient_matrix, border_points):
            multiplication_functions = construct_multiplication_strings(inequality_coefficients, headers)
            constructed_left_side = construct_linear_equation_string(multiplication_functions)
            inequalities.add(f"(<= {constructed_left_side} {border_point})")

        return list(inequalities)

    def _create_disjunctive_preconditions(
            self, previous_state_matrix: DataFrame, equality_conditions: List[str] = []) -> Precondition:
        """Create the disjunctive representation of the preconditions.

        :param previous_state_matrix: the matrix containing the previous state values.
        :return: a disjunctive representation for the precondition in case a convex hull cannot be created.
        """
        if previous_state_matrix.shape[0] == 1:
            extended_conditions = [*equality_conditions]
            extended_conditions.extend(
                [f"(= {column} {previous_state_matrix[column].iloc[0]})" for column in previous_state_matrix.columns])
            return construct_numeric_conditions(
                extended_conditions, condition_type=ConditionType.conjunctive,
                domain_functions=self.domain_functions)

        disjunctive_precondition = Precondition("or")
        for _, row in previous_state_matrix.iterrows():
            conjunctive_conditions = []
            for column in previous_state_matrix.columns:
                conjunctive_conditions.append(f"(= {column} {row[column]})")

            disjunctive_precondition.add_condition(
                construct_numeric_conditions(
                    conjunctive_conditions, condition_type=ConditionType.conjunctive,
                    domain_functions=self.domain_functions))

        if len(equality_conditions) > 0:
            combined_condition = construct_numeric_conditions(
                equality_conditions, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)
            combined_condition.add_condition(disjunctive_precondition)
            return combined_condition

        return disjunctive_precondition

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

    def _construct_single_dimension_inequalities(
            self, single_column_df: Series, equality_strs: List[str] = []) -> Precondition:
        """Construct a single dimension precondition representation.

        :param single_column_df: the fluent only fluent that is relevant to the preconditions' creation.
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
        return construct_numeric_conditions(
            conditions, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)

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

        filtered_matrix, equality_conditions, _ = filter_constant_features(previous_state_matrix.copy())
        if len(filtered_matrix.columns) <= 1:
            return equality_conditions, filtered_matrix

        filtered_matrix, linear_dependency_conditions, _ = detect_linear_dependent_features(filtered_matrix)
        combined_conditions = equality_conditions + linear_dependency_conditions
        return combined_conditions, filtered_matrix

    def construct_safe_linear_inequalities(
            self, state_storge: Dict[str, List[float]],
            relevant_fluents: Optional[List[str]] = []) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        irrelevant_fluents = [fluent for fluent in state_storge.keys() if fluent not in relevant_fluents] if \
            relevant_fluents is not None else []
        state_data = DataFrame(state_storge).drop_duplicates().drop(columns=irrelevant_fluents)
        if relevant_fluents is not None and len(relevant_fluents) == 1:
            self.logger.debug("Only one dimension is needed in the preconditions!")
            return self._construct_single_dimension_inequalities(state_data.loc[:, relevant_fluents[0]])

        equality_conditions, filtered_pre_state_df = self._filter_all_convex_hull_inconsistencies(state_data)
        if len(filtered_pre_state_df.columns) == 0:
            self.logger.debug("After filtering, no dimensions remained in the preconditions!")
            return construct_numeric_conditions(
                equality_conditions, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)

        needed_dimensions = len(filtered_pre_state_df.columns) + 1

        if filtered_pre_state_df.shape[1] < 2:
            self.logger.debug("After filtering, only one dimension remained in the preconditions!")
            return self._construct_single_dimension_inequalities(
                filtered_pre_state_df.loc[:, filtered_pre_state_df.columns[0]], equality_conditions)

        if get_num_independent_equations(filtered_pre_state_df) < needed_dimensions:
            return self._create_disjunctive_preconditions(filtered_pre_state_df, equality_conditions)

        A, b = self._create_convex_hull_linear_inequalities(filtered_pre_state_df, display_mode=False)
        inequalities_strs = self._construct_pddl_inequality_scheme(A, b, filtered_pre_state_df.columns)
        inequalities_strs.extend(equality_conditions)
        return construct_numeric_conditions(
            inequalities_strs, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions)

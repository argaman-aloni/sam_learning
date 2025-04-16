"""This module contains the ConvexHullLearner class."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from pddl_plus_parser.models import Precondition, PDDLFunction
from scipy.spatial import ConvexHull, QhullError

from sam_learning.core.learning_types import ConditionType
from sam_learning.core.numeric_learning.convex_hull_learner import ConvexHullLearner
from sam_learning.core.numeric_learning.numeric_utils import (
    prettify_coefficients,
    construct_numeric_conditions,
    construct_projected_variable_strings,
    extended_gram_schmidt,
    display_convex_hull,
    create_monomials,
    create_polynomial_string,
    divide_span_by_common_denominator,
    remove_complex_linear_dependencies
)


class IncrementalConvexHullLearner(ConvexHullLearner):
    """Class that learns the convex hull of the preconditions of an action."""

    _convex_hull: Optional[ConvexHull]
    _gsp_base: Optional[List[List[float]]]
    _complementary_base: Optional[List[List[float]]]
    _spanning_standard_base: bool
    data: DataFrame
    relevant_fluents: Optional[List[str]]
    affine_independent_data: DataFrame
    additional_dependency_conditions: List[str]

    def __init__(self, action_name: str, domain_functions: Dict[str, PDDLFunction], polynom_degree: int = 0):
        super().__init__(action_name, {function.name: function for function in domain_functions.values()})
        self._convex_hull = None
        self._gsp_base = None
        self._complementary_base = None
        self._spanning_standard_base = False
        functions = list([function.untyped_representation for function in domain_functions.values()])
        monomials = create_monomials(functions, polynom_degree)
        self.data = DataFrame(columns=[create_polynomial_string(monomial) for monomial in monomials])
        self.relevant_fluents = None
        self.affine_independent_data = None
        self.additional_dependency_conditions = []

    @staticmethod
    def _calculate_orthonormal_complementary_base(orthonormal_base: List[List[float]], num_dimensions: int) -> List[List[float]]:
        """Calculates the orthonormal complementary base of the given orthonormal base.

        :param orthonormal_base: the orthonormal base spanning the points in the storage dataframe.
        :param num_dimensions: the number of dimensions of the original space.
        :return: the orthonormal complementary base of the given orthonormal base.
        """
        diagonal_eye = [list(vector) for vector in np.eye(num_dimensions)]
        return extended_gram_schmidt(diagonal_eye, orthonormal_base)

    @staticmethod
    def _calculate_orthonormal_base(points_df: DataFrame) -> List[List[float]]:
        """Calculates the orthonormal base of the given points.

        :param points_df: the points to calculate the orthonormal base from.
        :return: the orthonormal base of the given points.
        """
        points = points_df.to_numpy()
        shift_axis = points[0].tolist()  # selected the first vector to be the start of the axis.
        shifted_points = points - shift_axis
        return extended_gram_schmidt(shifted_points)

    def _shift_new_point(self) -> np.ndarray:
        """Shifts the points based on the first sample in the dataframe.

        Note: if the points are spanning the original space, we do not need to shift the points.

        :return: the shifted point based on the first sample in the dataframe.
        """
        last_sample = self.affine_independent_data.iloc[-1].to_numpy()
        shifted_last_sample = last_sample - self.affine_independent_data.iloc[0] if not self._spanning_standard_base else last_sample
        return shifted_last_sample

    def _is_spanned_in_base(self) -> bool:
        """Checks if the new point is spanned by the base.

        :return: whether the new point is spanned by the base.
        """
        shifted_last_sample = self._shift_new_point()
        return len(extended_gram_schmidt([list(shifted_last_sample)], self._gsp_base)) == 0

    def _learn_new_bases(self) -> None:
        """Learns the new bases of the points in the storage dataframe.

        :return: the new bases of the points in the storage dataframe.
        """
        self.logger.debug("Learning the new normal and the complementary bases of the points in the storage dataframe.")
        self._gsp_base = self._calculate_orthonormal_base(self.affine_independent_data)
        self._complementary_base = self._calculate_orthonormal_complementary_base(self._gsp_base, self.affine_independent_data.shape[1])
        self._complementary_base = divide_span_by_common_denominator(self._complementary_base)
        if len(self._gsp_base) == self.affine_independent_data.shape[1]:
            self.logger.debug("The points are spanning the original space, no need to project the points.")
            self._gsp_base = [list(vector) for vector in np.eye(self.affine_independent_data.shape[1])]
            self._spanning_standard_base = True

    def _learn_new_convex_hull(self) -> None:
        """
        Learns the convex hull of the points in the storage dataframe.
        :return: the convex hull of the points in the storage dataframe.
        """
        if self._convex_hull is not None:
            self._convex_hull.close()
            self._convex_hull = None

        self.logger.debug("Creating the convex hull for the first time (or in case that the base had changed).")
        points = self.affine_independent_data.to_numpy()
        shift_axis = points[0].tolist() if not self._spanning_standard_base else [0] * len(self.affine_independent_data.columns.tolist())
        projected_points = np.dot(points - shift_axis, np.array(self._gsp_base).T)
        self._convex_hull = ConvexHull(projected_points, incremental=True)

    def add_new_point(self, point: Dict[str, float]) -> None:
        """Adds a new point to the convex hull learner.

        Note:
            This method is supposed to improve the performance of the CH calculations by incrementally adding points.

        :param point: the point to add to the convex hull learner.
        """
        new_sample = DataFrame.from_dict(data={key: [value] for key, value in point.items()}, orient="columns")
        # assuming that if a feature is relevant to the preconditions it should always appear in the dataframe.
        if len(self.data) == 0:
            self.data = new_sample  # This is to avoid observing warnings when adding the first sample.
            self.affine_independent_data, self.additional_dependency_conditions = remove_complex_linear_dependencies(self.data)
            self.logger.debug("Added the first sample to the points dataframe.")
            return

        # check if the new sample is already in the dataframe
        concat_data = pd.concat([self.data, new_sample], ignore_index=True).dropna(axis=1)
        if concat_data.drop_duplicates().shape[0] == self.data.shape[0]:
            self.logger.debug("The new point is already in the storage, not adding it again.")
            return

        self.data = concat_data

        old_additional_dependency_conditions = self.additional_dependency_conditions
        self.affine_independent_data, self.additional_dependency_conditions = remove_complex_linear_dependencies(self.data)
        
        recalculate_base = False
        recalculate_ch = False
        if old_additional_dependency_conditions != self.additional_dependency_conditions:
            self.logger.debug("The new sample not holds the additional dependency conditions.")
            recalculate_base = True
            if self._gsp_base is not None:
                recalculate_ch = True
        elif self._gsp_base is None:
            recalculate_base = True
        else:
            # There exists a base, and we need to check if the new point is spanned by the base.
            self.logger.debug("Validating if the new point is spanned by the GSP base.")
            point_spanned_by_base = self._is_spanned_in_base()
            if not point_spanned_by_base:
                self.logger.debug("The base has changed, need to recalculate the convex hull.")
                recalculate_base = True
                recalculate_ch = True
            else:
                # add the new point to the convex hull
                projected_new_point = np.dot(self._shift_new_point(), np.array(self._gsp_base).T)
                if self._convex_hull is not None:
                    try:
                        self._convex_hull.add_points([projected_new_point])
                    except QhullError:
                        self.logger.debug(
                            f"Failed to add a new point to the convex hull. \n\tPrevious data - {self._convex_hull.points}, new point - {projected_new_point}"
                        )
                        recalculate_ch = True

            if len(self._gsp_base) == 1:
                self.logger.debug("The created base is one dimensional, cannot create a convex hull.")
                return
        
        if recalculate_base:
            self._learn_new_bases()
            
        if recalculate_ch:
            self._learn_new_convex_hull()

    def _create_ch_coefficients_data(self, display_mode: bool = True) -> Tuple[List[List[float]], List[float]]:
        """Runs the convex hull algorithm on the given input points.

        :param points: the points to run the convex hull algorithm on.
        :param display_mode: whether to display the convex hull.
        :return: the coefficients of the planes that represent the convex hull and the border point.
        """
        display_convex_hull(self.action_name, display_mode, self._convex_hull)
        equations = np.unique(self._convex_hull.equations, axis=0)
        A = equations[:, : self._convex_hull.points.shape[1]]
        b = -equations[:, self._convex_hull.points.shape[1]]
        coefficients = [prettify_coefficients(row) for row in A]
        border_point = prettify_coefficients(b)
        return coefficients, border_point

    def _incremental_create_ch_inequalities(
        self, display_mode: bool = False
    ) -> Tuple[List[List[float]], List[float], List[str], Optional[List[str]]]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :return: the matrix representing the inequalities of the planes created by the convex hull as well as the
            names of the features that are part of the convex hull.

        Note: the returned values represents the linear inequalities of the convex hull, i.e.,  Ax <= b.
        """
        if self._gsp_base is None:
            self.logger.debug("The base is not yet learned since didn't receive enough points.")
            raise ValueError()

        points = self.affine_independent_data.to_numpy()
        shift_axis = points[0].tolist()  # selected the first vector to be the start of the axis.

        if len(self._gsp_base) == self.affine_independent_data.shape[1]:
            self.logger.debug("The points are spanning the original space and the basis is full rank.")
            coefficients, border_point = self._create_ch_coefficients_data(display_mode)
            return coefficients, border_point, self.affine_independent_data.columns.tolist(), None

        projected_points = np.dot(points - shift_axis, np.array(self._gsp_base).T)

        if self._convex_hull is None:
            self.logger.debug("The projected points are 1D, creating min-max conditions on the new base.")
            coefficients = [[-1], [1]]
            border_point = prettify_coefficients([-projected_points.min(), projected_points.max()])

        else:
            coefficients, border_point = self._create_ch_coefficients_data(display_mode)

        transformed_vars = construct_projected_variable_strings(self.affine_independent_data.columns.tolist(), shift_axis, self._gsp_base)
        self.logger.debug("Constructing the conditions to verify that points are in the correct span.")
        transformed_orthonormal_vars = construct_projected_variable_strings(
            self.affine_independent_data.columns.tolist(), shift_axis, [list(vector) for vector in np.eye(points.shape[1])]
        )
        span_verification_conditions = self._construct_pddl_inequality_scheme(
            np.array(self._complementary_base), np.zeros(len(self._complementary_base)), transformed_orthonormal_vars, sign_to_use="=",
        )
        return coefficients, border_point, transformed_vars, span_verification_conditions

    def construct_safe_linear_inequalities(self, *args, **kwargs) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        if len(self.affine_independent_data) == 0:
            self.logger.debug("No observations were given to learn any conditions.")
            return Precondition("and")

        if len(self.affine_independent_data.columns.tolist()) == 1:
            self.logger.debug("Only one dimension is needed in the preconditions!")
            return self._construct_single_dimension_inequalities(self.affine_independent_data.loc[:, self.affine_independent_data.columns.tolist()[0]])

        try:
            A, b, column_names, additional_projection_conditions = self._incremental_create_ch_inequalities()
            self.logger.debug(f"Constructing the PDDL inequality scheme for the action {self.action_name}.")
            inequalities_strs = self._construct_pddl_inequality_scheme(A, b, column_names)
            if additional_projection_conditions is not None:
                inequalities_strs.extend(additional_projection_conditions)

            self.logger.debug("Constructing the precondition object from the constructed strings.")
            return construct_numeric_conditions(
                [*inequalities_strs, *self.additional_dependency_conditions],
                condition_type=ConditionType.conjunctive,
                domain_functions=self.domain_functions
            )

        except (QhullError, ValueError):
            self.logger.warning(
                "Convex hull failed to create a convex hull, using disjunctive preconditions "
                "(probably since the rank of the matrix is 2 and it cannot create a degraded "
                "convex hull)."
            )
            return self._create_disjunctive_preconditions(self.affine_independent_data, [])

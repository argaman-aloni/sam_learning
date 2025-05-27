"""This module contains the ConvexHullLearner class."""

import logging
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pddl_plus_parser.models import Precondition, PDDLFunction
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cdist

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import ConditionType, EquationSolutionType
from sam_learning.core.numeric_learning.numeric_utils import (
    prettify_coefficients,
    construct_numeric_conditions,
    create_monomials,
    create_polynomial_string,
    EPSILON,
    construct_pddl_inequality_scheme,
    create_grounded_monomials,
)

LABEL_COLUMN = "label"
MAX_ALLOWED_CONDITIONS = 100
MIN_ALLOWED_UNRESOLVED_POINTS = 5


def plot_planes_2d(X: DataFrame, y: Series, planes, axis_names: List[str], grid: int = 200):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 1][axis_names[0]], X[y == 1][axis_names[1]], marker="o", label="Positive", edgecolor="black")
    plt.scatter(X[y == -1][axis_names[0]], X[y == -1][axis_names[1]], marker="x", label="Negative", color="gray")

    xmin, xmax = X[axis_names[0]].min(), X[axis_names[0]].max()
    ymin, ymax = X[axis_names[1]].min(), X[axis_names[1]].max()
    xgrid = np.linspace(xmin - 0.5, xmax + 0.5, grid)
    ygrid = np.linspace(ymin - 0.5, ymax + 0.5, grid)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    colors = plt.cm.tab10(np.arange(len(planes)))

    for idx, (w, b) in enumerate(planes):
        # line endpoints
        color = colors[idx]
        Z = w[0] * Xg + w[1] * Yg
        plt.contourf(Xg, Yg, Z >= b, levels=[0.5, 1], alpha=0.3, colors=[color], zorder=1)

        # Plot boundary line
        if abs(w[1]) > 1e-6:
            x_vals = np.array([xmin - 0.5, xmax + 0.5])
            y_vals = (b - w[0] * x_vals) / w[1]
            plt.plot(x_vals, y_vals, "--", linewidth=2, color=color, label=f"Plane {idx + 1}", zorder=2)
        else:
            # vertical line
            x0 = b / w[0]
            plt.axvline(x0, color=color, linestyle="--", linewidth=2, label=f"Plane {idx + 1}", zorder=2)

    plt.xlim(xmin - 0.5, xmax + 0.5)
    plt.ylim(X[axis_names[1]].min() - 0.5, X[axis_names[1]].max() + 0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("2D Polytope Planes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class IncrementalSVMLearner:
    """A class for incrementally learning support vector machine (SVM) conditions for convex-based models.

    This class is primarily designed to process and analyze a dataset to incrementally determine support
    vector machine linear conditions that describe convex hull boundaries. These boundaries can then
    be used in tasks such as constructing mathematical preconditions for PDDL (Planning Domain Definition
    Language) models of dynamic processes. The core functionality includes adding new points to the model,
    generating linear inequalities via SVM classification, and constructing PDDL-representable conditions.
    """

    data: DataFrame
    relevant_fluents: Optional[List[str]]

    def __init__(
        self,
        action_name: str,
        domain_functions: Dict[str, PDDLFunction],
        polynom_degree: int = 0,
        relevant_fluents: Optional[List[str]] = None,
    ):
        self.action_name = action_name
        functions = list([function.untyped_representation for function in domain_functions.values()])
        self.monomials = create_monomials(functions, polynom_degree)
        self.data = DataFrame(columns=[*[create_polynomial_string(monomial) for monomial in self.monomials], LABEL_COLUMN])
        self.relevant_fluents = relevant_fluents if relevant_fluents is not None else self.data.columns.tolist()
        self.domain_functions = {function.name: function for function in domain_functions.values()}
        self.logger = logging.getLogger(__name__)

    def add_new_point(self, point: Dict[str, PDDLFunction], is_successful: bool = True) -> None:
        """Adds a new point to the labeled dataset.

        Note:
            This method is supposed to improve the performance of the CH calculations by incrementally adding points.

        :param point: the point to add to the convex hull learner.
        :param is_successful:
        """
        grounded_monomials = create_grounded_monomials(self.monomials, point)
        new_sample = pd.DataFrame({**{k: [v] for k, v in grounded_monomials.items()}, LABEL_COLUMN: [1 if is_successful else -1]})
        self.data = pd.concat([self.data, new_sample], ignore_index=True)
        self.data.dropna(axis=1, inplace=True)

    def _incremental_create_svm_linear_conditions(self, debug: bool = False) -> List[Tuple[List[float], float]]:
        """Create the convex hull and returns the matrix representing the inequalities.

        :return: the matrix representing the inequalities of the planes created by the convex hull as well as the
            names of the features that are part of the convex hull.

        Note: the returned values represents the linear inequalities of the convex hull, i.e.,  Ax <= b.
        """
        planes = []
        no_label_columns = self.data.drop(columns=[LABEL_COLUMN])
        X_reminder, y_reminder = no_label_columns.to_numpy(), self.data[LABEL_COLUMN]
        for i in range(MAX_ALLOWED_CONDITIONS):
            if len(X_reminder) <= MIN_ALLOWED_UNRESOLVED_POINTS:
                self.logger.debug(f"The number of points to classify - {len(X_reminder)} is less than the minimum allowed points.")
                break
            if len(y_reminder.unique()) <= 1:
                self.logger.debug("Remaining points have only one label cannot continue running SVM.")
                break

            # Find the closest negative point to any positive
            neg_mask = y_reminder == -1
            pos_mask = y_reminder == 1
            X_neg = X_reminder[neg_mask]
            X_pos = X_reminder[pos_mask]
            distances = cdist(X_pos.tolist(), X_neg.tolist(), metric="euclidean")
            min_dist_index = distances.argmin()
            _, col = divmod(min_dist_index, distances.shape[1])
            closest_neg_idx_in_X_neg = col
            closest_neg_global_idx = np.where(neg_mask)[0][closest_neg_idx_in_X_neg]

            # Filter the data to keep only the closest negative point and all positive points
            final_mask = np.zeros(len(y_reminder), dtype=bool)
            final_mask[pos_mask] = True
            final_mask[closest_neg_global_idx] = True
            X_filtered = X_reminder[final_mask]
            y_filtered = y_reminder[final_mask]

            classifier = LinearSVC(random_state=0, tol=EPSILON, dual=False, max_iter=5000, C=1e10)
            classifier.fit(X_filtered, y_filtered.tolist())
            A = classifier.coef_[0].copy()
            b = -classifier.intercept_[0].copy()
            coefficients, border_point = prettify_coefficients(A), prettify_coefficients([b])[0]

            # Remove correctly classified points
            preds = classifier.predict(X_reminder)  # numpy array
            is_neg = y_reminder.values == -1  # numpy bool array
            pred_is_neg = preds == -1  # numpy bool array

            # correctly classified negatives
            correctly_classified_neg = is_neg & pred_is_neg

            # keep mask: everything except those
            keep_mask = ~correctly_classified_neg

            # The classifier is not able to classify any more points
            # Specifically the last negative point can't be classified
            if np.sum(~keep_mask) == 0:
                # remove the last negative point from the data
                final_mask = np.ones(len(y_reminder), dtype=bool)
                final_mask[closest_neg_global_idx] = False
                X_reminder = X_reminder[final_mask]
                y_reminder = y_reminder[final_mask]
                continue

            # apply the mask
            X_reminder = X_reminder[keep_mask]
            y_reminder = y_reminder.iloc[keep_mask]

            self.logger.debug(
                f"Plane {' + '.join([f'{coefficients[j]} ' f'* {col}' for j, col in enumerate(no_label_columns.columns.tolist())])} >= {b} (removed {np.sum(~keep_mask)} points)"
            )

            # Add predictions of negative points for later comparison
            preds = classifier.predict(no_label_columns.to_numpy())
            pred_is_neg = preds == -1

            planes.append((coefficients, border_point, pred_is_neg.astype(int)))

        # Remove planes that are subsets of others
        filtered_planes = []
        for i in range(len(planes)):
            coeff_i, bp_i, pred_i = planes[i]
            is_subset = False

            for j in range(i + 1, len(planes)):
                # Check if the plane i is a subset of another
                result = planes[j][2] - pred_i
                if -1 not in result:  # plain j is superset of plane i
                    is_subset = True
                    break

            if not is_subset:
                filtered_planes.append((coeff_i, bp_i))
        planes = filtered_planes

        if debug:
            if len(no_label_columns.columns.tolist()) == 2:
                plot_planes_2d(self.data, self.data[LABEL_COLUMN].to_numpy(), planes, no_label_columns.columns.tolist())

        return planes

    def construct_linear_inequalities(self) -> Precondition:
        """Constructs the linear inequalities strings that will be used in the learned model later.

        :return: the inequality strings and the type of equations that were constructed (injunctive / disjunctive)
        """
        if len(self.data) == 0:
            self.logger.debug("No observations were given - no conditions could be learned.")
            return Precondition("and")

        try:
            planes = self._incremental_create_svm_linear_conditions()
            self.logger.debug(f"Constructing the PDDL inequality scheme for the action {self.action_name}.")
            preconditions = Precondition("and")
            for A, b in planes:
                inequalities_strs = construct_pddl_inequality_scheme(A, b, self.data.columns.tolist(), sign_to_use=">=")
                # Since each plane is a linear inequality, we can extract the condition and add it to the preconditions.
                inequality_condition = construct_numeric_conditions(
                    inequalities_strs,
                    condition_type=ConditionType.conjunctive,
                    domain_functions=self.domain_functions,
                )
                for operand in inequality_condition.operands:
                    preconditions.add_condition(operand)

            return preconditions

        except ValueError:
            self.logger.warning("Failed to create the SVM based conditions.")
            raise NotSafeActionError(
                name=self.action_name, reason="SVM failed to execute.", solution_type=EquationSolutionType.svm_failed_to_train
            )

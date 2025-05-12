"""This module contains the ConvexHullLearner class."""
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from pddl_plus_parser.models import Precondition, PDDLFunction
from sklearn.svm import LinearSVC

from sam_learning.core.exceptions import NotSafeActionError
from sam_learning.core.learning_types import ConditionType, EquationSolutionType
from sam_learning.core.numeric_learning.numeric_utils import (
    prettify_coefficients,
    construct_numeric_conditions,
    create_monomials,
    create_polynomial_string,
    EPSILON,
    construct_pddl_inequality_scheme,
    prettify_floating_point_number,
    create_grounded_monomials,
)

LABEL_COLUMN = "label"
MAX_ALLOWED_CONDITIONS = 10
MIN_ALLOWED_UNRESOLVED_POINTS = 5


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
        self, action_name: str, domain_functions: Dict[str, PDDLFunction], polynom_degree: int = 0, relevant_fluents: Optional[List[str]] = None,
    ):
        self.action_name = action_name
        functions = list([function.untyped_representation for function in domain_functions.values()])
        self.monomials = create_monomials(functions, polynom_degree)
        self.data = DataFrame(columns=[*[create_polynomial_string(monomial) for monomial in monomials], LABEL_COLUMN])
        self.relevant_fluents = relevant_fluents if relevant_fluents is not None else self.data.columns.tolist()
        self.domain_functions = {function.name: function for function in domain_functions.values()}
        self.logger = logging.getLogger(__name__)

    def add_new_point(self, point: Dict[str, float], is_successful: bool = True) -> None:
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

    def _incremental_create_svm_linear_conditions(self) -> List[Tuple[List[float], float]]:
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

            classifier = LinearSVC(random_state=0, tol=EPSILON)
            classifier.fit(X_reminder, y_reminder.tolist())
            A = classifier.coef_[0].copy()
            b = -classifier.intercept_[0].copy()
            coefficients, border_point = prettify_coefficients(A), prettify_floating_point_number(b)
            planes.append((coefficients, border_point))

            # Remove correctly classified points
            predictions = classifier.predict(X_reminder)
            mask = predictions != y_reminder
            X_reminder, y_reminder = X_reminder[mask], y_reminder[mask]
            self.logger.debug(f"Plane {i+1}: {A[0]:+.3f}·x + {A[1]:+.3f}·y {-b:+.3f} = 0 (removed {np.sum(~mask)} points)")

            if len(X_reminder) == 0:
                print("All points classified. Stopping.")
                break

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
                    inequalities_strs, condition_type=ConditionType.conjunctive, domain_functions=self.domain_functions,
                )
                for operand in inequality_condition.operands:
                    preconditions.add_condition(operand)

            return preconditions

        except ValueError:
            self.logger.warning("Failed to create the SVM based conditions.")
            raise NotSafeActionError(name=self.action_name, reason="SVM failed to execute.", solution_type=EquationSolutionType.svm_failed_to_train)

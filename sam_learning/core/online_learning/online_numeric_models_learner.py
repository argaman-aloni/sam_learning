"""An online numeric model learning algorithm."""
from pandas import DataFrame, Series

from sam_learning.core.numeric_learning.linear_regression_learner import LinearRegressionLearner
from sam_learning.core.numeric_learning.numeric_utils import create_grounded_monomials, create_monomials, create_polynomial_string
from sam_learning.core.online_learning.incremental_convex_hull_learner import IncrementalConvexHullLearner
from sam_learning.core.online_learning.incremental_svm_learner import IncrementalSVMLearner

"""Module to calculate the information gain of the propositional part of an action."""
import logging
from typing import Set, Tuple, Dict, List

from pddl_plus_parser.models import Precondition, PDDLFunction, NumericalExpressionTree


class OnlineNumericModelLearner:
    """
    Manages the online learning of numeric action models for a given action. This class integrates
    multiple learning algorithms to construct safe and optimistic models of numeric actions over time.
    The primary purpose is to update, evaluate, and retrieve action models based on observed data.

    The class maintains three types of learners:
    1. Incremental convex hull learner: Handles the boundaries of valid preconditions.
    2. Support vector machine (SVM) learner: Manages decision boundaries between successful and unsuccessful states.
        Used to learn the precondition of the action.
    3. Linear regression learner: Determines numeric effects of actions.

    These learners work collaboratively to construct action models that guide decisions during planning
    in dynamic environments. The class also offers utility methods to add data, validate safe state
    models, and retrieve copies of internal learners when needed.
    """

    logger: logging.Logger
    action_name: str

    def __init__(
        self, action_name: str, pb_functions: Dict[str, PDDLFunction], polynom_degree: int = 0, epsilon: float = 0.0, qhull_options: str = "",
    ):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self._convex_hull_learner = IncrementalConvexHullLearner(
            action_name=action_name, domain_functions=pb_functions, polynom_degree=polynom_degree, qhull_options=qhull_options, epsilon=epsilon,
        )
        self._svm_learner = IncrementalSVMLearner(action_name=action_name, domain_functions=pb_functions, polynom_degree=polynom_degree,)
        self._linear_regression_learner = LinearRegressionLearner(
            action_name=action_name, domain_functions=pb_functions, polynom_degree=polynom_degree,
        )
        self._monomials = create_monomials(list(pb_functions.keys()), polynom_degree)
        self._data_columns = [create_polynomial_string(monomial) for monomial in self._monomials]

    @property
    def monomials(self) -> List[List[str]]:
        """
        Returns the list of monomials used for feature construction in the learning process.

        :return: A list of lists, where each inner list represents a monomial as a list of strings.
        """
        return self._monomials

    @property
    def data_columns(self) -> List[str]:
        """Returns the data columns of the convex hull learner.

        :return: the data columns of the convex hull learner.
        """
        return self._data_columns

    def _add_positive_pre_state_observation(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds a new numeric positive pre-state observation for the action associated with the current instance.

        This method utilizes state fluents and processes them through different machine learning
        learners, such as convex hull, linear regression, and SVM learners, to extract meaningful
        information about the pre-state conditions of the action being analyzed.

        :param state_fluents: Dictionary containing state variables and their associated
            PDDLFunction values that represent the state of the environment.
        """
        self.logger.info(f"Adding a new numeric positive pre-state observation for the action {self.action_name}.")
        sample_dataset = create_grounded_monomials(self._monomials, state_fluents)
        self._convex_hull_learner.add_new_point(sample_dataset)
        self._linear_regression_learner.add_new_observation(sample_dataset, store_in_prev_state=True)
        self._svm_learner.add_new_point(state_fluents, is_successful=True)

    def _add_positive_post_state_observation(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds a new numeric positive post-state observation for the action associated with the current instance.

        :param state_fluents: Dictionary containing state variables and their associated PDDLFunction with their values.
        """
        self.logger.info(f"Adding a new numeric positive post-state observation for the action {self.action_name}.")
        next_state_values = {state_fluent_lifted_str: state_fluent_data.value for state_fluent_lifted_str, state_fluent_data in state_fluents.items()}
        self._linear_regression_learner.add_new_observation(next_state_values, store_in_prev_state=False)

    def _add_negative_pre_state_observation(self, state_fluents: Dict[str, PDDLFunction]) -> None:
        """Adds a new negative sample observation for the pre-state of the action being learned.
        The method records a state of fluents, marking it as a failed attempt to perform
        the associated action. This helps the SVM model in improving its classification or
        decision-making capability for the given action.

        :param state_fluents: Dictionary mapping string keys to PDDLFunction
            objects which define the current state of fluents.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        self._svm_learner.add_new_point(state_fluents, is_successful=False)

    def add_transition_data(
        self, pre_state_functions: Dict[str, PDDLFunction], post_state_functions: Dict[str, PDDLFunction] = None, is_transition_successful: bool = True
    ) -> None:
        """Adds transition data, including pre-state and post-state functions, to the system. If the transition is
        successful, it updates positive pre- and post-state observations. Otherwise, it updates negative
        pre-state observations and logs a debug message indicating that no effects can be learned from an inapplicable
        state transition.

        :param pre_state_functions: A dictionary of pre-state functions representing
            the state of the system before the transition occurs.
        :param post_state_functions: A dictionary of post-state functions representing
            the state of the system after the transition occurs.
        :param is_transition_successful: A boolean indicating whether the transition
            was successful.
        """
        if is_transition_successful:
            self._add_positive_pre_state_observation(pre_state_functions)
            self._add_positive_post_state_observation(post_state_functions)
            return

        self._add_negative_pre_state_observation(pre_state_functions)
        self.logger.debug("Cannot learn anything on the effects from an inapplicable state transition!")

    def get_safe_model(self) -> Tuple[Precondition, Set[NumericalExpressionTree]]:
        """Returns the safe model of the action.

        :return: the safe model of the action.
        """
        self.logger.info(f"Getting the safe model for the action {self.action_name}.")
        safe_precondition = self._convex_hull_learner.construct_safe_linear_inequalities()
        numeric_effects, _, _ = self._linear_regression_learner.construct_assignment_equations()

        return safe_precondition, numeric_effects

    def get_optimistic_model(self) -> Tuple[Precondition, Set[NumericalExpressionTree]]:
        """Returns the optimistic model of the action.

        :return: the optimistic model of the action.
        """
        self.logger.info(f"Getting the optimistic model for the action {self.action_name}.")
        safe_precondition = self._svm_learner.construct_linear_inequalities()
        numeric_effects, _, _ = self._linear_regression_learner.construct_assignment_equations(allow_unsafe=True)

        return safe_precondition, numeric_effects

    def is_state_in_safe_model(self, state: Dict[str, PDDLFunction]) -> bool:
        """Checks if state predicates hold in the safe model.

        :param state: The state predicates to check.
        :return: True if the state is in the safe model, False otherwise.
        """
        sample_dataset = create_grounded_monomials(self._monomials, state)
        return self._convex_hull_learner.is_point_in_convex_hull(Series({func_name: val for func_name, val in sample_dataset.items()}))

    def copy_convex_hull_learner(self, one_shot: bool) -> IncrementalConvexHullLearner:
        """Returns a copy of the convex hull learner.

        :return: a copy of the convex hull learner.
        """
        return self._convex_hull_learner.copy(one_shot)

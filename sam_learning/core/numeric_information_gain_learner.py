"""A module containing the algorithm to calculate the information gain of new samples."""
import logging
from typing import Dict, List, Optional

import pandas as pd
from pandas import DataFrame, Series
from pddl_plus_parser.models import PDDLFunction, Predicate
from scipy.spatial import Delaunay, QhullError

from sam_learning.core.consistent_model_validator import NumericConsistencyValidator


class InformationGainLearner(NumericConsistencyValidator):
    """Information gain calculation of the numeric part of an action."""

    logger: logging.Logger
    action_name: str
    lifted_predicates: List[str]
    positive_discrete_sample_df: DataFrame
    negative_combined_sample_df: DataFrame
    _cached_convex_hull: Optional[Delaunay]

    def __init__(self, action_name: str):
        super().__init__(action_name)
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.lifted_predicates = []
        self.positive_discrete_sample_df = None
        self.negative_combined_sample_df = None

    def _remove_redundant_propositional_features(self, positive_propositional_sample: List[Predicate]) -> List[str]:
        """Removes features that are not needed for the calculation of the information gain.

        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        :return: the list of the propositional predicates that are needed for the calculation.
        """
        self.logger.info("Removing propositional features that are not needed for the calculation.")
        state_predicates_names = [predicate.untyped_representation for predicate in positive_propositional_sample]
        columns_to_drop = [feature for feature in self.lifted_predicates if feature not in state_predicates_names]
        self.positive_discrete_sample_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
        self.positive_discrete_sample_df.drop(columns_to_drop, axis=1, errors="ignore", inplace=True)
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

        if len(self.numeric_positive_samples) == 0 and relevant_numeric_features and len(relevant_numeric_features) > 0:
            self.logger.debug("There are no positive samples to calculate the information "
                              "gain from and there are features to learn numerically.")
            return False

        functions_to_explore = self.numeric_positive_samples.columns.tolist() \
            if relevant_numeric_features is None else relevant_numeric_features
        if len(functions_to_explore) == 0:
            return True

        positive_points_data = self.numeric_positive_samples[functions_to_explore]
        new_point_data = DataFrame({col: [new_numeric_sample[col].value] for col in functions_to_explore})
        try:
            return self._in_hull(new_point_data, self.numeric_positive_samples[functions_to_explore],
                                 use_cached_ch=use_cache)

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
        new_sample_predicates = [predicate.untyped_representation for predicate in new_propositional_sample]
        if len(self.lifted_predicates) > 0 and \
                len(set(new_sample_predicates).intersection(self.lifted_predicates)) == 0:
            self.logger.debug("None of the existing preconditions hold in the new sample. "
                              "It is not informative since it will never be applicable.")
            return True

        self.logger.debug("Creating a new model from the new sample and validating if the new model "
                          "contains a negative sample.")
        functions_to_explore = self.numeric_positive_samples.columns.tolist() \
            if relevant_numeric_features is None else relevant_numeric_features
        if len(functions_to_explore) == 0:
            return True

        new_model_data = self.numeric_positive_samples[functions_to_explore].copy()
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           new_numeric_sample.items()}
        self.logger.debug("Adding the new sample to the model and creating a model from the combined data.")
        new_model_data.loc[len(new_model_data)] = new_sample_data

        for _, negative_sample in self.negative_combined_sample_df.iterrows():
            if not self._validate_negative_sample_in_state_predicates(
                    negative_sample[self.lifted_predicates], new_sample_predicates):
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

    def init_dataframes(self, valid_lifted_functions: List[str], lifted_predicates: List[str]) -> None:
        """Initializes the data frames used to calculate the information gain.

        :param valid_lifted_functions: the lifted functions matching the action (used to avoid adding redundant ones).
        :param lifted_predicates: the lifted predicates matching the action.
        """
        super().init_numeric_dataframes(valid_lifted_functions)
        self.positive_discrete_sample_df = DataFrame(columns=lifted_predicates)
        self.negative_combined_sample_df = DataFrame(columns=lifted_predicates + valid_lifted_functions)
        self.lifted_predicates = lifted_predicates

    def add_positive_sample(self, positive_numeric_sample: Dict[str, PDDLFunction],
                            positive_propositional_sample: List[Predicate]) -> None:
        """Adds a positive sample to the samples list used to create the action's precondition.

        :param positive_numeric_sample: the numeric functions representing the positive sample.
        :param positive_propositional_sample: the propositional predicates representing the positive sample.
        """
        super().add_positive_numeric_sample(positive_numeric_sample)
        self.logger.info(f"Adding a new positive sample for the action {self.action_name}.")
        filtered_predicates_names = self._remove_redundant_propositional_features(positive_propositional_sample)
        new_sample_data = {predicate: 1.0 for predicate in filtered_predicates_names}
        if len(self.positive_discrete_sample_df.columns) == 0:
            return

        self.positive_discrete_sample_df.loc[len(self.positive_discrete_sample_df)] = new_sample_data

    def add_negative_sample(self, numeric_negative_sample: Dict[str, PDDLFunction],
                            negative_propositional_sample: List[Predicate]) -> None:
        """Adds a negative sample that represent a state in which an action .

        :param numeric_negative_sample: the numeric functions representing the negative sample.
        :param negative_propositional_sample: the propositional predicates representing the negative sample.
        """
        self.logger.info(f"Adding a new negative sample for the action {self.action_name}.")
        super().add_negative_numeric_sample(numeric_negative_sample)
        new_sample_data = {lifted_fluent_name: fluent.value for lifted_fluent_name, fluent in
                           numeric_negative_sample.items()}
        observed_literals = [predicate.untyped_representation for predicate in negative_propositional_sample if
                             predicate.untyped_representation in self.lifted_predicates]
        for predicate in self.lifted_predicates:
            new_sample_data[predicate] = 1.0 if predicate in observed_literals else 0.0

        self.negative_combined_sample_df.loc[len(self.negative_combined_sample_df)] = new_sample_data

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

        relevant_functions = self.numeric_positive_samples.columns.tolist()
        new_point_data = DataFrame({col: [new_numeric_sample[col].value] for col in relevant_functions})
        sample_values = [new_numeric_sample[col].value for col in relevant_functions]
        try:
            return self._in_hull(new_point_data, self.numeric_positive_samples) and \
                self._locate_sample_in_df(sample_values, self.numeric_positive_samples) == -1

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
        if (self._are_numeric_dataframes_empty()
                and len(self.positive_discrete_sample_df) == 0 and len(self.negative_combined_sample_df) == 0):
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

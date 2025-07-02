"""A module containing the algorithm to calculate the information gain of new samples."""

import logging
from typing import Dict, List, Set

import pandas as pd
from pandas import DataFrame
from pddl_plus_parser.models import PDDLFunction, Predicate

from sam_learning.core.numeric_learning.numeric_utils import create_grounded_monomials
from sam_learning.core.online_learning.incremental_convex_hull_learner import IncrementalConvexHullLearner
from sam_learning.core.online_learning.online_discrete_models_learner import OnlineDiscreteModelLearner
from sam_learning.core.online_learning.online_numeric_models_learner import OnlineNumericModelLearner

LABEL_COLUMN = "label"


class InformationStatesLearner:
    """Represents a learner for informative states that integrates discrete and numeric
    representations to analyze whether a given state or transition is valid, previously
    visited, or informative.

    This class combines data from both numeric and propositional domains, applies models validations
    to determine informativeness of new samples or applicability of action
    transitions. It interacts with underlying learners for discrete and numeric models to
    manage state and perform the relevant calculations.

    :ivar logger: Logger instance for logging internal operations.
    :ivar action_name: Name of the action associated with this learner.
    :ivar combined_data: Combined data integrating both numeric and discrete observations
        with associated success/failure labels.
    :ivar numeric_data: Data structure containing numeric data samples with associated
        labels.
    :ivar discrete_model_learner: Learner handling discrete model aspects.
    :ivar convex_hull_learner: Learner managing incremental convex hull computations for
        numeric models.
    :ivar monomials: List of monomials represented by combinations of numeric variables.
    """

    logger: logging.Logger
    action_name: str
    combined_data: DataFrame
    numeric_failure_data: DataFrame
    discrete_model_learner: OnlineDiscreteModelLearner
    convex_hull_learner: IncrementalConvexHullLearner
    monomials: List[List[str]]

    def __init__(
        self,
        action_name: str,
        discrete_model_learner: OnlineDiscreteModelLearner,
        numeric_model_learner: OnlineNumericModelLearner,
    ):
        self.logger = logging.getLogger(__name__)
        self.action_name = action_name
        self.discrete_model_learner = discrete_model_learner
        self.numeric_model_learner = numeric_model_learner
        self.monomials = self.numeric_model_learner.monomials
        monomial_strs = self.numeric_model_learner.data_columns
        self.parameter_bound_predicates = [p.untyped_representation for p in self.discrete_model_learner.predicates_superset]
        self.combined_data = DataFrame(columns=[*monomial_strs, self.parameter_bound_predicates, LABEL_COLUMN])
        self.numeric_failure_data = DataFrame(columns=[*monomial_strs])

    def _create_combined_sample_data(
        self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: Set[Predicate]
    ) -> DataFrame:
        """Creates a combined sample data from the numeric and propositional samples.

        :param new_numeric_sample: the numeric part of the sample.
        :param new_propositional_sample: the propositional predicates represent the new sample.
        :return: a DataFrame containing the combined sample data.
        """
        # Create a dataframe
        grounded_monomials = create_grounded_monomials(self.monomials, new_numeric_sample)
        predicates_map = {
            **{p.untyped_representation: True for p in new_propositional_sample},
            **{
                p.untyped_representation: False
                for p in self.discrete_model_learner.predicates_superset.difference(new_propositional_sample)
            },
        }
        combined_sample_data = {
            **{k: [v] for k, v in grounded_monomials.items()},
            **{k: [v] for k, v in predicates_map.items()},
        }
        new_sample_data = DataFrame(combined_sample_data)
        return new_sample_data

    def _visited_previously_failed_execution(
        self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: Set[Predicate]
    ) -> bool:
        """Validates whether the new sample is a previously visited failed state.

        :param new_numeric_sample: the numeric part of the sample.
        :param new_propositional_sample: the discrete part of the sample.
        :return: whether the new sample is a previously visited failed state.
        """
        # Create combined sample data
        new_sample_data = self._create_combined_sample_data(new_numeric_sample, new_propositional_sample)
        new_sample_data[LABEL_COLUMN] = False
        return len(pd.concat([self.combined_data, new_sample_data], ignore_index=True).drop_duplicates()) == len(self.combined_data)

    def _is_state_not_applicable_in_safe_numeric_model(self, new_numeric_sample: Dict[str, PDDLFunction]) -> bool:
        """Determines if a given numeric sample results in a state that is not applicable
        in the context of the numeric model. This is assessed by checking if any negative
        samples from the numeric data are inside the convex hull created using the provided
        numeric sample.

        The function uses grounded monomials from the numeric sample, transforms them
        into a DataFrame, and validates whether the convex hull learner identifies negative
        samples within the constructed hull.

        :param new_numeric_sample: A dictionary where keys are string representations of
            PDDL functions, and values are the corresponding numeric values representing
            the sample.
        :return: Returns True if a negative sample is located within the constructed convex hull,
            indicating the state is not applicable. Otherwise, returns False.
        """
        validation_convex_hull = self.numeric_model_learner.copy_convex_hull_learner()
        if len(validation_convex_hull.data) <= 1:
            return False

        validation_convex_hull.add_new_point(create_grounded_monomials(self.monomials, new_numeric_sample))
        for _, negative_sample in self.numeric_failure_data.iterrows():
            data_columns = [col for col in self.numeric_failure_data.columns.tolist() if col != LABEL_COLUMN]
            negative_sample_in_hull = validation_convex_hull.is_point_in_convex_hull(negative_sample[data_columns])
            if negative_sample_in_hull:
                self.logger.debug("Found a negative sample inside the convex hull.")
                validation_convex_hull.close_convex_hull()
                return True

        self.logger.debug(f"The given sample does not create a convex hull with inapplicable points.")
        validation_convex_hull.close_convex_hull()
        return False

    def add_new_numeric_failure(self, new_numeric_sample: Dict[str, PDDLFunction]) -> None:
        """

        Args:
            new_numeric_sample:

        Returns:

        """
        grounded_monomials = create_grounded_monomials(self.monomials, new_numeric_sample)
        new_numeric_df = DataFrame({k: [v] for k, v in grounded_monomials.items()})
        if self.numeric_failure_data.empty:
            self.numeric_failure_data = new_numeric_df
            return

        self.numeric_failure_data = pd.concat([self.numeric_failure_data, new_numeric_df], ignore_index=True)
        self.numeric_failure_data.dropna(axis="columns", inplace=True)
        self.numeric_failure_data = self.numeric_failure_data.drop_duplicates()

    def add_new_sample(
        self,
        new_numeric_sample: Dict[str, PDDLFunction],
        new_propositional_sample: Set[Predicate] = frozenset(),
        is_successful: bool = True,
    ) -> None:
        """Adds a new sample to the combined observation data. The function integrates a numeric
        sample and a propositional sample into a combined data structure. The resulting data
        is then appended to the existing combined observation data, including an indication
        of whether the operation was successful.

        :param new_numeric_sample: A dictionary representing numeric PDDL functions
            as the keys and their respective values for the provided sample.
        :param new_propositional_sample: A list of propositional predicates contained
            in the sample.
        :param is_successful: Indicates if the operation to add a new sample succeeded.
            Defaults to True.
        """
        new_observation_entry = self._create_combined_sample_data(new_numeric_sample, new_propositional_sample)
        new_observation_entry[LABEL_COLUMN] = is_successful
        if self.combined_data.empty:
            self.combined_data = new_observation_entry

        else:
            self.combined_data = pd.concat([self.combined_data, new_observation_entry], ignore_index=True)
            self.combined_data.dropna(axis="columns", inplace=True)
            self.combined_data = self.combined_data.drop_duplicates()

    def is_applicable(self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: Set[Predicate]) -> bool:
        """Checks whether the action is applicable in the state.

        :param new_numeric_sample: The numeric sample to check.
        :param new_propositional_sample: The propositional predicates representing the new sample.
        :return: whether the action is applicable in the state.
        """
        return self.discrete_model_learner.is_state_in_safe_model(
            new_propositional_sample
        ) and self.numeric_model_learner.is_state_in_safe_model(new_numeric_sample)

    def is_sample_informative(self, new_numeric_sample: Dict[str, PDDLFunction], new_propositional_sample: Set[Predicate]) -> bool:
        """Checks whether the sample is informative.

        :param new_numeric_sample: The new sample to calculate whether it is informative.
        :param new_propositional_sample: the propositional predicates representing the new sample.
        :return: whether the sample is informative as well as whether the action should be applicable.
        """
        # First, check if the sample is definitely applicable and non-informative
        is_applicable_and_non_informative = self.discrete_model_learner.is_state_in_safe_model(
            new_propositional_sample
        ) and self.numeric_model_learner.is_state_in_safe_model(new_numeric_sample)
        if is_applicable_and_non_informative:
            return False

        # Otherwise, check if it's definitely not applicable - if either is not applicable, return False
        is_definitely_not_applicable = self.discrete_model_learner.is_state_not_applicable_in_safe_discrete_model(
            new_propositional_sample
        ) or self._is_state_not_applicable_in_safe_numeric_model(new_numeric_sample)

        # Return whether it's informative and whether it's applicable
        return not is_definitely_not_applicable

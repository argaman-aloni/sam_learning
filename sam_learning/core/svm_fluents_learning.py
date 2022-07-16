"""Learns fluents using SVM."""
import itertools
import logging
from typing import Dict, List, Tuple, NoReturn

import numpy
import numpy as np
import pandas as pd
from pddl_plus_parser.models import PDDLFunction
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.svm import LinearSVC
from stree import Stree

from sam_learning.core.polynomial_fluents_learning_algorithm import PolynomialFluentsLearningAlgorithm

TOLERANCE = 0.01


class SVMFluentsLearning:
    """Learns models by using linear SVM classifier and evaluating the coefficients of the fluents."""

    positive_observations: Dict[str, List[float]]
    negative_observations: Dict[str, List[float]]

    def __init__(self, action_name: str, polynomial_degree: int = 0):
        self.positive_observations = {}
        self.negative_observations = {}
        self.polynom_degree = polynomial_degree
        self.logger = logging.getLogger(__name__)
        self.oblique_tree = BUTIF(linear_model=LinearSVC(random_state=0, tol=1e-5), task="classification", max_leaf=10)
        super().__init__(action_name, polynomial_degree)

    def _create_polynomial_string_recursive(self, fluents: List[str]) -> str:
        """

        :param fluents:
        :return:
        """
        if len(fluents) == 1:
            return fluents[0]

        return f"(* {fluents[0]} {self._create_polynomial_string_recursive(fluents[1:])})"

    def _create_monomial_string(self, fluents: List[str]) -> str:
        """

        :param fluents:
        :return:
        """
        return self._create_polynomial_string_recursive(fluents)

    def _reduce_num_variables(self, input_df: pd.DataFrame, observation_class: List[int]) -> pd.DataFrame:
        """

        :param input_df:
        :param observation_class:
        :return:
        """
        feature_selection_obj = SelectPercentile(chi2, percentile=65)
        feature_selection_obj.fit(input_df, observation_class)
        selected_feature_names = feature_selection_obj.get_feature_names_out(list(self.positive_observations.keys()))
        x_new = feature_selection_obj.transform(input_df)
        return pd.DataFrame(x_new, columns=selected_feature_names)

    def _add_polynom(self, state_fluents: Dict[str, PDDLFunction], is_positive_observation: bool = True) -> NoReturn:
        """

        :param state_fluents:
        :param is_positive_observation:
        :return:
        """
        observations_store = self.positive_observations if is_positive_observation else self.negative_observations
        if self.polynom_degree == 1:
            for first_fluent, second_fluent in itertools.combinations(list(state_fluents.keys()), r=2):
                multiplied_fluent = self._create_monomial_string([first_fluent, second_fluent])
                observations_store[multiplied_fluent].append(
                    state_fluents[first_fluent].value * state_fluents[second_fluent].value)
            return

        for degree in range(2, self.polynom_degree + 1):
            for fluent_combination in itertools.combinations_with_replacement(
                    list(state_fluents.keys()), r=degree):
                polynomial_fluent = self._create_monomial_string(list(fluent_combination))
                values = [state_fluents[fluent].value for fluent in fluent_combination]
                observations_store[polynomial_fluent].append(numpy.prod(values))

    def add_to_store(self, state_fluents: Dict[str, PDDLFunction], is_positive_observation: bool = True) -> NoReturn:
        """

        :param state_fluents:
        :param is_positive_observation:
        :return:
        """
        observations_store = self.positive_observations if is_positive_observation else self.negative_observations
        for state_fluent_lifted_str, state_fluent_data in state_fluents.items():
            observations_store[state_fluent_lifted_str].append(state_fluent_data.value)

        self._add_polynom_to_storage(state_fluents, is_positive_observation)

    def convert_dataset_to_dataframe(self) -> pd.DataFrame:
        """

        :return:
        """
        self.logger.info("Converting the dataset to dataframe...")
        all_observations = {**self.positive_observations}
        for negative_observation_fluent, negative_observation_data in self.negative_observations.items():
            all_observations[negative_observation_fluent].extend(negative_observation_data)

        return pd.DataFrame(all_observations)

    def run_linear_svc(self) -> Tuple[List[float], float]:
        """

        :return:
        """
        self.logger.info("Running linear SVC model to learn the fluents coefficients...")
        labels = np.array([1] * len(list(self.positive_observations.values())[0]) +
                          [0] * len(list(self.negative_observations.values())[0]))
        features = self.convert_dataset_to_dataframe()
        self.logger.info("Training the model...")
        self.oblique_tree.fit(features, labels)

        false_positives = []
        inequalities = []
        while len(false_positives) > 0:
            svm_fluents_learning = LinearSVC(random_state=0, tol=1e-5)
            svm_fluents_learning.fit(features, labels)
            coefficients =

        return svm_fluents_learning.coef_[0], svm_fluents_learning.intercept_[0]

"""Learns fluents using SVM."""
import itertools
import logging
import pandas as pd
import numpy as np
from pddl_plus_parser.models import Observation
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, chi2
from typing import Dict, List, Union, Tuple

from sam_learning.core.polynomial_fluents_learning_algorithm import PolynomialFluentsLearningAlgorithm

TOLERANCE = 0.01


class SVMFluentsLearning(PolynomialFluentsLearningAlgorithm):
    """Learns models by using linear SVM classifier and evaluating the coefficients of the fluents."""

    positive_observations: Dict[str, List[float]]
    negative_observations: Dict[str, List[float]]

    def __init__(self, action_name: str, polynomial_degree: int = 0):
        self.positive_observations = {}
        self.negative_observations = {}
        super().__init__(action_name, polynomial_degree)

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

    def run_linear_svc(self) -> Tuple[List[float], float]:
        """

        :return:
        """
        self.logger.info("Running linear SVC model to learn the fluents coefficients...")
        labels = np.array([1] * len(self.positive_observations) + [0] * len(self.negative_observations))
        features = np.array(list(self.positive_observations.values()) + list(self.negative_observations.values()))
        false_positives = []
        inequalities = []
        while len(false_positives) > 0:
            svm_fluents_learning = LinearSVC(random_state=0, tol=1e-5)
            svm_fluents_learning.fit(features, labels)

        return svm_fluents_learning.coef_[0], svm_fluents_learning.intercept_[0]

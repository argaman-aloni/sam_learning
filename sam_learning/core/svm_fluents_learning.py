"""Learns fluents using SVM."""
from sklearn.svm import LinearSVC
from typing import Dict, List, Union


class SVMFluentsLearning:
    """Learns models by using linear SVM classifier and evaluating the coefficients of the fluents."""

    def __init__(self, positive_observations: Dict[str, List[Union[bool, float]]],
                 negative_observations: Dict[str, List[Union[bool, float]]]):
        self.positive_observations = positive_observations
        self.negative_observations = negative_observations

    # def learn_pre